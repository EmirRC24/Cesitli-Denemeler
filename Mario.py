# XOX oyun tahtasını gösteren fonksiyon
def print_board(board):
    print("\n")
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

# Kazananı kontrol eden fonksiyon
def check_winner(board, player):
    # Satırları kontrol et
    for row in board:
        if all([cell == player for cell in row]):
            return True

    # Sütunları kontrol et
    for col in range(3):
        if all([board[row][col] == player for row in range(3)]):
            return True

    # Çaprazları kontrol et
    if all([board[i][i] == player for i in range(3)]) or all([board[i][2 - i] == player for i in range(3)]):
        return True

    return False

# Boş yer olup olmadığını kontrol eden fonksiyon
def check_draw(board):
    for row in board:
        if any([cell == " " for cell in row]):
            return False
    return True

# Oyuncu hamlesi yapma fonksiyonu
def make_move(board, player):
    while True:
        try:
            move = input(f"{player}'nin hamlesi (satır sütun: 1 1, 1 2, 2 2 gibi): ")
            row, col = map(int, move.split())
            if board[row - 1][col - 1] == " ":
                board[row - 1][col - 1] = player
                break
            else:
                print("Bu kare dolu! Başka bir kare seçin.")
        except (ValueError, IndexError):
            print("Geçersiz giriş! Lütfen geçerli bir satır ve sütun seçin.")

# XOX oyunu ana döngüsü
def play_game():
    board = [[" " for _ in range(3)] for _ in range(3)]
    current_player = "X"

    print("XOX Oyununa Hoşgeldiniz!")
    print_board(board)

    while True:
        make_move(board, current_player)
        print_board(board)

        if check_winner(board, current_player):
            print(f"Tebrikler! {current_player} kazandı!")
            break

        if check_draw(board):
            print("Oyun berabere!")
            break

        # Sıradaki oyuncuya geçiş
        current_player = "O" if current_player == "X" else "X"

# Oyunu başlat
if __name__ == "__main__":
    play_game()



