import pygame
import sys
import random

# Pygame'i başlat
pygame.init()

# Ekran boyutu
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption('Python Öğreten Oyun - Veri Tipleri Bulmacası')

# FPS ayarlaması
clock = pygame.time.Clock()

# Karakterin başlangıç pozisyonu ve hızı
x, y = 100, 100
speed = 5

# Veri tipi bulmacası
bulmaca_sorusu = "Bu kapıyı açmak için bir sayı girin: "
veri_tipi_dogru_mu = False

# Ana oyun döngüsü
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Tuşlarla karakter hareketi
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        x -= speed
    if keys[pygame.K_RIGHT]:
        x += speed
    if keys[pygame.K_UP]:
        y -= speed
    if keys[pygame.K_DOWN]:
        y += speed

    # Ekranı beyaza boyama
    screen.fill((255, 255, 255))

    # Karakteri ekrana çiz (basit bir mavi kare ile)
    pygame.draw.rect(screen, (0, 0, 255), (x, y, 50, 50))

    # Veri tipi bulmacası ekranı
    font = pygame.font.Font(None, 36)
    soru_text = font.render(bulmaca_sorusu, True, (0, 0, 0))
    screen.blit(soru_text, (150, 250))

    # Veri tipi seçimi (örnek bir sayı girme)
    if keys[pygame.K_1]:
        veri_tipi_cevap = 1  # Oyuncu 1'e bastığında bir integer seçiyor
        if isinstance(veri_tipi_cevap, int):
            veri_tipi_dogru_mu = True

    # Cevap doğruysa, başarı mesajı
    if veri_tipi_dogru_mu:
        dogru_text = font.render("Tebrikler! Doğru veri tipini seçtiniz!", True, (0, 255, 0))
        screen.blit(dogru_text, (150, 300))

    # Ekranı güncelle
    pygame.display.flip()

    # FPS limiti
    clock.tick(60)

pygame.quit()
sys.exit()
