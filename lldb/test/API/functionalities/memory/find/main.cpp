#include <stdio.h>
#include <stdint.h>

template <size_t T> struct [[gnu::packed]] Payload {
  uint8_t data[T];
};

using ThreeBytes = Payload<3>;
using FiveBytes = Payload<5>;
using SixBytes = Payload<5>;
using SevenBytes = Payload<7>;
using NineBytes = Payload<9>;

int main (int argc, char const *argv[])
{
    const char* stringdata = "hello world; I like to write text in const char pointers";
    uint8_t bytedata[] = {0xAA,0xBB,0xCC,0xDD,0xEE,0xFF,0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99};
    ThreeBytes b1;
    FiveBytes b2;
    SixBytes b3;
    SevenBytes b4;
    NineBytes b5;
    return 0; // break here
}
