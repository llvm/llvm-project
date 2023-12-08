#include <stdint.h>

union S
{
    int32_t n;     // occupies 4 bytes
    uint16_t s[2]; // occupies 4 bytes
    uint8_t c;     // occupies 1 byte
};                 // the whole union occupies 4 bytes

int main()
{
  union S u;

  u.s[0] = 1234;
  u.s[1] = 4321;

  return 0; // Break here
}
