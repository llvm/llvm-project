#include <stdint.h>
uint32_t f1(uint32_t x, uint32_t y) { if((x & y)==y) return x - y; return 0; }
uint32_t f2(uint32_t x, uint32_t y) { if((x & y)==y) return x ^ y; return 0; }
uint32_t f3(uint32_t x, uint32_t y) { if((x & y)==y) return x & ~y; return 0; }
