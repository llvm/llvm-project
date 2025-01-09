#include <stdint.h>

// base is added to each value. If base = 2, then
// $vr0 = { 0x02 } * 16
// $vr1 = { 0x03 } * 16 etc.
void write_lsx_regs(unsigned base) {
#define WRITE_LSX(NUM)                                                         \
  asm volatile("vreplgr2vr.b $vr" #NUM ", %0\n\t" ::"r"(base + NUM))
  WRITE_LSX(0);
  WRITE_LSX(1);
  WRITE_LSX(2);
  WRITE_LSX(3);
  WRITE_LSX(4);
  WRITE_LSX(5);
  WRITE_LSX(6);
  WRITE_LSX(7);
  WRITE_LSX(8);
  WRITE_LSX(9);
  WRITE_LSX(10);
  WRITE_LSX(11);
  WRITE_LSX(12);
  WRITE_LSX(13);
  WRITE_LSX(14);
  WRITE_LSX(15);
  WRITE_LSX(16);
  WRITE_LSX(17);
  WRITE_LSX(18);
  WRITE_LSX(19);
  WRITE_LSX(20);
  WRITE_LSX(21);
  WRITE_LSX(22);
  WRITE_LSX(23);
  WRITE_LSX(24);
  WRITE_LSX(25);
  WRITE_LSX(26);
  WRITE_LSX(27);
  WRITE_LSX(28);
  WRITE_LSX(29);
  WRITE_LSX(30);
  WRITE_LSX(31);
}

unsigned verify_lsx_regs() {
  uint8_t lsx_reg[16];
  uint8_t target = 0;

#define VERIFY_LSX(NUM)                                                        \
  do {                                                                         \
    for (int i = 0; i < 16; ++i)                                               \
      lsx_reg[i] = 0;                                                          \
    asm volatile("vst $vr" #NUM ", %0\n\t" ::"m"(lsx_reg));                    \
    target = NUM + 1;                                                          \
    for (int i = 0; i < 16; ++i)                                               \
      if (lsx_reg[i] != target)                                                \
        return 1;                                                              \
  } while (0)

  VERIFY_LSX(0);
  VERIFY_LSX(1);
  VERIFY_LSX(2);
  VERIFY_LSX(3);
  VERIFY_LSX(4);
  VERIFY_LSX(5);
  VERIFY_LSX(6);
  VERIFY_LSX(7);
  VERIFY_LSX(8);
  VERIFY_LSX(9);
  VERIFY_LSX(10);
  VERIFY_LSX(11);
  VERIFY_LSX(12);
  VERIFY_LSX(13);
  VERIFY_LSX(14);
  VERIFY_LSX(15);
  VERIFY_LSX(16);
  VERIFY_LSX(17);
  VERIFY_LSX(18);
  VERIFY_LSX(19);
  VERIFY_LSX(20);
  VERIFY_LSX(21);
  VERIFY_LSX(22);
  VERIFY_LSX(23);
  VERIFY_LSX(24);
  VERIFY_LSX(25);
  VERIFY_LSX(26);
  VERIFY_LSX(27);
  VERIFY_LSX(28);
  VERIFY_LSX(29);
  VERIFY_LSX(30);
  VERIFY_LSX(31);

  return 0;
}
int main(int argc, char *argv[]) {
  write_lsx_regs(0);

  return verify_lsx_regs(); // Set break point at this line.
}
