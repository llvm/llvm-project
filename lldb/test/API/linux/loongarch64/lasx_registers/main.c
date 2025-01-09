#include <stdint.h>

// base is added to each value. If base = 2, then
// $vr0 = { 0x02 } * 32
// $vr1 = { 0x03 } * 32 etc.
void write_lasx_regs(unsigned base) {
#define WRITE_LASX(NUM)                                                        \
  asm volatile("xvreplgr2vr.b $xr" #NUM ", %0\n\t" ::"r"(base + NUM))
  WRITE_LASX(0);
  WRITE_LASX(1);
  WRITE_LASX(2);
  WRITE_LASX(3);
  WRITE_LASX(4);
  WRITE_LASX(5);
  WRITE_LASX(6);
  WRITE_LASX(7);
  WRITE_LASX(8);
  WRITE_LASX(9);
  WRITE_LASX(10);
  WRITE_LASX(11);
  WRITE_LASX(12);
  WRITE_LASX(13);
  WRITE_LASX(14);
  WRITE_LASX(15);
  WRITE_LASX(16);
  WRITE_LASX(17);
  WRITE_LASX(18);
  WRITE_LASX(19);
  WRITE_LASX(20);
  WRITE_LASX(21);
  WRITE_LASX(22);
  WRITE_LASX(23);
  WRITE_LASX(24);
  WRITE_LASX(25);
  WRITE_LASX(26);
  WRITE_LASX(27);
  WRITE_LASX(28);
  WRITE_LASX(29);
  WRITE_LASX(30);
  WRITE_LASX(31);
}

unsigned verify_lasx_regs() {
  uint8_t lasx_reg[32];
  uint8_t target = 0;

#define VERIFY_LASX(NUM)                                                       \
  do {                                                                         \
    for (int i = 0; i < 32; ++i)                                               \
      lasx_reg[i] = 0;                                                         \
    asm volatile("xvst $xr" #NUM ", %0\n\t" ::"m"(lasx_reg));                  \
    target = NUM + 1;                                                          \
    for (int i = 0; i < 32; ++i)                                               \
      if (lasx_reg[i] != target)                                               \
        return 1;                                                              \
  } while (0)

  VERIFY_LASX(0);
  VERIFY_LASX(1);
  VERIFY_LASX(2);
  VERIFY_LASX(3);
  VERIFY_LASX(4);
  VERIFY_LASX(5);
  VERIFY_LASX(6);
  VERIFY_LASX(7);
  VERIFY_LASX(8);
  VERIFY_LASX(9);
  VERIFY_LASX(10);
  VERIFY_LASX(11);
  VERIFY_LASX(12);
  VERIFY_LASX(13);
  VERIFY_LASX(14);
  VERIFY_LASX(15);
  VERIFY_LASX(16);
  VERIFY_LASX(17);
  VERIFY_LASX(18);
  VERIFY_LASX(19);
  VERIFY_LASX(20);
  VERIFY_LASX(21);
  VERIFY_LASX(22);
  VERIFY_LASX(23);
  VERIFY_LASX(24);
  VERIFY_LASX(25);
  VERIFY_LASX(26);
  VERIFY_LASX(27);
  VERIFY_LASX(28);
  VERIFY_LASX(29);
  VERIFY_LASX(30);
  VERIFY_LASX(31);

  return 0;
}
int main(int argc, char *argv[]) {
  write_lasx_regs(0);

  return verify_lasx_regs(); // Set break point at this line.
}
