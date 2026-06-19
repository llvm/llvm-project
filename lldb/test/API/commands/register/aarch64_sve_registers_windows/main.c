#include <arm_sve.h>
#include <stdlib.h>

#define NUM_Z_REGS 32
#define NUM_P_REGS 16

void setup_z_and_p_regs() {
  uint8_t *zregs;       // memory for z registers
  uint8_t *pregs;       // memory for p registers
  uint8_t *reg_pointer; // pointer to current register memory location
  int i, j;
  int z_size = svcntb();
  int p_size = z_size / 8;

  zregs = malloc(NUM_Z_REGS * z_size);
  pregs = malloc(NUM_P_REGS * p_size);

  // Load 'zregs' with a byte pattern consisting of register number followed by
  // 0-(up to)7f depending on vector size:
  // 00 00 00 01 00 02 00 03 ... 00 7f
  // 01 00 01 01 01 02 01 03 ... 01 7f
  // ...
  // 31 00 31 01 31 02 31 03 ... 31 7f
  for (i = 0; i < NUM_Z_REGS; ++i) {
    for (j = 0; j < z_size; j += 2) {
      zregs[i * z_size + j] = i;
      zregs[i * z_size + j + 1] = j / 2;
    }
  }

  // Load 'pregs' with a byte pattern consisting of register number followed by
  // 0-(up to)7f depending on vector size:
  // 00 00 00 01 00 02 00 03 ... 00 7f
  // 01 00 01 01 01 02 01 03 ... 01 7f
  // ...
  // 31 00 31 01 31 02 31 03 ... 31 7f
  for (i = 0; i < NUM_P_REGS; ++i) {
    for (j = 0; j < p_size; j += 2) {
      pregs[i * p_size + j] = i;
      pregs[i * p_size + j + 1] = j / 2;
    }
  }

  // Copy values from memory to Z registers, using 'ldr'.
  reg_pointer = zregs;
  asm("ldr z0, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z1, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z2, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z3, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z4, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z5, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z6, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z7, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z8, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z9, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z10, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z11, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z12, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z13, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z14, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z15, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z16, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z17, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z18, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z19, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z20, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z21, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z22, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z23, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z24, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z25, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z26, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z27, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z28, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z29, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z30, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += z_size;
  asm("ldr z31, [%0]" ::"r"(reg_pointer) :);

  // Copy values from memory to P registers, using 'ldr'.
  reg_pointer = pregs;
  asm("ldr p0, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += p_size;
  asm("ldr p1, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += p_size;
  asm("ldr p2, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += p_size;
  asm("ldr p3, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += p_size;
  asm("ldr p4, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += p_size;
  asm("ldr p5, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += p_size;
  asm("ldr p6, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += p_size;
  asm("ldr p7, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += p_size;
  asm("ldr p8, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += p_size;
  asm("ldr p9, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += p_size;
  asm("ldr p10, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += p_size;
  asm("ldr p11, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += p_size;
  asm("ldr p12, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += p_size;
  asm("ldr p13, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += p_size;
  asm("ldr p14, [%0]" ::"r"(reg_pointer) :);
  reg_pointer += p_size;
  asm("ldr p15, [%0]" ::"r"(reg_pointer) :);

  asm("setffr");
}

int main() {
  // Set up Z and P registers.
  setup_z_and_p_regs();

  return 0; // breakpoint 1
}
