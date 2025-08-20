// Inputs/seed_reg_const_undef.c
__attribute__((noinline))
int main(int argc, char **argv) {
  int i = argc;                 // i in a reg (DW_OP_regN)
  asm volatile("" :: "r"(i));   // keep i live here
  i = 0;                        // i becomes const 0 (DW_OP_constu 0, stack_value)
  asm volatile("" :: "r"(i));   // keep the const range materialized
  return 0;                     // i ends -> <undef> after its range
}
