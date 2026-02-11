void function_with_zcd_instructions() {
  asm volatile("c.fsdsp ft0, 0(sp)\n\t"
               "c.fsdsp ft1, 8(sp)\n\t"
               "c.fsdsp fa0, 16(sp)\n\t"
               "c.fsdsp fa1, 24(sp)\n\t");
}
