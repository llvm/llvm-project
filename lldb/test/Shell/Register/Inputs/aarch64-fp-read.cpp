int main() {
  asm volatile(
    "fmov     d0,  #0.5\n\t"
    "fmov     d1,  #1.5\n\t"
    "fmov     d2,  #2.5\n\t"
    "fmov     d3,  #3.5\n\t"
    "fmov     s4,  #4.5\n\t"
    "fmov     s5,  #5.5\n\t"
    "fmov     s6,  #6.5\n\t"
    "fmov     s7,  #7.5\n\t"
    "\n\t"
    "brk      #0\n\t"
    :
    :
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
  );

  return 0;
}
