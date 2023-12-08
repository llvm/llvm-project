int main() {
  asm volatile(
    "vmov.f64 d0,  #0.5\n\t"
    "vmov.f64 d1,  #1.5\n\t"
    "vmov.f64 d2,  #2.5\n\t"
    "vmov.f64 d3,  #3.5\n\t"
    "vmov.f32 s8,  #4.5\n\t"
    "vmov.f32 s9,  #5.5\n\t"
    "vmov.f32 s10, #6.5\n\t"
    "vmov.f32 s11, #7.5\n\t"
    "\n\t"
    "bkpt     #0\n\t"
    :
    :
    : "d0", "d1", "d2", "d3", "s8", "s9", "s10", "s11"
  );

  return 0;
}
