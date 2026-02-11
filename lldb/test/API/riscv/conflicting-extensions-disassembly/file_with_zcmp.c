void function_with_zcmp_extension() {
  asm volatile("cm.push {ra, s0-s2}, -32\n\t"
               "cm.mva01s s0, s1\n\t"
               "cm.mvsa01 s0, s1\n\t"
               "cm.popret {ra, s0-s2}, 32\n\t");
}
