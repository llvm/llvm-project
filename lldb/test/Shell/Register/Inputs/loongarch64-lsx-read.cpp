#include <cstdint>

int main() {
  asm volatile("vrepli.d $vr0, 1\n\t"
               "vrepli.w $vr1, 1\n\t"
               "vrepli.h $vr2, 1\n\t"
               "vrepli.b $vr3, 1\n\t"
               "vrepli.d $vr4, 2\n\t"
               "vrepli.w $vr5, 2\n\t"
               "vrepli.h $vr6, 2\n\t"
               "vrepli.b $vr7, 2\n\t"
               "vrepli.d $vr8, 3\n\t"
               "vrepli.w $vr9, 3\n\t"
               "vrepli.h $vr10, 3\n\t"
               "vrepli.b $vr11, 3\n\t"
               "vrepli.d $vr12, 4\n\t"
               "vrepli.w $vr13, 4\n\t"
               "vrepli.h $vr14, 4\n\t"
               "vrepli.b $vr15, 4\n\t"
               "vrepli.d $vr16, 5\n\t"
               "vrepli.w $vr17, 5\n\t"
               "vrepli.h $vr18, 5\n\t"
               "vrepli.b $vr19, 5\n\t"
               "vrepli.d $vr20, 6\n\t"
               "vrepli.w $vr21, 6\n\t"
               "vrepli.h $vr22, 6\n\t"
               "vrepli.b $vr23, 6\n\t"
               "vrepli.d $vr24, 7\n\t"
               "vrepli.w $vr25, 7\n\t"
               "vrepli.h $vr26, 7\n\t"
               "vrepli.b $vr27, 7\n\t"
               "vrepli.d $vr28, 8\n\t"
               "vrepli.w $vr29, 8\n\t"
               "vrepli.h $vr30, 8\n\t"
               "vrepli.b $vr31, 8\n\t"
               "nop\n\t"
               "break 5\n\t" ::
                   : "$vr24", "$vr25", "$vr26", "$vr27", "$vr28", "$vr29", "$vr30", "$vr31");

  return 0;
}
