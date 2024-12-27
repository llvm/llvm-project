#include <cstdint>

int main() {
  asm volatile("xvrepli.d $xr0, 16\n\t"
               "xvrepli.w $xr1, 16\n\t"
               "xvrepli.h $xr2, 16\n\t"
               "xvrepli.b $xr3, 16\n\t"
               "xvrepli.d $xr4, 15\n\t"
               "xvrepli.w $xr5, 15\n\t"
               "xvrepli.h $xr6, 15\n\t"
               "xvrepli.b $xr7, 15\n\t"
               "xvrepli.d $xr8, 14\n\t"
               "xvrepli.w $xr9, 14\n\t"
               "xvrepli.h $xr10, 14\n\t"
               "xvrepli.b $xr11, 14\n\t"
               "xvrepli.d $xr12, 13\n\t"
               "xvrepli.w $xr13, 13\n\t"
               "xvrepli.h $xr14, 13\n\t"
               "xvrepli.b $xr15, 13\n\t"
               "xvrepli.d $xr16, 12\n\t"
               "xvrepli.w $xr17, 12\n\t"
               "xvrepli.h $xr18, 12\n\t"
               "xvrepli.b $xr19, 12\n\t"
               "xvrepli.d $xr20, 11\n\t"
               "xvrepli.w $xr21, 11\n\t"
               "xvrepli.h $xr22, 11\n\t"
               "xvrepli.b $xr23, 11\n\t"
               "xvrepli.d $xr24, 10\n\t"
               "xvrepli.w $xr25, 10\n\t"
               "xvrepli.h $xr26, 10\n\t"
               "xvrepli.b $xr27, 10\n\t"
               "xvrepli.d $xr28, 9\n\t"
               "xvrepli.w $xr29, 9\n\t"
               "xvrepli.h $xr30, 9\n\t"
               "xvrepli.b $xr31, 9\n\t"
               "nop\n\t"
               "break 5\n\t" ::
                   : "$xr24", "$xr25", "$xr26", "$xr27", "$xr28", "$xr29", "$xr30", "$xr31");

  return 0;
}
