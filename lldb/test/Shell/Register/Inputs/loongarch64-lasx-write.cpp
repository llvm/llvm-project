#include <cinttypes>
#include <cstdint>
#include <cstdio>

union alignas(32) lasx_t {
  uint64_t as_uint64[4];
  uint8_t as_uint8[32];
};

int main() {
  lasx_t lasx[32] = {0};

  asm volatile("xvrepli.b $xr0, 0\n\t"
               "xvrepli.b $xr1, 0\n\t"
               "xvrepli.b $xr2, 0\n\t"
               "xvrepli.b $xr3, 0\n\t"
               "xvrepli.b $xr4, 0\n\t"
               "xvrepli.b $xr5, 0\n\t"
               "xvrepli.b $xr6, 0\n\t"
               "xvrepli.b $xr7, 0\n\t"
               "xvrepli.b $xr8, 0\n\t"
               "xvrepli.b $xr9, 0\n\t"
               "xvrepli.b $xr10, 0\n\t"
               "xvrepli.b $xr11, 0\n\t"
               "xvrepli.b $xr12, 0\n\t"
               "xvrepli.b $xr13, 0\n\t"
               "xvrepli.b $xr14, 0\n\t"
               "xvrepli.b $xr15, 0\n\t"
               "xvrepli.b $xr16, 0\n\t"
               "xvrepli.b $xr17, 0\n\t"
               "xvrepli.b $xr18, 0\n\t"
               "xvrepli.b $xr19, 0\n\t"
               "xvrepli.b $xr20, 0\n\t"
               "xvrepli.b $xr21, 0\n\t"
               "xvrepli.b $xr22, 0\n\t"
               "xvrepli.b $xr23, 0\n\t"
               "xvrepli.b $xr24, 0\n\t"
               "xvrepli.b $xr25, 0\n\t"
               "xvrepli.b $xr26, 0\n\t"
               "xvrepli.b $xr27, 0\n\t"
               "xvrepli.b $xr28, 0\n\t"
               "xvrepli.b $xr29, 0\n\t"
               "xvrepli.b $xr30, 0\n\t"
               "xvrepli.b $xr31, 0\n\t"
               "break 5\n\t"
               "move $t0, %0\n\t"
               "xvst $xr0, $t0, 0\n\t"
               "xvst $xr1, $t0, 32\n\t"
               "xvst $xr2, $t0, 64\n\t"
               "xvst $xr3, $t0, 96\n\t"
               "xvst $xr4, $t0, 128\n\t"
               "xvst $xr5, $t0, 160\n\t"
               "xvst $xr6, $t0, 192\n\t"
               "xvst $xr7, $t0, 224\n\t"
               "xvst $xr8, $t0, 256\n\t"
               "xvst $xr9, $t0, 288\n\t"
               "xvst $xr10, $t0, 320\n\t"
               "xvst $xr11, $t0, 352\n\t"
               "xvst $xr12, $t0, 384\n\t"
               "xvst $xr13, $t0, 416\n\t"
               "xvst $xr14, $t0, 448\n\t"
               "xvst $xr15, $t0, 480\n\t"
               "xvst $xr16, $t0, 512\n\t"
               "xvst $xr17, $t0, 544\n\t"
               "xvst $xr18, $t0, 576\n\t"
               "xvst $xr19, $t0, 608\n\t"
               "xvst $xr20, $t0, 640\n\t"
               "xvst $xr21, $t0, 672\n\t"
               "xvst $xr22, $t0, 704\n\t"
               "xvst $xr23, $t0, 736\n\t"
               "xvst $xr24, $t0, 768\n\t"
               "xvst $xr25, $t0, 800\n\t"
               "xvst $xr26, $t0, 832\n\t"
               "xvst $xr27, $t0, 864\n\t"
               "xvst $xr28, $t0, 896\n\t"
               "xvst $xr29, $t0, 928\n\t"
               "xvst $xr30, $t0, 960\n\t"
               "xvst $xr31, $t0, 992\n\t" ::"r"(&lasx)
               : "$t0", "$xr24", "$xr25", "$xr26", "$xr27", "$xr28", "$xr29", "$xr30", "$xr31");

  for (int i = 0; i < 32; ++i) {
    printf("xr%d = { ", i);
    for (int j = 0; j < sizeof(lasx->as_uint8); ++j)
      printf("0x%02x ", lasx[i].as_uint8[j]);
    printf("}\n");
  }

  return 0;
}
