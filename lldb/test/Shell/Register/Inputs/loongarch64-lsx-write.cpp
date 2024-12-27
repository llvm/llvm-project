#include <cinttypes>
#include <cstdint>
#include <cstdio>

union alignas(16) lsx_t {
  uint64_t as_uint64[2];
  uint8_t as_uint8[16];
};

int main() {
  lsx_t lsx[32] = {0};

  asm volatile("vrepli.b $vr0, 0\n\t"
               "vrepli.b $vr1, 0\n\t"
               "vrepli.b $vr2, 0\n\t"
               "vrepli.b $vr3, 0\n\t"
               "vrepli.b $vr4, 0\n\t"
               "vrepli.b $vr5, 0\n\t"
               "vrepli.b $vr6, 0\n\t"
               "vrepli.b $vr7, 0\n\t"
               "vrepli.b $vr8, 0\n\t"
               "vrepli.b $vr9, 0\n\t"
               "vrepli.b $vr10, 0\n\t"
               "vrepli.b $vr11, 0\n\t"
               "vrepli.b $vr12, 0\n\t"
               "vrepli.b $vr13, 0\n\t"
               "vrepli.b $vr14, 0\n\t"
               "vrepli.b $vr15, 0\n\t"
               "vrepli.b $vr16, 0\n\t"
               "vrepli.b $vr17, 0\n\t"
               "vrepli.b $vr18, 0\n\t"
               "vrepli.b $vr19, 0\n\t"
               "vrepli.b $vr20, 0\n\t"
               "vrepli.b $vr21, 0\n\t"
               "vrepli.b $vr22, 0\n\t"
               "vrepli.b $vr23, 0\n\t"
               "vrepli.b $vr24, 0\n\t"
               "vrepli.b $vr25, 0\n\t"
               "vrepli.b $vr26, 0\n\t"
               "vrepli.b $vr27, 0\n\t"
               "vrepli.b $vr28, 0\n\t"
               "vrepli.b $vr29, 0\n\t"
               "vrepli.b $vr30, 0\n\t"
               "vrepli.b $vr31, 0\n\t"
               "break 5\n\t"
               "move $t0, %0\n\t"
               "vst $vr0, $t0, 0\n\t"
               "vst $vr1, $t0, 16\n\t"
               "vst $vr2, $t0, 32\n\t"
               "vst $vr3, $t0, 48\n\t"
               "vst $vr4, $t0, 64\n\t"
               "vst $vr5, $t0, 80\n\t"
               "vst $vr6, $t0, 96\n\t"
               "vst $vr7, $t0, 112\n\t"
               "vst $vr8, $t0, 128\n\t"
               "vst $vr9, $t0, 144\n\t"
               "vst $vr10, $t0, 160\n\t"
               "vst $vr11, $t0, 176\n\t"
               "vst $vr12, $t0, 192\n\t"
               "vst $vr13, $t0, 208\n\t"
               "vst $vr14, $t0, 224\n\t"
               "vst $vr15, $t0, 240\n\t"
               "vst $vr16, $t0, 256\n\t"
               "vst $vr17, $t0, 272\n\t"
               "vst $vr18, $t0, 288\n\t"
               "vst $vr19, $t0, 304\n\t"
               "vst $vr20, $t0, 320\n\t"
               "vst $vr21, $t0, 336\n\t"
               "vst $vr22, $t0, 352\n\t"
               "vst $vr23, $t0, 368\n\t"
               "vst $vr24, $t0, 384\n\t"
               "vst $vr25, $t0, 400\n\t"
               "vst $vr26, $t0, 416\n\t"
               "vst $vr27, $t0, 432\n\t"
               "vst $vr28, $t0, 448\n\t"
               "vst $vr29, $t0, 464\n\t"
               "vst $vr30, $t0, 480\n\t"
               "vst $vr31, $t0, 496\n\t" ::"r"(&lsx)
               : "$t0", "$vr24", "$vr25", "$vr26", "$vr27", "$vr28", "$vr29", "$vr30", "$vr31");

  for (int i = 0; i < 32; ++i) {
    printf("vr%d = { ", i);
    for (int j = 0; j < sizeof(lsx->as_uint8); ++j)
      printf("0x%02x ", lsx[i].as_uint8[j]);
    printf("}\n");
  }

  return 0;
}
