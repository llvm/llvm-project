// fpr_sse_x86_64.core was generated with:
// ./make-core.sh fpr_sse.cpp
//
// fpr_sse_i386.core was generated with:
//   export CFLAGS=-m32
//   ./make-core.sh fpr_sse.cpp

void _start(void) {
  __asm__("fldpi;"
          "fldz;"
          "fld1;"
          "fldl2e;"
          "fldln2;"
          "fldl2t;"
          "fld1;"
          "fldlg2;");

  unsigned int values[8] = {
      0x46643129, 0x6486ed9c, 0xd71fc207, 0x254820a2,
      0xc4a85aeb, 0x0b204149, 0x4f8bf1f8, 0xcd30f113,
  };

  __asm__("vbroadcastss %0, %%xmm0;"
          "vbroadcastss %1, %%xmm1;"
          "vbroadcastss %2, %%xmm2;"
          "vbroadcastss %3, %%xmm3;"
          "vbroadcastss %4, %%xmm4;"
          "vbroadcastss %5, %%xmm5;"
          "vbroadcastss %6, %%xmm6;"
          "vbroadcastss %7, %%xmm7;"

          ::"m"(values[0]),
          "m"(values[1]), "m"(values[2]), "m"(values[3]), "m"(values[4]),
          "m"(values[5]), "m"(values[6]), "m"(values[7]));

  volatile int *a = 0;
  *a = 0;
}
