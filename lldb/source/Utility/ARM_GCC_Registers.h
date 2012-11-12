//===-- ARM_GCC_Registers.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef utility_ARM_GCC_Registers_h_
#define utility_ARM_GCC_Registers_h_

enum
{
    gcc_r0 = 0,
    gcc_r1,
    gcc_r2,
    gcc_r3,
    gcc_r4,
    gcc_r5,
    gcc_r6,
    gcc_r7,
    gcc_r8,
    gcc_r9,
    gcc_r10,
    gcc_r11,
    gcc_r12,
    gcc_sp,
    gcc_lr,
    gcc_pc,
    gcc_cpsr
};

enum
{
//  Name                    Nr   Rel Offset    Size  Type            Raw value
    gdb_arm_r0          =   0, //  0      0       4 int32_t
    gdb_arm_r1          =   1, //  1      4       4 int32_t
    gdb_arm_r2          =   2, //  2      8       4 int32_t
    gdb_arm_r3          =   3, //  3     12       4 int32_t
    gdb_arm_r4          =   4, //  4     16       4 int32_t
    gdb_arm_r5          =   5, //  5     20       4 int32_t
    gdb_arm_r6          =   6, //  6     24       4 int32_t
    gdb_arm_r7          =   7, //  7     28       4 int32_t
    gdb_arm_r8          =   8, //  8     32       4 int32_t
    gdb_arm_r9          =   9, //  9     36       4 int32_t
    gdb_arm_r10         =  10, // 10     40       4 int32_t
    gdb_arm_r11         =  11, // 11     44       4 int32_t
    gdb_arm_r12         =  12, // 12     48       4 int32_t
    gdb_arm_sp          =  13, // 13     52       4 int32_t
    gdb_arm_lr          =  14, // 14     56       4 int32_t
    gdb_arm_pc          =  15, // 15     60       4 int32_t
    gdb_arm_f0          =  16, // 16     64      12 _arm_ext_littlebyte_bigword
    gdb_arm_f1          =  17, // 17     76      12 _arm_ext_littlebyte_bigword
    gdb_arm_f2          =  18, // 18     88      12 _arm_ext_littlebyte_bigword
    gdb_arm_f3          =  19, // 19    100      12 _arm_ext_littlebyte_bigword
    gdb_arm_f4          =  20, // 20    112      12 _arm_ext_littlebyte_bigword
    gdb_arm_f5          =  21, // 21    124      12 _arm_ext_littlebyte_bigword
    gdb_arm_f6          =  22, // 22    136      12 _arm_ext_littlebyte_bigword
    gdb_arm_f7          =  23, // 23    148      12 _arm_ext_littlebyte_bigword
    gdb_arm_f8          =  24, // 24    160      12 _arm_ext_littlebyte_bigword
    gdb_arm_cpsr        =  25, // 25    172       4 int32_t
    gdb_arm_s0          =  26, // 26    176       4 _ieee_single_little
    gdb_arm_s1          =  27, // 27    180       4 _ieee_single_little
    gdb_arm_s2          =  28, // 28    184       4 _ieee_single_little
    gdb_arm_s3          =  29, // 29    188       4 _ieee_single_little
    gdb_arm_s4          =  30, // 30    192       4 _ieee_single_little
    gdb_arm_s5          =  31, // 31    196       4 _ieee_single_little
    gdb_arm_s6          =  32, // 32    200       4 _ieee_single_little
    gdb_arm_s7          =  33, // 33    204       4 _ieee_single_little
    gdb_arm_s8          =  34, // 34    208       4 _ieee_single_little
    gdb_arm_s9          =  35, // 35    212       4 _ieee_single_little
    gdb_arm_s10         =  36, // 36    216       4 _ieee_single_little
    gdb_arm_s11         =  37, // 37    220       4 _ieee_single_little
    gdb_arm_s12         =  38, // 38    224       4 _ieee_single_little
    gdb_arm_s13         =  39, // 39    228       4 _ieee_single_little
    gdb_arm_s14         =  40, // 40    232       4 _ieee_single_little
    gdb_arm_s15         =  41, // 41    236       4 _ieee_single_little
    gdb_arm_s16         =  42, // 42    240       4 _ieee_single_little
    gdb_arm_s17         =  43, // 43    244       4 _ieee_single_little
    gdb_arm_s18         =  44, // 44    248       4 _ieee_single_little
    gdb_arm_s19         =  45, // 45    252       4 _ieee_single_little
    gdb_arm_s20         =  46, // 46    256       4 _ieee_single_little
    gdb_arm_s21         =  47, // 47    260       4 _ieee_single_little
    gdb_arm_s22         =  48, // 48    264       4 _ieee_single_little
    gdb_arm_s23         =  49, // 49    268       4 _ieee_single_little
    gdb_arm_s24         =  50, // 50    272       4 _ieee_single_little
    gdb_arm_s25         =  51, // 51    276       4 _ieee_single_little
    gdb_arm_s26         =  52, // 52    280       4 _ieee_single_little
    gdb_arm_s27         =  53, // 53    284       4 _ieee_single_little
    gdb_arm_s28         =  54, // 54    288       4 _ieee_single_little
    gdb_arm_s29         =  55, // 55    292       4 _ieee_single_little
    gdb_arm_s30         =  56, // 56    296       4 _ieee_single_little
    gdb_arm_s31         =  57, // 57    300       4 _ieee_single_little
    gdb_arm_fpscr       =  58, // 58    304       4 int32_t
    gdb_arm_d16         =  59, // 59    308       8 _ieee_double_little
    gdb_arm_d17         =  60, // 60    316       8 _ieee_double_little
    gdb_arm_d18         =  61, // 61    324       8 _ieee_double_little
    gdb_arm_d19         =  62, // 62    332       8 _ieee_double_little
    gdb_arm_d20         =  63, // 63    340       8 _ieee_double_little
    gdb_arm_d21         =  64, // 64    348       8 _ieee_double_little
    gdb_arm_d22         =  65, // 65    356       8 _ieee_double_little
    gdb_arm_d23         =  66, // 66    364       8 _ieee_double_little
    gdb_arm_d24         =  67, // 67    372       8 _ieee_double_little
    gdb_arm_d25         =  68, // 68    380       8 _ieee_double_little
    gdb_arm_d26         =  69, // 69    388       8 _ieee_double_little
    gdb_arm_d27         =  70, // 70    396       8 _ieee_double_little
    gdb_arm_d28         =  71, // 71    404       8 _ieee_double_little
    gdb_arm_d29         =  72, // 72    412       8 _ieee_double_little
    gdb_arm_d30         =  73, // 73    420       8 _ieee_double_little
    gdb_arm_d31         =  74, // 74    428       8 _ieee_double_little
    gdb_arm_d0          =  75, //  0    436       8 _ieee_double_little
    gdb_arm_d1          =  76, //  1    444       8 _ieee_double_little
    gdb_arm_d2          =  77, //  2    452       8 _ieee_double_little
    gdb_arm_d3          =  78, //  3    460       8 _ieee_double_little
    gdb_arm_d4          =  79, //  4    468       8 _ieee_double_little
    gdb_arm_d5          =  80, //  5    476       8 _ieee_double_little
    gdb_arm_d6          =  81, //  6    484       8 _ieee_double_little
    gdb_arm_d7          =  82, //  7    492       8 _ieee_double_little
    gdb_arm_d8          =  83, //  8    500       8 _ieee_double_little
    gdb_arm_d9          =  84, //  9    508       8 _ieee_double_little
    gdb_arm_d10         =  85, // 10    516       8 _ieee_double_little
    gdb_arm_d11         =  86, // 11    524       8 _ieee_double_little
    gdb_arm_d12         =  87, // 12    532       8 _ieee_double_little
    gdb_arm_d13         =  88, // 13    540       8 _ieee_double_little
    gdb_arm_d14         =  89, // 14    548       8 _ieee_double_little
    gdb_arm_d15         =  90, // 15    556       8 _ieee_double_little
    gdb_arm_q0          =  91, // 16    564      16 _vec128
    gdb_arm_q1          =  92, // 17    580      16 _vec128
    gdb_arm_q2          =  93, // 18    596      16 _vec128
    gdb_arm_q3          =  94, // 19    612      16 _vec128
    gdb_arm_q4          =  95, // 20    628      16 _vec128
    gdb_arm_q5          =  96, // 21    644      16 _vec128
    gdb_arm_q6          =  97, // 22    660      16 _vec128
    gdb_arm_q7          =  98, // 23    676      16 _vec128
    gdb_arm_q8          =  99, // 24    692      16 _vec128
    gdb_arm_q9          = 100, // 25    708      16 _vec128
    gdb_arm_q10         = 101, // 26    724      16 _vec128
    gdb_arm_q11         = 102, // 27    740      16 _vec128
    gdb_arm_q12         = 103, // 28    756      16 _vec128
    gdb_arm_q13         = 104, // 29    772      16 _vec128
    gdb_arm_q14         = 105, // 30    788      16 _vec128
    gdb_arm_q15         = 106  // 31    804      16 _vec128
};
#endif // utility_ARM_GCC_Registers_h_

