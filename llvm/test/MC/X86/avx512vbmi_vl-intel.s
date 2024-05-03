// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK:  vpmultishiftqb xmm1, xmm2, qword ptr [rcx]{1to2}
// CHECK:  encoding: [0x62,0xf2,0xed,0x18,0x83,0x09]
          vpmultishiftqb xmm1, xmm2, qword ptr [rcx]{1to2}

// CHECK:  vpmultishiftqb xmm1 {k1}, xmm2, qword ptr [rcx]{1to2}
// CHECK:  encoding: [0x62,0xf2,0xed,0x19,0x83,0x09]
          vpmultishiftqb xmm1 {k1}, xmm2, qword ptr [rcx]{1to2}

// CHECK:  vpmultishiftqb xmm1 {k1} {z}, xmm2, qword ptr [rcx]{1to2}
// CHECK:  encoding: [0x62,0xf2,0xed,0x99,0x83,0x09]
          vpmultishiftqb xmm1 {k1} {z}, xmm2, qword ptr [rcx]{1to2}

// CHECK:  vpmultishiftqb ymm1, ymm2, qword ptr [rcx]{1to4}
// CHECK:  encoding: [0x62,0xf2,0xed,0x38,0x83,0x09]
          vpmultishiftqb ymm1, ymm2, qword ptr [rcx]{1to4}

// CHECK:  vpmultishiftqb ymm1 {k1}, ymm2, qword ptr [rcx]{1to4}
// CHECK:  encoding: [0x62,0xf2,0xed,0x39,0x83,0x09]
          vpmultishiftqb ymm1 {k1}, ymm2, qword ptr [rcx]{1to4}

// CHECK:  vpmultishiftqb ymm1 {k1} {z}, ymm2, qword ptr [rcx]{1to4}
// CHECK:  encoding: [0x62,0xf2,0xed,0xb9,0x83,0x09]
          vpmultishiftqb ymm1 {k1} {z}, ymm2, qword ptr [rcx]{1to4}

// CHECK:  vpmultishiftqb zmm1, zmm2, qword ptr [rcx]{1to8}
// CHECK:  encoding: [0x62,0xf2,0xed,0x58,0x83,0x09]
          vpmultishiftqb zmm1, zmm2, qword ptr [rcx]{1to8}

// CHECK:  vpmultishiftqb zmm1 {k1}, zmm2, qword ptr [rcx]{1to8}
// CHECK:  encoding: [0x62,0xf2,0xed,0x59,0x83,0x09]
          vpmultishiftqb zmm1 {k1}, zmm2, qword ptr [rcx]{1to8}

// CHECK:  vpmultishiftqb zmm1 {k1} {z}, zmm2, qword ptr [rcx]{1to8}
// CHECK:  encoding: [0x62,0xf2,0xed,0xd9,0x83,0x09]
          vpmultishiftqb zmm1 {k1} {z}, zmm2, qword ptr [rcx]{1to8}
