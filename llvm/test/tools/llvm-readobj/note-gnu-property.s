// REQUIRES: x86-registered-target
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t
// RUN: llvm-readobj -elf-output-style GNU --notes %t | FileCheck %s --check-prefix=GNU
// RUN: llvm-readobj -elf-output-style LLVM --notes %t | FileCheck %s --check-prefix=LLVM

// GNU:      Displaying notes found at file offset 0x00000040 with length 0x000000f8:
// GNU-NEXT:   Owner                 Data size       Description
// GNU-NEXT:   GNU                   0x000000e8      NT_GNU_PROPERTY_TYPE_0 (property note)
// GNU-NEXT:     Properties:  stack size: 0x100
// GNU-NEXT:     stack size: 0x100
// GNU-NEXT:     no copy on protected
// GNU-NEXT:     x86 feature: SHSTK
// GNU-NEXT:     x86 feature: IBT, SHSTK
// GNU-NEXT:     x86 feature: <None>
// GNU-NEXT:     x86 ISA needed: CMOV, SSE, SSE2, SSE3, SSSE3, SSE4_1, SSE4_2, AVX, AVX2, FMA, AVX512F, AVX512CD
// GNU-NEXT:     x86 ISA used: AVX512ER, AVX512PF, AVX512VL, AVX512DQ, AVX512BW, AVX512_4FMAPS, AVX512_4VNNIW, AVX512_BITALG, AVX512_IFMA, AVX512_VBMI, AVX512_VBMI2, AVX512_VNNI
// GNU-NEXT:     x86 feature needed: x86, x87, MMX, XMM, YMM
// GNU-NEXT:     x86 feature used: ZMM, FXSR, XSAVE, XSAVEOPT, XSAVEC
// GNU-NEXT:     <application-specific type 0xfefefefe>
// GNU-NEXT:     stack size: <corrupt length: 0x0>
// GNU-NEXT:     stack size: <corrupt length: 0x4>
// GNU-NEXT:     no copy on protected <corrupt length: 0x1>
// GNU-NEXT:     x86 feature: <corrupt length: 0x0>
// GNU-NEXT:     x86 feature: IBT, <unknown flags: 0xf000f000>
// GNU-NEXT:     <corrupt type (0x2) datasz: 0x1>

// LLVM:      Notes [
// LLVM-NEXT:   NoteSection {
// LLVM-NEXT:     Offset: 0x40
// LLVM-NEXT:     Size: 0xF8
// LLVM-NEXT:     Note {
// LLVM-NEXT:       Owner: GNU
// LLVM-NEXT:       Data size: 0xE8
// LLVM-NEXT:       Type: NT_GNU_PROPERTY_TYPE_0 (property note)
// LLVM-NEXT:       Property [
// LLVM-NEXT:         stack size: 0x100
// LLVM-NEXT:         stack size: 0x100
// LLVM-NEXT:         no copy on protected
// LLVM-NEXT:         x86 feature: SHSTK
// LLVM-NEXT:         x86 feature: IBT, SHSTK
// LLVM-NEXT:         x86 feature: <None>
// LLVM-NEXT:         x86 ISA needed: CMOV, SSE, SSE2, SSE3, SSSE3, SSE4_1, SSE4_2, AVX, AVX2, FMA, AVX512F, AVX512CD
// LLVM-NEXT:         x86 ISA used: AVX512ER, AVX512PF, AVX512VL, AVX512DQ, AVX512BW, AVX512_4FMAPS, AVX512_4VNNIW, AVX512_BITALG, AVX512_IFMA, AVX512_VBMI, AVX512_VBMI2, AVX512_VNNI
// LLVM-NEXT:         x86 feature needed: x86, x87, MMX, XMM, YMM
// LLVM-NEXT:         x86 feature used: ZMM, FXSR, XSAVE, XSAVEOPT, XSAVEC
// LLVM-NEXT:         <application-specific type 0xfefefefe>
// LLVM-NEXT:         stack size: <corrupt length: 0x0>
// LLVM-NEXT:         stack size: <corrupt length: 0x4>
// LLVM-NEXT:         no copy on protected <corrupt length: 0x1>
// LLVM-NEXT:         x86 feature: <corrupt length: 0x0>
// LLVM-NEXT:         x86 feature: IBT, <unknown flags: 0xf000f000>
// LLVM-NEXT:         <corrupt type (0x2) datasz: 0x1>
// LLVM-NEXT:       ]
// LLVM-NEXT:     }
// LLVM-NEXT:   }
// LLVM-NEXT: ]

.section ".note.gnu.property", "a"
.align 4
  .long 4           /* Name length is always 4 ("GNU") */
  .long end - begin /* Data length */
  .long 5           /* Type: NT_GNU_PROPERTY_TYPE_0 */
  .asciz "GNU"      /* Name */
  .p2align 3
begin:
  .long 1           /* Type: GNU_PROPERTY_STACK_SIZE */
  .long 8           /* Data size */
  .quad 0x100       /* Data (stack size) */
  .p2align 3        /* Align to 8 byte for 64 bit */

  /* Test we handle alignment properly */
  .long 1           /* Type: GNU_PROPERTY_STACK_SIZE */
  .long 8           /* Data size */
  .long 0x100       /* Data (stack size) */
  .p2align 3        /* Align to 8 byte for 64 bit */

  .long 2           /* Type: GNU_PROPERTY_NO_COPY_ON_PROTECTED */
  .long 0           /* Data size */
  .p2align 3        /* Align to 8 byte for 64 bit */

  /* CET property note */
  .long 0xc0000002  /* Type: GNU_PROPERTY_X86_FEATURE_1_AND */
  .long 4           /* Data size */
  .long 2           /* GNU_PROPERTY_X86_FEATURE_1_SHSTK */
  .p2align 3        /* Align to 8 byte for 64 bit */

  /* CET property note with padding */
  .long 0xc0000002  /* Type: GNU_PROPERTY_X86_FEATURE_1_AND */
  .long 4           /* Data size */
  .long 3           /* Full CET support */
  .p2align 3        /* Align to 8 byte for 64 bit */

  .long 0xc0000002  /* Type: GNU_PROPERTY_X86_FEATURE_1_AND */
  .long 4           /* Data size */
  .long 0           /* Empty flags, not an error */
  .p2align 3        /* Align to 8 byte for 64 bit */

  .long 0xc0008000         /* Type: GNU_PROPERTY_X86_ISA_1_NEEDED */
  .long 4                  /* Data size */
  .long 0x00000fff         /* CMOV, ... */
  .p2align 3               /* Align to 8 byte for 64 bit */

  .long 0xc0010000         /* Type: GNU_PROPERTY_X86_ISA_1_USED */
  .long 4                  /* Data size */
  .long 0x00fff000         /* AVX512_ER, ... */
  .p2align 3               /* Align to 8 byte for 64 bit */

  .long 0xc0008001         /* Type: GNU_PROPERTY_X86_FEATURE_2_NEEDED */
  .long 4                  /* Data size */
  .long 0x0000001f         /* X86, ... */
  .p2align 3               /* Align to 8 byte for 64 bit */

  .long 0xc0010001         /* Type: GNU_PROPERTY_X86_FEATURE_2_USED */
  .long 4                  /* Data size */
  .long 0x000003e0         /* ZMM, ... */
  .p2align 3               /* Align to 8 byte for 64 bit */

  /* All notes below are broken. Test we are able to report them. */

  /* Broken note type */
  .long 0xfefefefe  /* Invalid type for testing */
  .long 0           /* Data size */
  .p2align 3        /* Align to 8 byte for 64 bit */

  /* GNU_PROPERTY_STACK_SIZE with zero stack size */
  .long 1           /* Type: GNU_PROPERTY_STACK_SIZE */
  .long 0           /* Data size */
  .p2align 3        /* Align to 8 byte for 64 bit */

  /* GNU_PROPERTY_STACK_SIZE with data size 4 (should be 8) */
  .long 1           /* Type: GNU_PROPERTY_STACK_SIZE */
  .long 4           /* Data size */
  .long 0x100       /* Data (stack size) */
  .p2align 3        /* Align to 8 byte for 64 bit */

  /* GNU_PROPERTY_NO_COPY_ON_PROTECTED with pr_datasz and some data */
  .long 2           /* Type: GNU_PROPERTY_NO_COPY_ON_PROTECTED */
  .long 1           /* Data size (corrupted) */
  .byte 1           /* Data */
  .p2align 3        /* Align to 8 byte for 64 bit */

  /* CET note with size zero */
  .long 0xc0000002  /* Type: GNU_PROPERTY_X86_FEATURE_1_AND */
  .long 0           /* Data size */
  .p2align 3        /* Align to 8 byte for 64 bit */

  /* CET note with bad flags */
  .long 0xc0000002         /* Type: GNU_PROPERTY_X86_FEATURE_1_AND */
  .long 4                  /* Data size */
  .long 0xf000f001         /* GNU_PROPERTY_X86_FEATURE_1_IBT and bad bits */
  .p2align 3               /* Align to 8 byte for 64 bit */

  /* GNU_PROPERTY_NO_COPY_ON_PROTECTED with pr_datasz and without data */
  .long 2           /* Type: GNU_PROPERTY_NO_COPY_ON_PROTECTED */
  .long 1           /* Data size (corrupted) */
  .p2align 3        /* Align to 8 byte for 64 bit */
end:
