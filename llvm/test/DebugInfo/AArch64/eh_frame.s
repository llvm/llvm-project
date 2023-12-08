// RUN: llvm-mc -triple aarch64-none-linux-gnu -filetype=obj %s -o %t
// RUN: llvm-objdump -s %t | FileCheck %s
        .text
        .globl foo
        .type foo,@function
foo:
        .cfi_startproc
        ret
        .cfi_endproc

// The really key points we're checking here are:
//  * Return register is x30.
//  * Pointer format is 0x1b (GNU doesn't appear to understand others).

// The rest is largely incidental, but not expected to change regularly.

// Output is:

// CHECK: Contents of section .eh_frame:
// CHECK-NEXT: 0000 10000000 00000000 017a5200 017c1e01  .........zR..|..
// CHECK-NEXT: 0010 1b0c1f00 10000000 18000000 00000000  ................


// Won't check the rest, it's rather incidental.
// 0020 04000000 00000000                    ........



// The first CIE:
// -------------------
// 10000000: length of first CIE = 0x10
// 00000000: This is a CIE
// 03: version = 0x3
// 7a 52 00: augmentation string "zR" -- pointer format is specified
// 01: code alignment factor 1
// 7c: data alignment factor -4
// 1e: return address register 30 (== x30).
// 01: 1 byte of augmentation
// 1b: pointer format 1b: DW_EH_PE_pcrel | DW_EH_PE_sdata4
// 0c 1f 00: initial instructions: "DW_CFA_def_cfa x31 ofs 0" in this case

// Next the FDE:
// -------------
// 10000000: FDE length 0x10
// 18000000: Uses CIE 0x18 backwards (only coincidentally same as above)
// 00000000: PC begin for this FDE is at 00000000 (relocation is applied here)
// 04000000: FDE applies up to PC begin+0x14
// 00: Augmentation string length 0 for this FDE
