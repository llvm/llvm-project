// RUN: llvm-mc -triple i686-unknown-linux-gnu -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-ASM-ROUNDTRIP
// RUN: llvm-mc -triple i686-unknown-linux-gnu -filetype obj -o - %s | llvm-objdump -s -j .eh_frame - | FileCheck %s -check-prefix CHECK-EH_FRAME
// REQUIRES: x86-registered-target

	.text

proc:
	.cfi_startproc
	.cfi_return_column 0
	.cfi_endproc

// CHECK-ASM-ROUNDTRIP: .cfi_startproc
// CHECK-ASM-ROUNDTRIP-NEXT: .cfi_return_column 0
// CHECK-ASM-ROUNDTRIP: .cfi_endproc

// CHECK-EH_FRAME: Contents of section .eh_frame:
// CHECK-EH_FRAME:  0000 14000000 00000000 017a5200 017c0001  .........zR..|..

