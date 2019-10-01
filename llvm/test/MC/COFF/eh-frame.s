// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s -o - | llvm-readobj -S | FileCheck %s

	.def	 _main;
	.scl	2;
	.type	32;
	.endef
	.text
	.globl	_main
_main:
	.cfi_startproc
	ret
	.cfi_endproc

// CHECK:    Name: .eh_frame
// CHECK-NEXT:    VirtualSize:
// CHECK-NEXT:    VirtualAddress:
// CHECK-NEXT:    RawDataSize:
// CHECK-NEXT:    PointerToRawData:
// CHECK-NEXT:    PointerToRelocations:
// CHECK-NEXT:    PointerToLineNumbers:
// CHECK-NEXT:    RelocationCount:
// CHECK-NEXT:    LineNumberCount:
// CHECK-NEXT:    Characteristics [
// CHECK-NEXT:      IMAGE_SCN_ALIGN_4BYTES
// CHECK-NEXT:      IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:      IMAGE_SCN_MEM_READ
// CHECK-NEXT:    ]
