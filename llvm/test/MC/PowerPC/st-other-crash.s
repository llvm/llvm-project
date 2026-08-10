// RUN: llvm-mc < %s -filetype=obj -triple powerpc64le-pc-linux | \
// RUN:   llvm-readobj --symbols - | FileCheck %s

// This used to crash. Make sure it produce the correct symbol.

// CHECK:       Symbol {
// CHECK:         Name: _ZN4llvm11SmallVectorIcLj0EEC2Ev (12)
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)
// CHECK-NEXT:    Type: None (0x0)
// CHECK-NEXT:    Other [ (0x40)
// CHECK-NEXT:    ]
// CHECK-NEXT:    Section: .group
// CHECK-NEXT:  }


	.section	.text._ZN4llvm11SmallVectorIcLj0EEC2Ev,"axG",@progbits,_ZN4llvm11SmallVectorIcLj0EEC2Ev,comdat
.Ltmp2:
	addis 2, 12, .TOC.-.Ltmp2@ha
.Ltmp3:
	.localentry	_ZN4llvm11SmallVectorIcLj0EEC2Ev, .Ltmp3-.Ltmp2
