// RUN: llvm-mc < %s -triple=x86_64-apple-macosx -filetype=obj -o - | llvm-objdump --fault-map-section - | FileCheck %s

	.section	__LLVM_FAULTMAPS,__llvm_faultmaps
__LLVM_FaultMaps:
	.byte	1
	.byte	0
	.short	0
	.long	2
	.quad	0xFFDEAD
	.long	1
	.long	0
	.long	1
	.long	100
	.long	200

	.quad	0xFFDAED
	.long	1
	.long	0
	.long	1
	.long	400
	.long	500

// CHECK: FaultMap table:
// CHECK-NEXT: Version: 0x1
// CHECK-NEXT: NumFunctions: 2
// CHECK-NEXT: FunctionAddress: 0xffdead, NumFaultingPCs: 1
// CHECK-NEXT: Fault kind: FaultingLoad, faulting PC offset: 100, handling PC offset: 200
// CHECK-NEXT: FunctionAddress: 0xffdaed, NumFaultingPCs: 1
// CHECK-NEXT: Fault kind: FaultingLoad, faulting PC offset: 400, handling PC offset: 500
