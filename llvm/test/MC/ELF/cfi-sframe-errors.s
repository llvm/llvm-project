// TODO: Add other architectures as they gain sframe support
// REQUIRES: x86-registered-target
// RUN: llvm-mc --assemble --filetype=obj --gsframe -triple x86_64 %s -o %t.o  2>&1 | FileCheck %s
// RUN: llvm-readelf --sframe %t.o | FileCheck --check-prefix=CHECK-NOFDES %s


	.cfi_sections .sframe
f1:
	.cfi_startproc simple
	nop
	.cfi_def_cfa_offset 16     // No base register yet
	nop
	.cfi_adjust_cfa_offset 16  // No base register yet
	nop
	.cfi_return_column 0
	nop
	.cfi_endproc

f2:	
	.cfi_startproc
	nop
	.cfi_def_cfa 0, 4
	nop
	
        .cfi_endproc

// CHECK: Non-default RA register {{.*}}
// asm parser doesn't give a location with .cfi_def_cfa_offset
// CHECK: :0:{{.*}} Adjusting CFA offset without a base register.{{.*}}
// .cfi_adjust_cfa_offset
// CHECK: :13:{{.*}} Adjusting CFA offset without a base register. {{.*}}
// CHECK: Canonical Frame Address not in stack- or frame-pointer. {{.*}}

// CHECK-NOFDES:    Num FDEs: 0
// CHECK-NOFDES:    Num FREs: 0
