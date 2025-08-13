// TODO: Add other architectures as they gain sframe support
// REQUIRES: x86-registered-target
// RUN: llvm-mc --assemble --filetype=obj --gsframe -triple x86_64 %s -o %t.o
// RUN: llvm-readelf --sframe %t.o | FileCheck %s

	.cfi_sections .sframe
	
f1:
	.cfi_startproc
	nop
        .cfi_endproc

// CHECK: SFrame section '.sframe' {
// CHECK-NEXT:  Header {
// CHECK-NEXT:    Magic: 0xDEE2
// CHECK-NEXT:    Version: V2 (0x2)
// CHECK-NEXT:    Flags [ (0x0)
// CHECK:    ABI: AMD64EndianLittle (0x3)
// CHECK-NEXT:    CFA fixed FP offset (unused): 0
// CHECK-NEXT:    CFA fixed RA offset: 0
// CHECK-NEXT:    Auxiliary header length: 0
// CHECK-NEXT:    Num FDEs: 0
// CHECK-NEXT:    Num FREs: 0
// CHECK-NEXT:    FRE subsection length: 0
// CHECK-NEXT:    FDE subsection offset: 0
// CHECK-NEXT:    FRE subsection offset: 0
