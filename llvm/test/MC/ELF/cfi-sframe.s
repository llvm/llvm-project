// TODO: Add other architectures as they gain sframe support
// REQUIRES: x86-registered-target
// RUN: llvm-mc --assemble --filetype=obj --gsframe -triple x86_64 %s -o %t.o
// RUN: objdump --sframe %t.o | FileCheck %s

.cfi_sections .sframe
	
f1:
        .cfi_startproc
        nop
        .cfi_endproc


// CHECK:	Contents of the SFrame section .sframe:
// CHECK-NEXT:	  Header :
// CHECK:	    Version: SFRAME_VERSION_2
// CHECK-NEXT:	    Flags: NONE
// CHECK-NEXT:	    Num FDEs: 0
// CHECK-NEXT:	    Num FREs: 0
// CHECK:        Function Index :
