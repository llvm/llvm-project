// TODO: Add other architectures as they gain sframe support
// REQUIRES: x86-registered-target
// RUN: llvm-mc --assemble --filetype=obj -triple x86_64 %s -o %t.o  2>&1 | FileCheck %s
// RUN: llvm-readelf --sframe %t.o | FileCheck --check-prefix=CHECK-NOFDES %s


        .cfi_sections .sframe
f1:
        .cfi_startproc simple
// CHECK: non-default RA register {{.*}}
        .cfi_return_column 0
        nop
// CHECK: {{.*}} adjusting CFA offset without a base register.{{.*}}
        .cfi_def_cfa_offset 16 // no line number reported here.
        nop
// CHECK: [[@LINE+1]]:{{.*}} adjusting CFA offset without a base register.{{.*}}
        .cfi_adjust_cfa_offset 16
        nop
        .cfi_endproc

f2:
        .cfi_startproc
        nop
// CHECK: canonical Frame Address not in stack- or frame-pointer. {{.*}}
        .cfi_def_cfa 0, 4
        nop

        .cfi_endproc

// CHECK-NOFDES:    Num FDEs: 0
// CHECK-NOFDES:    Num FREs: 0
