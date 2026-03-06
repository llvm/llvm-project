// RUN: llvm-mc -triple aarch64_lfi --aarch64-lfi-no-guard-elim %s | FileCheck %s

.arch_extension rcpc

ldapr x0, [x8]
// CHECK:      add x28, x27, w8, uxtw
// CHECK-NEXT: ldapr x0, [x28]

ldapr w0, [x8]
// CHECK:      add x28, x27, w8, uxtw
// CHECK-NEXT: ldapr w0, [x28]

ldaprh w0, [x8]
// CHECK:      add x28, x27, w8, uxtw
// CHECK-NEXT: ldaprh w0, [x28]

ldaprb w0, [x8]
// CHECK:      add x28, x27, w8, uxtw
// CHECK-NEXT: ldaprb w0, [x28]
