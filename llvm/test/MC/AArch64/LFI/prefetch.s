// RUN: llvm-mc -triple aarch64_lfi --aarch64-lfi-no-guard-elim %s | FileCheck %s

prfm pldl1keep, [x0]
// CHECK: prfm pldl1keep, [x27, w0, uxtw]

prfm pldl1strm, [x0]
// CHECK: prfm pldl1strm, [x27, w0, uxtw]

prfm pldl2keep, [x0]
// CHECK: prfm pldl2keep, [x27, w0, uxtw]

prfm pldl2strm, [x0]
// CHECK: prfm pldl2strm, [x27, w0, uxtw]

prfm pldl3keep, [x0]
// CHECK: prfm pldl3keep, [x27, w0, uxtw]

prfm pldl3strm, [x0]
// CHECK: prfm pldl3strm, [x27, w0, uxtw]

prfm pstl1keep, [x0]
// CHECK: prfm pstl1keep, [x27, w0, uxtw]

prfm pstl1strm, [x0]
// CHECK: prfm pstl1strm, [x27, w0, uxtw]

prfm pstl2keep, [x0]
// CHECK: prfm pstl2keep, [x27, w0, uxtw]

prfm pstl2strm, [x0]
// CHECK: prfm pstl2strm, [x27, w0, uxtw]

prfm pldl1keep, [x0, #8]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: prfm pldl1keep, [x28, #8]

prfm pstl1strm, [x0, #16]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: prfm pstl1strm, [x28, #16]

prfm pldl1keep, [x0, x1]
// CHECK:      add x26, x0, x1
// CHECK-NEXT: prfm pldl1keep, [x27, w26, uxtw]

prfm pldl1keep, [x0, x1, lsl #3]
// CHECK:      add x26, x0, x1, lsl #3
// CHECK-NEXT: prfm pldl1keep, [x27, w26, uxtw]

prfm pldl1keep, [x0, w1, uxtw]
// CHECK:      add x26, x0, w1, uxtw
// CHECK-NEXT: prfm pldl1keep, [x27, w26, uxtw]

prfm pldl1keep, [x0, w1, sxtw]
// CHECK:      add x26, x0, w1, sxtw
// CHECK-NEXT: prfm pldl1keep, [x27, w26, uxtw]

prfm pldl1keep, [x0, w1, uxtw #3]
// CHECK:      add x26, x0, w1, uxtw #3
// CHECK-NEXT: prfm pldl1keep, [x27, w26, uxtw]

prfm pldl1keep, [x0, w1, sxtw #3]
// CHECK:      add x26, x0, w1, sxtw #3
// CHECK-NEXT: prfm pldl1keep, [x27, w26, uxtw]

prfum pldl1keep, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: prfum pldl1keep, [x28]

prfum pldl1keep, [x0, #1]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: prfum pldl1keep, [x28, #1]

prfum pstl1strm, [x0, #-8]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: prfum pstl1strm, [x28, #-8]

prfm pldl1keep, [sp]
// CHECK: prfm pldl1keep, [sp]

prfm pldl1keep, [sp, #8]
// CHECK: prfm pldl1keep, [sp, #8]
