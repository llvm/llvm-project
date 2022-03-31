; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: 'vscale_range' minimum must be greater than 0
declare i8* @a(i32*) vscale_range(0, 1)

; CHECK: 'vscale_range' minimum cannot be greater than maximum
declare i8* @b(i32*) vscale_range(8, 1)
