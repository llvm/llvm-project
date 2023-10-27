; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: 'vscale_range' minimum must be greater than 0
declare ptr @a(ptr) vscale_range(0, 1)

; CHECK: 'vscale_range' minimum cannot be greater than maximum
declare ptr @b(ptr) vscale_range(8, 1)

; CHECK: 'vscale_range' minimum must be power-of-two value
declare ptr @c(ptr) vscale_range(3, 16)

; CHECK: 'vscale_range' maximum must be power-of-two value
declare ptr @d(ptr) vscale_range(2, 5)
