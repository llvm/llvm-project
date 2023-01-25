; RUN: not --crash llc -mtriple aarch64-linux-gnu -mattr=+sve <%s 2>&1 | FileCheck %s

declare i32 @sve_printf(ptr, <vscale x 4 x i32>, ...)

@.str_1 = internal constant [6 x i8] c"boo!\0A\00"

; CHECK: Passing SVE types to variadic functions is currently not supported
define void @foo(<vscale x 4 x i32> %x) {
  call i32 (ptr, <vscale x 4 x i32>, ...) @sve_printf(ptr @.str_1, <vscale x 4 x i32> %x, <vscale x 4 x i32> %x)
  ret void
}
