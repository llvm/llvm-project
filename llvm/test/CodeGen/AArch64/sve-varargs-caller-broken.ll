; RUN: split-file %s %t

; RUN: not --crash llc -mtriple aarch64-linux-gnu -mattr=+sve < %t/test-non-tailcall.ll 2>&1 | FileCheck %s --check-prefix=CHECKNONTAIL
; RUN: not --crash llc -mtriple aarch64-linux-gnu -mattr=+sve < %t/test-tailcall.ll 2>&1 | FileCheck %s --check-prefix=CHECKTAIL

;--- test-non-tailcall.ll
declare i32 @sve_printf(ptr, <vscale x 4 x i32>, ...)
@.str_1 = internal constant [6 x i8] c"boo!\0A\00"

; CHECKTAIL: Passing SVE types to variadic functions is currently not supported
define void @foo_nontail(<vscale x 4 x i32> %x) {
  call i32 (ptr, <vscale x 4 x i32>, ...) @sve_printf(ptr @.str_1, <vscale x 4 x i32> %x, <vscale x 4 x i32> %x)
  ret void
}

;--- test-tailcall.ll
declare i32 @sve_printf(ptr, <vscale x 4 x i32>, ...)
@.str_1 = internal constant [6 x i8] c"boo!\0A\00"

; CHECKNONTAIL: Passing SVE types to variadic functions is currently not supported
define void @foo_tail(<vscale x 4 x i32> %x) {
  tail call i32 (ptr, <vscale x 4 x i32>, ...) @sve_printf(ptr @.str_1, <vscale x 4 x i32> %x, <vscale x 4 x i32> %x)
  ret void
}
