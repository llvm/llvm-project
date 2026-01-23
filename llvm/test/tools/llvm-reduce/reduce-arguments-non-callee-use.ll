; RUN: llvm-reduce %s -o %t --abort-on-invalid-reduction --delta-passes=arguments --test FileCheck --test-arg %s --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefix=REDUCED

; INTERESTING: @initializer_user
; REDUCED: @initializer_user = global [1 x ptr] [ptr @captured_func]
@initializer_user = global [1 x ptr] [ptr @captured_func ]

; INTERESTING-LABEL: define i32 @captured_func(

; REDUCED-LABEL: define i32 @captured_func() {
define i32 @captured_func(i32 %a, i32 %b) {
  %mul = mul i32 %a, %b
  ret i32 %mul
}

; INTERESTING-LABEL: declare void @captures(
declare void @captures(i32, ptr, i32)


; INTERESTING-LABEL: define i32 @caller(
; INTERESTING: = call
; INTERESTING: = call

; REDUCED-LABEL: define i32 @caller(i32 %a, i32 %b) {
; REDUCED: %call0 = call i32 @captures(i32 %a, ptr @captured_func, i32 %b)
; REDUCED: %call1 = call i32 @captured_func()
define i32 @caller(i32 %a, i32 %b) {
  %call0 = call i32 @captures(i32 %a, ptr @captured_func, i32 %b)
  %call1 = call i32 @captured_func(i32 %a, i32 %b)
  %add = add i32 %call0, %call1
  ret i32 %add
}
