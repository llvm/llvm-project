; RUN: opt -S -O1 < %s | FileCheck %s

@a = common global i32 0, align 4

; @hook_me is weak, so GMR must not eliminate the reload of @a in @f,
; even though @hook_me doesn't mod or ref @a.

; Function Attrs: nounwind ssp uwtable
define weak i32 @hook_me() {
  ret i32 0
}

; Function Attrs: nounwind ssp uwtable
define i32 @f() {
  %1 = alloca i32, align 4
  store i32 4, ptr @a, align 4
  %2 = call i32 @hook_me()
  ; CHECK: load i32, ptr @a, align 4
  %3 = load i32, ptr @a, align 4
  %4 = add nsw i32 %3, %2
  store i32 %4, ptr @a, align 4
  %5 = load i32, ptr %1
  ret i32 %5
}
