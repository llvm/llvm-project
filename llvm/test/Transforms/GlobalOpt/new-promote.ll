; RUN: opt -passes=globalopt -S < %s | FileCheck %s

%s = type { i32 }
@g = internal global ptr null, align 8

; Test code pattern for:
;   class s { int a; s() { a = 1;} };
;   g = new s();
;

define internal void @f() {
; CHECK-LABEL: @f(
; CHECK-NEXT:    ret void
;
  %1 = tail call ptr @_Znwm(i64 4)
  store i32 1, ptr %1, align 4
  store ptr %1, ptr @g, align 8
  ret void
}

define dso_local signext i32 @main() {
; CHECK-LABEL: @main(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call fastcc void @f()
; CHECK-NEXT:    ret i32 1
;
entry:
  call void @f()
  %0 = load ptr, ptr @g, align 4
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

declare nonnull ptr @_Znwm(i64)

declare signext i32 @printf(ptr, ...)

