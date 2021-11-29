; RUN: opt -passes=globalopt -S < %s | FileCheck %s
; RUN: opt -passes=globalopt -S < %s | FileCheck %s

%s = type { i32 }
@g = internal global %s* null, align 8

; Test code pattern for:
;   class s { int a; s() { a = 1;} };
;   g = new s();
;

define internal void @f() {
; CHECK-LABEL: @f(
; CHECK-NEXT:    ret void
;
  %1 = tail call i8* @_Znwm(i64 4)
  %2 = bitcast i8* %1 to %s*
  %3 = getelementptr inbounds %s, %s* %2, i64 0, i32 0
  store i32 1, i32* %3, align 4
  store i8* %1, i8** bitcast (%s** @g to i8**), align 8
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
  %0 = load %s*, %s** @g, align 4
  %1 = getelementptr inbounds %s, %s* %0, i64 0, i32 0
  %2 = load i32, i32* %1, align 4
  ret i32 %2
}

declare nonnull i8* @_Znwm(i64)

declare signext i32 @printf(i8*, ...)

