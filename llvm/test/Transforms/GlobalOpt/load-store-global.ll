; RUN: opt < %s -globalopt -S | FileCheck %s

@G = internal global i32 17             ; <i32*> [#uses=3]
; CHECK-NOT: @G

define void @foo() {
        %V = load i32, i32* @G               ; <i32> [#uses=1]
        store i32 %V, i32* @G
        ret void
; CHECK-LABEL: @foo(
; CHECK-NEXT: ret void
}

define i32 @bar() {
        %X = load i32, i32* @G               ; <i32> [#uses=1]
        ret i32 %X
; CHECK-LABEL: @bar(
; CHECK-NEXT: ret i32 17
}

@a = internal global i64* null, align 8
; CHECK-NOT: @a

; PR13968
define void @qux() nounwind {
  %b = bitcast i64** @a to i8*
  %g = getelementptr i64*, i64** @a, i32 1
  %cmp = icmp ne i8* null, %b
  %cmp2 = icmp eq i8* null, %b
  %cmp3 = icmp eq i64** null, %g
  store i64* inttoptr (i64 1 to i64*), i64** @a, align 8
  %l = load i64*, i64** @a, align 8
  ret void
; CHECK-LABEL: @qux(
; CHECK-NOT: store
; CHECK-NOT: load
}

