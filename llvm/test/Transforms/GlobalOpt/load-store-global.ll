; RUN: opt < %s -passes=globalopt -S | FileCheck %s

@G = internal global i32 17             ; <ptr> [#uses=3]
; CHECK-NOT: @G

define void @foo() {
        %V = load i32, ptr @G               ; <i32> [#uses=1]
        store i32 %V, ptr @G
        ret void
; CHECK-LABEL: @foo(
; CHECK-NEXT: ret void
}

define i32 @bar() {
        %X = load i32, ptr @G               ; <i32> [#uses=1]
        ret i32 %X
; CHECK-LABEL: @bar(
; CHECK-NEXT: ret i32 17
}

@a = internal global ptr null, align 8
; CHECK-NOT: @a

; PR13968
define void @qux() nounwind {
  %g = getelementptr ptr, ptr @a, i32 1
  %cmp = icmp ne ptr null, @a
  %cmp2 = icmp eq ptr null, @a
  %cmp3 = icmp eq ptr null, %g
  store ptr inttoptr (i64 1 to ptr), ptr @a, align 8
  %l = load ptr, ptr @a, align 8
  ret void
; CHECK-LABEL: @qux(
; CHECK-NOT: store
; CHECK-NOT: load
}

@addrspacecast_a = internal global ptr null

define void @addrspacecast_user() {
; CHECK-LABEL: @addrspacecast_user
; CHECK-NOT: store
; CHECK-NOT: load
  %g = addrspacecast ptr @addrspacecast_a to ptr addrspace(1)
  store ptr inttoptr (i64 1 to ptr), ptr @addrspacecast_a, align 8
  %l = load ptr, ptr @addrspacecast_a, align 8
  ret void
}
