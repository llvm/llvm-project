; Check the recursive inliner doesn't inline a function with a stack size exceeding a given limit.
; RUN: opt < %s -passes=inline -S | FileCheck --check-prefixes=ALL,UNLIMITED %s
; RUN: opt < %s -passes=inline -S -recursive-inline-max-stacksize=256 | FileCheck --check-prefixes=ALL,LIMITED %s

declare void @init(ptr)

define internal i32 @foo() {
  %1 = alloca [65 x i32], align 16
  call void @init(ptr %1)
  %2 = load i32, ptr %1, align 4
  ret i32 %2
}

define i32 @bar() {
  %1 = call i32 @foo()
  ret i32 %1
; ALL: define {{.*}}@bar
; ALL-NOT: define
; UNLIMITED-NOT: call {{.*}}@foo
; LIMITED-NOT: call {{.*}}@foo
}

; Check that, under the tighter limit, baz() doesn't inline foo()
define i32 @baz() {
  %1 = call i32 @foo()
  %2 = call i32 @baz()
  %3 = add i32 %1, %2
  ret i32 %3
; ALL: define {{.*}}@baz
; ALL-NOT: define
; UNLIMITED-NOT: call {{.*}}@foo
; LIMITED: call {{.*}}@foo
}
