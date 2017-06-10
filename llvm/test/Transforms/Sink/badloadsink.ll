; RUN: opt < %s -basicaa -sink -S | FileCheck %s
declare void @foo(i64 *)
define i64 @sinkload(i1 %cmp) {
; CHECK-LABEL: @sinkload
top:
    %a = alloca i64
; CHECK: call void @foo(i64* %a)
; CHECK-NEXT: %x = load i64, i64* %a
    call void @foo(i64* %a)
    %x = load i64, i64* %a
    br i1 %cmp, label %A, label %B
A:
    store i64 0, i64 *%a
    br label %B
B:
; CHECK-NOT: load i64, i64 *%a
    ret i64 %x
}
