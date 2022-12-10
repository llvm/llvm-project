; RUN: opt < %s -passes=sink -S | FileCheck %s
declare void @foo(ptr)
declare ptr @llvm.load.relative.i32(ptr %ptr, i32 %offset) argmemonly nounwind readonly
define i64 @sinkload(i1 %cmp, ptr %ptr, i32 %off) {
; CHECK-LABEL: @sinkload
top:
    %a = alloca i64
; CHECK: call void @foo(ptr %a)
; CHECK-NEXT: %x = load i64, ptr %a
; CHECK-NEXT: %y = call ptr @llvm.load.relative.i32(ptr %ptr, i32 %off)
    call void @foo(ptr %a)
    %x = load i64, ptr %a
    %y = call ptr @llvm.load.relative.i32(ptr %ptr, i32 %off)
    br i1 %cmp, label %A, label %B
A:
    store i64 0, ptr %a
    store i8 0, ptr %ptr
    br label %B
B:
; CHECK-NOT: load i64, ptr %a
; CHECK-NOT: call ptr @llvm.load.relative(ptr %ptr, i32 off)
    %y2 = ptrtoint ptr %y to i64
    %retval = add i64 %y2, %x
    ret i64 %retval
}

