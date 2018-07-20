; RUN: opt < %s -basicaa -sink -S | FileCheck %s
declare void @foo(i64 *)
declare i8* @llvm.load.relative.i32(i8* %ptr, i32 %offset) argmemonly nounwind readonly
define i64 @sinkload(i1 %cmp, i8* %ptr, i32 %off) {
; CHECK-LABEL: @sinkload
top:
    %a = alloca i64
; CHECK: call void @foo(i64* %a)
; CHECK-NEXT: %x = load i64, i64* %a
; CHECK-NEXT: %y = call i8* @llvm.load.relative.i32(i8* %ptr, i32 %off)
    call void @foo(i64* %a)
    %x = load i64, i64* %a
    %y = call i8* @llvm.load.relative.i32(i8* %ptr, i32 %off)
    br i1 %cmp, label %A, label %B
A:
    store i64 0, i64 *%a
    store i8 0, i8 *%ptr
    br label %B
B:
; CHECK-NOT: load i64, i64 *%a
; CHECK-NOT: call i8* @llvm.load.relative(i8* %ptr, i32 off)
    %y2 = ptrtoint i8* %y to i64
    %retval = add i64 %y2, %x
    ret i64 %retval
}

