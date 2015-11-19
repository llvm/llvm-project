; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: alignment argument of memory intrinsics must be a constant int
; PR2318

define void @x(i8* %a, i8* %src, i64 %len, i32 %align) nounwind  {
entry:
        tail call void @llvm.memcpy.p0i8.p0i8.i64( i8* %a, i8* %src, i64 %len, i32 %align, i1 false) nounwind 
        ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64( i8* %a, i8* %src, i64 %len, i32, i1)

