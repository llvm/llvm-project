; RUN: not llvm-as -f %s -o /dev/null |& grep {alignment argument of memory intrinsics must be a constant int}
; PR2318

define void @x(i8* %a, i8* %src, i64 %len, i32 %align) nounwind  {
entry:
        tail call void @llvm.memcpy.i64( i8* %a, i8* %src, i64 %len, i32 %align) nounwind 
        ret void
}

declare void @llvm.memcpy.i64( i8* %a, i8* %src, i64 %len, i32)

