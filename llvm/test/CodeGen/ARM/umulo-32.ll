; RUN: llc < %s -mtriple=thumbv6-apple-darwin | FileCheck %s

%umul.ty = type { i32, i1 }

define i32 @func(i32 %a) nounwind {
; CHECK: func
; CHECK: muldi3
  %tmp0 = tail call %umul.ty @llvm.umul.with.overflow.i32(i32 %a, i32 37)
  %tmp1 = extractvalue %umul.ty %tmp0, 0
  %tmp2 = select i1 undef, i32 -1, i32 %tmp1
  ret i32 %tmp2
}

declare %umul.ty @llvm.umul.with.overflow.i32(i32, i32) nounwind readnone

define i32 @f(i32 %argc, i8** %argv) ssp {
; CHECK: func
; CHECK: str     r0
; CHECK: movs    r2
; CHECK: mov     r1
; CHECK: mov     r3
; CHECK: muldi3
%1 = alloca i32, align 4
%2 = alloca i32, align 4
%3 = alloca i8**, align 4
%m_degree = alloca i32, align 4
store i32 0, i32* %1
store i32 %argc, i32* %2, align 4
store i8** %argv, i8*** %3, align 4
store i32 10, i32* %m_degree, align 4
%4 = load i32* %m_degree, align 4
%5 = call %umul.ty @llvm.umul.with.overflow.i32(i32 %4, i32 8)
%6 = extractvalue %umul.ty %5, 1
%7 = extractvalue %umul.ty %5, 0
%8 = select i1 %6, i32 -1, i32 %7
%9 = call noalias i8* @_Znam(i32 %8)
%10 = bitcast i8* %9 to double*
ret i32 0
}

declare noalias i8* @_Znam(i32)
