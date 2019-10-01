; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s
; ModuleID = 'bug.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@func.flagmask = internal constant i64 1, align 8

define void @func() nounwind {
entry:
  %src = alloca i32, align 4
  %dst = alloca i32, align 4
  %flags = alloca i64, align 8
  %newflags = alloca i64, align 8
  store i32 0, i32* %src, align 4
  store i32 0, i32* %dst, align 4
  store i64 1, i64* %flags, align 8
  store i64 -1, i64* %newflags, align 8
  %0 = bitcast i32* %dst to i8*
  %tmp = load i64, i64* %flags, align 8
  %and = and i64 %tmp, 1
  %1 = bitcast i32* %src to i8*
  %tmp1 = load i8, i8* %1
  %2 = bitcast i32* %dst to i8*
  %tmp2 = load i8, i8* %2
  call void asm "pushfq \0Aandq $2, (%rsp) \0Aorq  $3, (%rsp) \0Apopfq \0Aaddb $4, $1 \0Apushfq \0Apopq $0 \0A", "=*&rm,=*&rm,i,r,r,1,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* %newflags, i8* %0, i64 -2, i64 %and, i8 %tmp1, i8 %tmp2) nounwind
  ret void
}
