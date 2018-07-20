; RUN: opt < %s -basicaa -memcpyopt -S | not grep "call.*memcpy"

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin9"

%0 = type { x86_fp80, x86_fp80 }

define void @ccosl(%0* noalias sret %agg.result, %0* byval align 8 %z) nounwind {
entry:
  %iz = alloca %0
  %memtmp = alloca %0, align 16
  %tmp1 = getelementptr %0, %0* %z, i32 0, i32 1
  %tmp2 = load x86_fp80, x86_fp80* %tmp1, align 16
  %tmp3 = fsub x86_fp80 0xK80000000000000000000, %tmp2
  %tmp4 = getelementptr %0, %0* %iz, i32 0, i32 1
  %real = getelementptr %0, %0* %iz, i32 0, i32 0
  %tmp7 = getelementptr %0, %0* %z, i32 0, i32 0
  %tmp8 = load x86_fp80, x86_fp80* %tmp7, align 16
  store x86_fp80 %tmp3, x86_fp80* %real, align 16
  store x86_fp80 %tmp8, x86_fp80* %tmp4, align 16
  call void @ccoshl(%0* noalias sret %memtmp, %0* byval align 8 %iz) nounwind
  %memtmp14 = bitcast %0* %memtmp to i8*
  %agg.result15 = bitcast %0* %agg.result to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 16 %agg.result15, i8* align 16 %memtmp14, i32 32, i1 false)
  ret void
}

declare void @ccoshl(%0* noalias nocapture sret, %0* byval) nounwind

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind
