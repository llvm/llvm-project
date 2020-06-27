; RUN: opt < %s -basic-aa -memcpyopt -dse -S | grep "call.*initialize" | not grep memtmp
; PR2077

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"

%0 = type { x86_fp80, x86_fp80 }

define internal fastcc void @initialize(%0* noalias nocapture sret %agg.result) nounwind {
entry:
  %agg.result.03 = getelementptr %0, %0* %agg.result, i32 0, i32 0
  store x86_fp80 0xK00000000000000000000, x86_fp80* %agg.result.03
  %agg.result.15 = getelementptr %0, %0* %agg.result, i32 0, i32 1
  store x86_fp80 0xK00000000000000000000, x86_fp80* %agg.result.15
  ret void
}

declare fastcc x86_fp80 @passed_uninitialized(%0* nocapture) nounwind

define fastcc void @badly_optimized() nounwind {
entry:
  %z = alloca %0
  %tmp = alloca %0
  %memtmp = alloca %0, align 8
  call fastcc void @initialize(%0* noalias sret %memtmp)
  %tmp1 = bitcast %0* %tmp to i8*
  %memtmp2 = bitcast %0* %memtmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %tmp1, i8* align 8 %memtmp2, i32 24, i1 false)
  %z3 = bitcast %0* %z to i8*
  %tmp4 = bitcast %0* %tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %z3, i8* align 8 %tmp4, i32 24, i1 false)
  %tmp5 = call fastcc x86_fp80 @passed_uninitialized(%0* %z)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind
