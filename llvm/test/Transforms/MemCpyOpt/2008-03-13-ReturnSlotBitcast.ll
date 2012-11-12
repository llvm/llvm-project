; RUN: opt < %s -basicaa -memcpyopt -S | not grep "call.*memcpy."
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"

%a = type { i32 }
%b = type { float }

declare void @g(%a*)

define float @f() {
entry:
  %a_var = alloca %a
  %b_var = alloca %b, align 1
  call void @g(%a* %a_var)
  %a_i8 = bitcast %a* %a_var to i8*
  %b_i8 = bitcast %b* %b_var to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %b_i8, i8* %a_i8, i32 4, i32 1, i1 false)
  %tmp1 = getelementptr %b* %b_var, i32 0, i32 0
  %tmp2 = load float* %tmp1
  ret float %tmp2
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
