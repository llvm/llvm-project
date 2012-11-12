; RUN: opt < %s -instcombine -S

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-pc-win32"

%myStruct = type { float, [3 x float], [4 x float], i32 }

; make sure that we are not crashing when creating an illegal type
define void @func(%myStruct addrspace(1)* nocapture %p) nounwind {
ST:
  %A = getelementptr inbounds %myStruct addrspace(1)* %p, i64 0
  %B = bitcast %myStruct addrspace(1)* %A to %myStruct*
  %C = getelementptr inbounds %myStruct* %B, i32 0, i32 1
  %D = getelementptr inbounds [3 x float]* %C, i32 0, i32 2
  %E = load float* %D, align 4
  %F = fsub float %E, undef
  ret void
}

