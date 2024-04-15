; RUN: llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs | FileCheck %s  
; RUN: llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs | FileCheck %s   -check-prefix=CHECK-FOUND

define void @kernel_func(ptr %in.vec, ptr %out.vec0) nounwind {
  entry:
  %wide.vec = load <32 x i8>, ptr %in.vec, align 64
  %vec0 = shufflevector <32 x i8> %wide.vec, <32 x i8> undef, <4 x i32> <i32 0, i32 8, i32 16, i32 24>
  store <4 x i8> %vec0, ptr %out.vec0, align 64
  ret void

; CHECK-FOUND: prmt.b32 	{{.*}} 16384;
; CHECK-FOUND: prmt.b32 	{{.*}} 64;
; CHECK-FOUND: prmt.b32 	{{.*}} 30224;

; CHECK:  @kernel_func
; CHECK-NOT: 	prmt.b32 	{{.*}} -1;
; CHECK:  -- End function
}
