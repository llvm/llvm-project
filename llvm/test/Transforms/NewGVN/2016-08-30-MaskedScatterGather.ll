; XFAIL: *
; RUN: opt < %s -basicaa -newgvn -S | FileCheck %s

declare void @llvm.masked.scatter.v2i32.v2p0i32(<2 x i32> , <2 x i32*> , i32 , <2 x i1> )
declare <2 x i32> @llvm.masked.gather.v2i32.v2p0i32(<2 x i32*>, i32, <2 x i1>, <2 x i32>)

; This test ensures that masked scatter and gather operations, which take vectors of pointers,
; do not have pointer aliasing ignored when being processed.
; No scatter/gather calls should end up eliminated
; CHECK: llvm.masked.gather
; CHECK: llvm.masked.gather
; CHECK: llvm.masked.scatter
; CHECK: llvm.masked.gather
; CHECK: llvm.masked.scatter
; CHECK: llvm.masked.gather
define spir_kernel void @test(<2 x i32*> %in1, <2 x i32*> %in2, i32* %out) {
entry:
  ; Just some temporary storage
  %tmp.0 = alloca i32
  %tmp.1 = alloca i32
  %tmp.i = insertelement <2 x i32*> undef, i32* %tmp.0, i32 0
  %tmp = insertelement <2 x i32*> %tmp.i, i32* %tmp.1, i32 1
  ; Read from in1 and in2
  %in1.v = call <2 x i32> @llvm.masked.gather.v2i32.v2p0i32(<2 x i32*> %in1, i32 1, <2 x i1> <i1 true, i1 true>, <2 x i32> undef) #1
  %in2.v = call <2 x i32> @llvm.masked.gather.v2i32.v2p0i32(<2 x i32*> %in2, i32 1, <2 x i1> <i1 true, i1 true>, <2 x i32> undef) #1
  ; Store in1 to the allocas
  call void @llvm.masked.scatter.v2i32.v2p0i32(<2 x i32> %in1.v, <2 x i32*> %tmp, i32 1, <2 x i1> <i1 true, i1 true>);
  ; Read in1 from the allocas
  ; This gather should alias the scatter we just saw
  %tmp.v.0 = call <2 x i32> @llvm.masked.gather.v2i32.v2p0i32(<2 x i32*> %tmp, i32 1, <2 x i1> <i1 true, i1 true>, <2 x i32> undef) #1
  ; Store in2 to the allocas
  call void @llvm.masked.scatter.v2i32.v2p0i32(<2 x i32> %in2.v, <2 x i32*> %tmp, i32 1, <2 x i1> <i1 true, i1 true>);
  ; Read in2 from the allocas
  ; This gather should alias the scatter we just saw, and not be eliminated
  %tmp.v.1 = call <2 x i32> @llvm.masked.gather.v2i32.v2p0i32(<2 x i32*> %tmp, i32 1, <2 x i1> <i1 true, i1 true>, <2 x i32> undef) #1
  ; Store in2 to out for good measure
  %tmp.v.1.0 = extractelement <2 x i32> %tmp.v.1, i32 0
  %tmp.v.1.1 = extractelement <2 x i32> %tmp.v.1, i32 1
  store i32 %tmp.v.1.0, i32* %out
  %out.1 = getelementptr i32, i32* %out, i32 1
  store i32 %tmp.v.1.1, i32* %out.1
  ret void
}
