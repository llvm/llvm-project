; RUN: opt -mtriple=amdgcn-unknown-amdhsa -passes=loop-unroll -S %s | FileCheck %s

%struct.wombat = type { %struct.zot, i32, [16 x i32], [16 x i32], i32, i32, [16 x i32], i32 }
%struct.zot = type { i32, i32, [1024 x i32] }

@global = external addrspace(3) global %struct.wombat

; Ensure that a cascaded GEP for local address space does not inhibit unrolling
;
; CHECK-LABEL: @unroll_when_cascaded_gep
; CHECK: bb:
; CHECK:   br {{.*}}, label %bb2.unr-lcssa, label %bb.new
; CHECK: bb.new:
; CHECK:   %unroll_iter = 
; CHECK:   br label %bb1
; CHECK: bb1:
; CHECK:   br {{.*}}, label %bb2.unr-lcssa.loopexit, label %bb1
; CHECK: bb2.unr-lcssa.loopexit:
; CHECK:   br label %bb2.unr-lcssa
; CHECK: bb2.unr-lcssa:
; CHECK:   br {{.*}}, label %bb1.epil.preheader, label %bb2
; CHECK: bb1.epil.preheader:
; CHECK:   br label %bb1.epil
; CHECK: bb1.epil:
; CHECK:   br {{.*}}, label %bb1.epil, label %bb2.epilog-lcssa
; CHECK: bb2.epilog-lcssa:
; CHECK:   br label %bb2
; CHECK: bb2:
; CHECK:   ret void
define amdgpu_kernel void @unroll_when_cascaded_gep(i32 %arg) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %phi = phi i32 [ 0, %bb ], [ %add, %bb1 ]
  %getelementptr = getelementptr [1024 x i32], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @global, i32 8), i32 0, i32 0
  %add = add i32 %phi, 1
  %icmp = icmp eq i32 %phi, %arg
  br i1 %icmp, label %bb2, label %bb1

bb2:                                              ; preds = %bb1
  ret void
}

