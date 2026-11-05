; RUN: opt -mtriple=amdgpu9.00 -S -structurizecfg -structurizecfg-skip-uniform-regions -verify-dom-info < %s | FileCheck %s
; RUN: opt -mtriple=amdgpu9.00 -S -passes='structurizecfg<skip-uniform-regions>,verify' < %s | FileCheck %s
; RUN: opt -mtriple=amdgpu9.00 -S -passes='structurizecfg,verify' < %s | FileCheck %s

; The header -> A region contains the header -> B -> header backedge,
; but not the A -> header backedge. Treating it as an inner loop would
; reconverge at A with values of u from different iterations, invalidating
; the uniformity of c. Structurize both backedges together instead.
define amdgpu_kernel void @shared_loop_header(ptr addrspace(1) %p, ptr addrspace(1) %out) {
; CHECK-LABEL: @shared_loop_header(
; CHECK:       header:
; CHECK:         %u = load volatile i32, ptr addrspace(1) %p
; CHECK:         %d = icmp uge i32 %u, %tid
; CHECK-NEXT:    br i1 %d, label %A, label %[[FLOW:[a-zA-Z0-9]+]]
; CHECK:       A:
; CHECK-NEXT:    %c = icmp ugt i32 %u, 100
; CHECK-NEXT:    br label %[[FLOW]]
; CHECK:       [[FLOW]]:
; CHECK-NEXT:    %[[EXIT_A:[0-9]+]] = phi i1 [ %c, %A ], [ true, %header ]
; CHECK-NEXT:    %[[GO_B:[0-9]+]] = phi i1 [ false, %A ], [ true, %header ]
; CHECK-NEXT:    br i1 %[[GO_B]], label %B, label %[[LATCH:[a-zA-Z0-9]+]]
; CHECK:       B:
; CHECK-NEXT:    br label %[[LATCH]]
; CHECK:       [[LATCH]]:
; CHECK-NEXT:    %[[EXIT:[0-9]+]] = phi i1 [ false, %B ], [ %[[EXIT_A]], %[[FLOW]] ]
; CHECK-NEXT:    br i1 %[[EXIT]], label %exit, label %header
; CHECK:       exit:
; CHECK:         store i32 %u, ptr addrspace(1) %gep
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:
  %u = load volatile i32, ptr addrspace(1) %p
  %d = icmp ult i32 %u, %tid
  br i1 %d, label %B, label %A

A:
  %c = icmp ugt i32 %u, 100
  br i1 %c, label %exit, label %header

B:
  br label %header

exit:
  %gep = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store i32 %u, ptr addrspace(1) %gep
  ret void
}

; Keep the divergent if inside B as a subregion when merging the backedges.
define amdgpu_kernel void @shared_loop_header_with_subregion(ptr addrspace(1) %p, ptr addrspace(1) %out) {
; CHECK-LABEL: @shared_loop_header_with_subregion(
; CHECK:       header:
; CHECK:         %d = icmp uge i32 %u, %tid
; CHECK-NEXT:    br i1 %d, label %A, label %[[FLOW2:[a-zA-Z0-9]+]]
; CHECK:       A:
; CHECK-NEXT:    %c = icmp ugt i32 %u, 100
; CHECK-NEXT:    br label %[[FLOW2]]
; CHECK:       [[FLOW2]]:
; CHECK-NEXT:    %[[EXIT_A2:[0-9]+]] = phi i1 [ %c, %A ], [ true, %header ]
; CHECK-NEXT:    %[[GO_B2:[0-9]+]] = phi i1 [ false, %A ], [ true, %header ]
; CHECK-NEXT:    br i1 %[[GO_B2]], label %B, label %[[LATCH2:[a-zA-Z0-9]+]]
; CHECK:       B:
; CHECK-NEXT:    %e = icmp eq i32 %tid, 0
; CHECK-NEXT:    br i1 %e, label %B.then, label %B.end
; CHECK:       B.then:
; CHECK-NEXT:    store volatile i32 %u, ptr addrspace(1) %out
; CHECK-NEXT:    br label %B.end
; CHECK:       [[LATCH2]]:
; CHECK-NEXT:    %[[EXIT2:[0-9]+]] = phi i1 [ false, %B.end ], [ %[[EXIT_A2]], %[[FLOW2]] ]
; CHECK-NEXT:    br i1 %[[EXIT2]], label %exit, label %header
; CHECK:       B.end:
; CHECK-NEXT:    br label %[[LATCH2]]
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:
  %u = load volatile i32, ptr addrspace(1) %p
  %d = icmp ult i32 %u, %tid
  br i1 %d, label %B, label %A

A:
  %c = icmp ugt i32 %u, 100
  br i1 %c, label %exit, label %header

B:
  %e = icmp eq i32 %tid, 0
  br i1 %e, label %B.then, label %B.end

B.then:
  store volatile i32 %u, ptr addrspace(1) %out
  br label %B.end

B.end:
  br label %header

exit:
  %gep = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store i32 %u, ptr addrspace(1) %gep
  ret void
}
