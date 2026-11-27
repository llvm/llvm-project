; RUN: opt -S < %s -passes='require<libcall-lowering-info>,atomic-expand' -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90 -mattr=+ptx87 -verify-each | FileCheck %s

; A generic pointer can designate thread-private local memory. The local path
; must preserve the comparison result and alignment without an atomic access.
define { i32, i1 } @generic_i32(ptr %p, i32 %cmp, i32 %new) {
; CHECK-LABEL: define { i32, i1 } @generic_i32(
; CHECK: [[ISLOCAL:%.*]] = call i1 @llvm.nvvm.isspacep.local(ptr %p)
; CHECK-NEXT: br i1 [[ISLOCAL]], label %cmpxchg.local, label %cmpxchg.atomic
; CHECK: cmpxchg.local:
; CHECK-NEXT: [[LOCAL:%.*]] = addrspacecast ptr %p to ptr addrspace(5)
; CHECK-NEXT: [[OLD:%.*]] = load i32, ptr addrspace(5) [[LOCAL]], align 4
; CHECK-NEXT: [[EQUAL:%.*]] = icmp eq i32 [[OLD]], %cmp
; CHECK-NEXT: [[NEXT:%.*]] = select i1 [[EQUAL]], i32 %new, i32 [[OLD]]
; CHECK-NEXT: store i32 [[NEXT]], ptr addrspace(5) [[LOCAL]], align 4
; CHECK-NEXT: [[PAIR0:%.*]] = insertvalue { i32, i1 } poison, i32 [[OLD]], 0
; CHECK-NEXT: [[PAIR1:%.*]] = insertvalue { i32, i1 } [[PAIR0]], i1 [[EQUAL]], 1
; CHECK-NEXT: br label %cmpxchg.end
; CHECK: cmpxchg.atomic:
; CHECK-NEXT: [[ATOMIC:%.*]] = cmpxchg ptr %p, i32 %cmp, i32 %new monotonic monotonic, align 4, !noalias.addrspace ![[NOTLOCAL:[0-9]+]]
; CHECK-NEXT: br label %cmpxchg.end
; CHECK: cmpxchg.end:
; CHECK-NEXT: [[RESULT:%.*]] = phi { i32, i1 } [ [[PAIR1]], %cmpxchg.local ], [ [[ATOMIC]], %cmpxchg.atomic ]
; CHECK-NEXT: ret { i32, i1 } [[RESULT]]
  %pair = cmpxchg ptr %p, i32 %cmp, i32 %new monotonic monotonic, align 4
  ret { i32, i1 } %pair
}

; Strong ordering, syncscope, weak, and volatile remain on the nonlocal path.
; The local path preserves volatility and an explicitly stronger alignment.
define { i64, i1 } @generic_volatile_i64(ptr %p, i64 %cmp, i64 %new) {
; CHECK-LABEL: define { i64, i1 } @generic_volatile_i64(
; CHECK: call i1 @llvm.nvvm.isspacep.local(ptr %p)
; CHECK: cmpxchg.local:
; CHECK-NOT: fence
; CHECK: [[OLD64:%.*]] = load volatile i64, ptr addrspace(5) {{%.*}}, align 16
; CHECK: store volatile i64 {{%.*}}, ptr addrspace(5) {{%.*}}, align 16
; CHECK-NOT: fence
; CHECK: cmpxchg.atomic:
; CHECK-NEXT: fence syncscope("device") seq_cst
; CHECK-NEXT: {{%.*}} = cmpxchg weak volatile ptr %p, i64 %cmp, i64 %new syncscope("device") acquire acquire, align 16, !noalias.addrspace ![[NOTLOCAL]]
  %pair = cmpxchg weak volatile ptr %p, i64 %cmp, i64 %new syncscope("device") seq_cst seq_cst, align 16
  ret { i64, i1 } %pair
}

; Partword emulation must only be introduced on the nonlocal path.
define { i8, i1 } @generic_i8(ptr %p, i8 %cmp, i8 %new) {
; CHECK-LABEL: define { i8, i1 } @generic_i8(
; CHECK: call i1 @llvm.nvvm.isspacep.local(ptr %p)
; CHECK: cmpxchg.local:
; CHECK: load i8, ptr addrspace(5) {{%.*}}, align 1
; CHECK: store i8 {{%.*}}, ptr addrspace(5) {{%.*}}, align 1
; CHECK: cmpxchg.atomic:
; CHECK: cmpxchg ptr {{%.*}}, i32 {{%.*}}, i32 {{%.*}} monotonic monotonic
  %pair = cmpxchg ptr %p, i8 %cmp, i8 %new monotonic monotonic, align 1
  ret { i8, i1 } %pair
}

; Do not create a result phi when the original result has no users.
define void @unused_result(ptr %p) {
; CHECK-LABEL: define void @unused_result(
; CHECK: call i1 @llvm.nvvm.isspacep.local(ptr %p)
; CHECK-NOT: insertvalue
; CHECK: cmpxchg.atomic:
; CHECK: cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic
; CHECK: cmpxchg.end:
; CHECK-NEXT: ret void
  %pair = cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic, align 4
  ret void
}

; Exclusions must survive widening a partword cmpxchg and converting a pointer
; cmpxchg to integer form, or those conversions would reintroduce local checks.
define { i8, i1 } @noalias_i8(ptr %p, i8 %cmp, i8 %new) {
; CHECK-LABEL: define { i8, i1 } @noalias_i8(
; CHECK-NOT: isspacep
; CHECK: cmpxchg ptr {{%.*}}, i32 {{%.*}}, i32 {{%.*}} monotonic monotonic, align 4, !noalias.addrspace ![[NOTLOCAL]]
; CHECK-NOT: isspacep
; CHECK: ret { i8, i1 }
  %pair = cmpxchg ptr %p, i8 %cmp, i8 %new monotonic monotonic, align 1, !noalias.addrspace !0
  ret { i8, i1 } %pair
}

define { ptr, i1 } @noalias_pointer(ptr %p, ptr %cmp, ptr %new) {
; CHECK-LABEL: define { ptr, i1 } @noalias_pointer(
; CHECK-NOT: isspacep
; CHECK: cmpxchg ptr %p, i64 {{%.*}}, i64 {{%.*}} monotonic monotonic, align 8, !noalias.addrspace ![[NOTLOCAL]]
; CHECK-NOT: isspacep
; CHECK: ret { ptr, i1 }
  %pair = cmpxchg ptr %p, ptr %cmp, ptr %new monotonic monotonic, align 8, !noalias.addrspace !0
  ret { ptr, i1 } %pair
}

define { i32, i1 } @global_i32(ptr addrspace(1) %p) {
; CHECK-LABEL: define { i32, i1 } @global_i32(
; CHECK-NOT: isspacep
; CHECK: cmpxchg ptr addrspace(1) %p, i32 0, i32 1 monotonic monotonic
; CHECK-NEXT: ret
  %pair = cmpxchg ptr addrspace(1) %p, i32 0, i32 1 monotonic monotonic, align 4
  ret { i32, i1 } %pair
}

define { i32, i1 } @shared_i32(ptr addrspace(3) %p) {
; CHECK-LABEL: define { i32, i1 } @shared_i32(
; CHECK-NOT: isspacep
; CHECK: cmpxchg ptr addrspace(3) %p, i32 0, i32 1 monotonic monotonic
; CHECK-NEXT: ret
  %pair = cmpxchg ptr addrspace(3) %p, i32 0, i32 1 monotonic monotonic, align 4
  ret { i32, i1 } %pair
}

; Excluding local memory in metadata avoids the dynamic check entirely.
define { i32, i1 } @noalias_local(ptr %p) {
; CHECK-LABEL: define { i32, i1 } @noalias_local(
; CHECK-NOT: isspacep
; CHECK: cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic, align 4, !noalias.addrspace ![[NOTLOCAL]]
; CHECK-NEXT: ret
  %pair = cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic, align 4, !noalias.addrspace !0
  ret { i32, i1 } %pair
}

define { i32, i1 } @noalias_wrapped_local(ptr %p) {
; CHECK-LABEL: define { i32, i1 } @noalias_wrapped_local(
; CHECK-NOT: isspacep
; CHECK: cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic
; CHECK-NEXT: ret
  %pair = cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic, align 4, !noalias.addrspace !1
  ret { i32, i1 } %pair
}

; Preserve disjoint exclusions, rather than widening them across local memory.
define { i32, i1 } @noalias_disjoint(ptr %p) {
; CHECK-LABEL: define { i32, i1 } @noalias_disjoint(
; CHECK: call i1 @llvm.nvvm.isspacep.local(ptr %p)
; CHECK: cmpxchg.atomic:
; CHECK-NEXT: {{%.*}} = cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic, align 4, !noalias.addrspace ![[DISJOINT:[0-9]+]]
  %pair = cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic, align 4, !noalias.addrspace !2
  ret { i32, i1 } %pair
}

; Merge contiguous ranges so the new metadata passes the verifier.
define { i32, i1 } @noalias_adjacent(ptr %p) {
; CHECK-LABEL: define { i32, i1 } @noalias_adjacent(
; CHECK: call i1 @llvm.nvvm.isspacep.local(ptr %p)
; CHECK: cmpxchg.atomic:
; CHECK-NEXT: {{%.*}} = cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic, align 4, !noalias.addrspace ![[ADJACENT:[0-9]+]]
  %pair = cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic, align 4, !noalias.addrspace !3
  ret { i32, i1 } %pair
}

define { i32, i1 } @noalias_wrapped_other(ptr %p) {
; CHECK-LABEL: define { i32, i1 } @noalias_wrapped_other(
; CHECK: call i1 @llvm.nvvm.isspacep.local(ptr %p)
; CHECK: cmpxchg.atomic:
; CHECK-NEXT: {{%.*}} = cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic, align 4, !noalias.addrspace ![[WRAPPED:[0-9]+]]
  %pair = cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic, align 4, !noalias.addrspace !4
  ret { i32, i1 } %pair
}

; Revisit both new atomic blocks when expanding multiple operations in one block.
define { i32, i1 } @two_operations(ptr %p, ptr %q) {
; CHECK-LABEL: define { i32, i1 } @two_operations(
; CHECK-COUNT-2: call i1 @llvm.nvvm.isspacep.local
; CHECK: ret { i32, i1 }
  %first = cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic, align 4
  %old = extractvalue { i32, i1 } %first, 0
  %second = cmpxchg ptr %q, i32 %old, i32 2 monotonic monotonic, align 4
  ret { i32, i1 } %second
}

; CHECK-DAG: ![[NOTLOCAL]] = !{i32 5, i32 6}
; CHECK-DAG: ![[DISJOINT]] = !{i32 1, i32 2, i32 5, i32 6, i32 7, i32 8}
; CHECK-DAG: ![[ADJACENT]] = !{i32 4, i32 7}
; CHECK-DAG: ![[WRAPPED]] = !{i32 5, i32 4}

!0 = !{i32 5, i32 6}
!1 = !{i32 5, i32 4}
!2 = !{i32 1, i32 2, i32 7, i32 8}
!3 = !{i32 4, i32 5, i32 6, i32 7}
!4 = !{i32 6, i32 4}
