; RUN: llc -mtriple=x86_64 < %s | FileCheck %s

;; Verify that custom event calls are done with proper stack alignment,
;; even in leaf functions.

@leaf_func.event_id = internal constant i32 1, align 4

define void @leaf_func() "xray-instruction-threshold"="999" "frame-pointer"="none" nounwind {
  ; CHECK-LABEL: leaf_func:
  ; CHECK-NEXT:  .Lfunc_begin0:
  ; CHECK-NEXT:  # %bb.0:
  ; CHECK-NEXT:    pushq %rax
  ; CHECK-NEXT:    movl $leaf_func.event_id, %eax
  ; CHECK-NEXT:    movl $8, %ecx
  ; CHECK-NEXT:    .p2align 1, 0x90
  ; CHECK-NEXT:  .Lxray_event_sled_0:
  call void @llvm.xray.customevent(ptr @leaf_func.event_id, i64 8)
  ret void
}

declare void @llvm.xray.customevent(ptr nocapture readonly, i64)
