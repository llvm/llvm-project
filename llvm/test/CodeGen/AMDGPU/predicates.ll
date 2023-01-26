; RUN: llc -spec-exec-max-speculation-cost=0 -march=r600 -r600-ir-structurize=0 -mcpu=redwood < %s | FileCheck %s

; These tests make sure the compiler is optimizing branches using predicates
; when it is legal to do so.

; CHECK-LABEL: {{^}}simple_if:
; CHECK: PRED_SET{{[EGN][ET]*}}_INT * Pred,
; CHECK: LSHL * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, 1, Pred_sel
define amdgpu_kernel void @simple_if(ptr addrspace(1) %out, i32 %in) {
entry:
  %cmp0 = icmp sgt i32 %in, 0
  br i1 %cmp0, label %IF, label %ENDIF

IF:
  %tmp1 = shl i32 %in, 1
  br label %ENDIF

ENDIF:
  %tmp2 = phi i32 [ %in, %entry ], [ %tmp1, %IF ]
  store i32 %tmp2, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}simple_if_else:
; CHECK: PRED_SET{{[EGN][ET]*}}_INT * Pred,
; CHECK: LSH{{[LR] \* T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, 1, Pred_sel
; CHECK: LSH{{[LR] \* T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, 1, Pred_sel
define amdgpu_kernel void @simple_if_else(ptr addrspace(1) %out, i32 %in) {
entry:
  %0 = icmp sgt i32 %in, 0
  br i1 %0, label %IF, label %ELSE

IF:
  %1 = shl i32 %in, 1
  br label %ENDIF

ELSE:
  %2 = lshr i32 %in, 1
  br label %ENDIF

ENDIF:
  %3 = phi i32 [ %1, %IF ], [ %2, %ELSE ]
  store i32 %3, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}nested_if:
; CHECK: ALU_PUSH_BEFORE
; CHECK: JUMP
; CHECK: POP
; CHECK: PRED_SET{{[EGN][ET]*}}_INT * Exec
; CHECK: PRED_SET{{[EGN][ET]*}}_INT * Pred,
; CHECK: LSHL * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, 1, Pred_sel
define amdgpu_kernel void @nested_if(ptr addrspace(1) %out, i32 %in) {
entry:
  %0 = icmp sgt i32 %in, 0
  br i1 %0, label %IF0, label %ENDIF

IF0:
  %1 = add i32 %in, 10
  %2 = icmp sgt i32 %1, 0
  br i1 %2, label %IF1, label %ENDIF

IF1:
  %3 = shl i32  %1, 1
  br label %ENDIF

ENDIF:
  %4 = phi i32 [%in, %entry], [%1, %IF0], [%3, %IF1]
  store i32 %4, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}nested_if_else:
; CHECK: ALU_PUSH_BEFORE
; CHECK: JUMP
; CHECK: POP
; CHECK: PRED_SET{{[EGN][ET]*}}_INT * Exec
; CHECK: PRED_SET{{[EGN][ET]*}}_INT * Pred,
; CHECK: LSH{{[LR] \* T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, 1, Pred_sel
; CHECK: LSH{{[LR] \* T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, 1, Pred_sel
define amdgpu_kernel void @nested_if_else(ptr addrspace(1) %out, i32 %in) {
entry:
  %0 = icmp sgt i32 %in, 0
  br i1 %0, label %IF0, label %ENDIF

IF0:
  %1 = add i32 %in, 10
  %2 = icmp sgt i32 %1, 0
  br i1 %2, label %IF1, label %ELSE1

IF1:
  %3 = shl i32  %1, 1
  br label %ENDIF

ELSE1:
  %4 = lshr i32 %in, 1
  br label %ENDIF

ENDIF:
  %5 = phi i32 [%in, %entry], [%3, %IF1], [%4, %ELSE1]
  store i32 %5, ptr addrspace(1) %out
  ret void
}
