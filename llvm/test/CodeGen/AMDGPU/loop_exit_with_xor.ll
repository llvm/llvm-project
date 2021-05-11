; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Where the mask of lanes wanting to exit the loop on this iteration is not
; obviously already masked by exec (in this case, the xor with -1 inserted by
; control flow annotation), then lower control flow must insert an S_AND_B64
; with exec.

; GCN-LABEL: {{^}}needs_and:
; GCN: s_xor_b64 [[REG1:[^ ,]*]], {{[^ ,]*, -1$}}
; GCN: s_and_b64 [[REG2:[^ ,]*]], exec, [[REG1]]
; GCN: s_or_b64 [[REG3:[^ ,]*]], [[REG2]],
; GCN: s_andn2_b64 exec, exec, [[REG3]]

define void @needs_and(i32 %arg) {
entry:
  br label %loop

loop:
  %tmp23phi = phi i32 [ %tmp23, %endif ], [ 0, %entry ]
  %tmp23 = add nuw i32 %tmp23phi, 1
  %tmp27 = icmp ult i32 %arg, %tmp23
  br i1 %tmp27, label %then, label %endif

then:                                             ; preds = %bb
  call void @llvm.amdgcn.raw.buffer.store.f32(float undef, <4 x i32> undef, i32 0, i32 undef, i32 0)
  br label %endif

endif:                                             ; preds = %bb28, %bb
  br i1 %tmp27, label %loop, label %loopexit

loopexit:
  ret void
}

; Where the mask of lanes wanting to exit the loop on this iteration is
; obviously already masked by exec (a V_CMP), then lower control flow can omit
; the S_AND_B64 to avoid an unnecessary instruction.

; GCN-LABEL: {{^}}doesnt_need_and:
; GCN: v_cmp{{[^ ]*}} [[REG1:[^ ,]*]]
; GCN: s_or_b64 [[REG2:[^ ,]*]], [[REG1]],
; GCN: s_andn2_b64 exec, exec, [[REG2]]

define void @doesnt_need_and(i32 %arg) {
entry:
  br label %loop

loop:
  %tmp23phi = phi i32 [ %tmp23, %loop ], [ 0, %entry ]
  %tmp23 = add nuw i32 %tmp23phi, 1
  %tmp27 = icmp ult i32 %arg, %tmp23
  call void @llvm.amdgcn.raw.buffer.store.f32(float undef, <4 x i32> undef, i32 0, i32 undef, i32 0)
  br i1 %tmp27, label %loop, label %loopexit

loopexit:
  ret void
}

; Another case where the mask of lanes wanting to exit the loop is not masked
; by exec, because it is a function parameter.

; GCN-LABEL: {{^}}break_cond_is_arg:
; GCN: s_xor_b64 [[REG1:[^ ,]*]], {{[^ ,]*, -1$}}
; GCN: s_andn2_b64 exec, exec, [[REG3:[^ ,]*]]
; GCN: s_and_b64 [[REG2:[^ ,]*]], exec, [[REG1]]
; GCN: s_or_b64 [[REG3]], [[REG2]],

define void @break_cond_is_arg(i32 %arg, i1 %breakcond) {
entry:
  br label %loop

loop:
  %tmp23phi = phi i32 [ %tmp23, %endif ], [ 0, %entry ]
  %tmp23 = add nuw i32 %tmp23phi, 1
  %tmp27 = icmp ult i32 %arg, %tmp23
  br i1 %tmp27, label %then, label %endif

then:                                             ; preds = %bb
  call void @llvm.amdgcn.raw.buffer.store.f32(float undef, <4 x i32> undef, i32 0, i32 undef, i32 0)
  br label %endif

endif:                                             ; preds = %bb28, %bb
  br i1 %breakcond, label %loop, label %loopexit

loopexit:
  ret void
}

declare void @llvm.amdgcn.raw.buffer.store.f32(float, <4 x i32>, i32, i32, i32 immarg) #0

attributes #0 = { nounwind writeonly }
