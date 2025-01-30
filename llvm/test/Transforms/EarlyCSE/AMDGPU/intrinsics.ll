; RUN: opt < %s -S -mtriple=amdgcn-- -passes=early-cse -earlycse-debug-hash | FileCheck %s

; CHECK-LABEL: @no_cse
; CHECK: call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 0, i32 0)
; CHECK: call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 4, i32 0)
define void @no_cse(ptr addrspace(1) %out, <4 x i32> %in) {
  %a = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 0, i32 0)
  %b = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 4, i32 0)
  %c = add i32 %a, %b
  store i32 %c, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @cse_zero_offset
; CHECK: [[CSE:%[a-z0-9A-Z]+]] = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 0, i32 0)
; CHECK: add i32 [[CSE]], [[CSE]]
define void @cse_zero_offset(ptr addrspace(1) %out, <4 x i32> %in) {
  %a = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 0, i32 0)
  %b = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 0, i32 0)
  %c = add i32 %a, %b
  store i32 %c, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @cse_nonzero_offset
; CHECK: [[CSE:%[a-z0-9A-Z]+]] = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 4, i32 0)
; CHECK: add i32 [[CSE]], [[CSE]]
define void @cse_nonzero_offset(ptr addrspace(1) %out, <4 x i32> %in) {
  %a = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 4, i32 0)
  %b = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 4, i32 0)
  %c = add i32 %a, %b
  store i32 %c, ptr addrspace(1) %out
  ret void
}

define i32 @readanylane_readanylane_divergent_block(i1 %cond, i32 %arg) {
; CHECK-LABEL: define i32 @readanylane_readanylane_divergent_block(
; CHECK-SAME: i1 [[COND:%.*]], i32 [[ARG:%.*]]) {
; CHECK-NEXT:  [[_ENTRY:.*:]]
; CHECK-NEXT:    [[READ0:%.*]] = call i32 @llvm.amdgcn.readanylane.i32(i32 [[ARG]])
; CHECK-NEXT:    br i1 [[COND]], label %[[DOTTHEN:.*]], [[DOTEXIT:label %.*]]
; CHECK:       [[_THEN:.*:]]
; CHECK-NEXT:    [[READ1:%.*]] = call i32 @llvm.amdgcn.readanylane.i32(i32 [[ARG]])
; CHECK-NEXT:    br [[DOTEXIT]]
; CHECK:       [[_EXIT:.*:]]
; CHECK-NEXT:    [[RESULT:%.*]] = phi i32 [ [[READ0]], [[DOTENTRY:%.*]] ], [ [[READ1]], %[[DOTTHEN]] ]
; CHECK-NEXT:    ret i32 [[RESULT]]
;
.entry:
  %read0 = call i32 @llvm.amdgcn.readanylane.i32(i32 %arg)
  br i1 %cond, label %.then, label %.exit

.then:
  %read1 = call i32 @llvm.amdgcn.readanylane.i32(i32 %arg)
  br label %.exit

.exit:
  %result = phi i32 [ %read0, %.entry ], [ %read1, %.then ]
  ret i32 %result
}

declare i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> nocapture, i32, i32)
