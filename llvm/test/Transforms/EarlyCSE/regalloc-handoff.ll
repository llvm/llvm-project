; RUN: opt -passes=early-cse -S %s | FileCheck %s --check-prefix=CSE
; RUN: opt -passes=gvn -S %s | FileCheck %s --check-prefix=GVN
; RUN: opt -passes=simplifycfg -S %s | FileCheck %s --check-prefix=CFG

declare i32 @llvm.experimental.regalloc.handoff(i32, metadata)

define i32 @preserve_distinct_handoffs(i32 %value) {
; CSE-LABEL: @preserve_distinct_handoffs(
; CSE-NEXT:    [[FIRST:%.*]] = call i32 @llvm.experimental.regalloc.handoff(i32 [[VALUE:%.*]], metadata [[CONSTRAINT:![0-9]+]])
; CSE-NEXT:    [[SECOND:%.*]] = call i32 @llvm.experimental.regalloc.handoff(i32 [[VALUE]], metadata [[CONSTRAINT]])
; CSE-NEXT:    [[SUM:%.*]] = add i32 [[FIRST]], [[SECOND]]
; CSE-NEXT:    ret i32 [[SUM]]
;
  %first = call i32 @llvm.experimental.regalloc.handoff(
      i32 %value, metadata !0)
  %second = call i32 @llvm.experimental.regalloc.handoff(
      i32 %value, metadata !0)
  %sum = add i32 %first, %second
  ret i32 %sum
}

define i32 @preserve_unused_handoff(i32 %value) {
; CSE-LABEL: @preserve_unused_handoff(
; CSE-NEXT:    [[HANDOFF:%.*]] = call i32 @llvm.experimental.regalloc.handoff(i32 [[VALUE:%.*]], metadata [[CONSTRAINT:![0-9]+]])
; CSE-NEXT:    ret i32 [[VALUE]]
;
  %handoff = call i32 @llvm.experimental.regalloc.handoff(
      i32 %value, metadata !0)
  ret i32 %value
}

define i32 @preserve_branch_handoffs(i1 %condition, i32 %value) {
; CFG-LABEL: @preserve_branch_handoffs(
; CFG:       left:
; CFG-NEXT:    [[LEFT_VALUE:%.*]] = call i32 @llvm.experimental.regalloc.handoff(i32 [[VALUE:%.*]], metadata [[CONSTRAINT:![0-9]+]])
; CFG-NEXT:    br label [[MERGE:%.*]]
; CFG:       right:
; CFG-NEXT:    [[RIGHT_VALUE:%.*]] = call i32 @llvm.experimental.regalloc.handoff(i32 [[VALUE]], metadata [[CONSTRAINT]])
; CFG-NEXT:    br label [[MERGE]]
; CFG:       merge:
; CFG-NEXT:    [[RESULT:%.*]] = phi i32 [ [[LEFT_VALUE]], [[LEFT:%.*]] ], [ [[RIGHT_VALUE]], [[RIGHT:%.*]] ]
; CFG-NEXT:    ret i32 [[RESULT]]
;
entry:
  br i1 %condition, label %left, label %right

left:
  %left.value = call i32 @llvm.experimental.regalloc.handoff(
      i32 %value, metadata !0)
  br label %merge

right:
  %right.value = call i32 @llvm.experimental.regalloc.handoff(
      i32 %value, metadata !0)
  br label %merge

merge:
  %result = phi i32 [ %left.value, %left ], [ %right.value, %right ]
  ret i32 %result
}

define i32 @do_not_speculate_handoff(i1 %condition, i32 %value) {
; CFG-LABEL: @do_not_speculate_handoff(
; CFG-NEXT:  entry:
; CFG-NEXT:    br i1 [[CONDITION:%.*]], label [[HANDOFF_BLOCK:%.*]], label [[EXIT:%.*]]
; CFG:       handoff:
; CFG-NEXT:    [[HANDOFF:%.*]] = call i32 @llvm.experimental.regalloc.handoff(i32 [[VALUE:%.*]], metadata [[CONSTRAINT:![0-9]+]])
; CFG-NEXT:    br label [[EXIT]]
; CFG:       exit:
; CFG-NEXT:    [[RESULT:%.*]] = phi i32 [ [[VALUE]], [[ENTRY:%.*]] ], [ [[HANDOFF]], [[HANDOFF_BLOCK]] ]
; CFG-NEXT:    ret i32 [[RESULT]]
;
entry:
  br i1 %condition, label %handoff, label %exit

handoff:
  %handoff.value = call i32 @llvm.experimental.regalloc.handoff(
      i32 %value, metadata !0)
  br label %exit

exit:
  %result = phi i32 [ %value, %entry ], [ %handoff.value, %handoff ]
  ret i32 %result
}

define i32 @does_not_clobber_program_memory(ptr %ptr, i32 %value) {
; GVN-LABEL: @does_not_clobber_program_memory(
; GVN-NEXT:    [[BEFORE:%.*]] = load i32, ptr [[PTR:%.*]], align 4
; GVN-NEXT:    [[HANDOFF:%.*]] = call i32 @llvm.experimental.regalloc.handoff(i32 [[VALUE:%.*]], metadata [[CONSTRAINT:![0-9]+]])
; GVN-NEXT:    [[LOAD_SUM:%.*]] = add i32 [[BEFORE]], [[BEFORE]]
; GVN-NEXT:    [[RESULT:%.*]] = add i32 [[LOAD_SUM]], [[HANDOFF]]
; GVN-NEXT:    ret i32 [[RESULT]]
;
  %before = load i32, ptr %ptr
  %handoff = call i32 @llvm.experimental.regalloc.handoff(
      i32 %value, metadata !0)
  %after = load i32, ptr %ptr
  %load.sum = add i32 %before, %after
  %result = add i32 %load.sum, %handoff
  ret i32 %result
}

!0 = !{!"amdgpu.vgpr"}
