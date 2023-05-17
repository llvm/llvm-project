; RUN: opt -passes=inline -S %s -inline-threshold=20 2>&1 | FileCheck %s

%struct.nodemask_t = type { [16 x i64] }
@numa_nodes_parsed = external constant %struct.nodemask_t, align 8

declare void @foo()
declare i64 @llvm.objectsize.i64.p0(ptr, i1 immarg, i1 immarg, i1 immarg)

; Test that we inline @callee into @caller.
define i64 @caller() {
; CHECK-LABEL: @caller(
; CHECK-NEXT:    [[TMP1:%.*]] = tail call i64 @llvm.objectsize.i64.p0(ptr @numa_nodes_parsed, i1 false, i1 false, i1 false)
; CHECK-NEXT:    [[TMP2:%.*]] = icmp uge i64 [[TMP1]], 128
; CHECK-NEXT:    br i1 [[TMP2]], label %[[CALLEE_EXIT:.*]], label %[[HANDLER_TYPE_MISMATCH94_I:.*]]
; CHECK:       [[HANDLER_TYPE_MISMATCH94_I]]:
; CHECK-NEXT:    call void @foo()
; CHECK-NEXT:    call void @foo()
; CHECK-NEXT:    call void @foo()
; CHECK-NEXT:    call void @foo()
; CHECK-NEXT:    call void @foo()
; CHECK-NEXT:    call void @foo()
; CHECK-NEXT:    br label %[[CALLEE_EXIT]]
; CHECK:       [[CALLEE_EXIT]]:
; CHECK-NEXT:    ret i64 [[TMP1]]
;
  %1 = tail call i64 @callee()
  ret i64 %1
}

; Testing the InlineCost of the call to @llvm.objectsize.i64.p0i8.
; Do not change the linkage of @callee; that will give it a severe discount in
; cost (LastCallToStaticBonus).
define i64 @callee() {
  %1 = tail call i64 @llvm.objectsize.i64.p0(ptr @numa_nodes_parsed, i1 false, i1 false, i1 false)
  %2 = icmp uge i64 %1, 128
  br i1 %2, label %cont95, label %handler.type_mismatch94

handler.type_mismatch94:
  call void @foo()
  call void @foo()
  call void @foo()
  call void @foo()
  call void @foo()
  call void @foo()
  br label %cont95

cont95:
  ret i64 %1
}

