; RUN: llvm-as < %s | llvm-dis | FileCheck %s

declare void @llvm.amdgcn.kill(i1)
declare void @llvm.amdgcn.unreachable()

define void @test_kill(i1 %c) {
; CHECK-LABEL: define void @test_kill(
; CHECK-SAME: i1 [[C:%.*]]) {
; CHECK-NEXT:    callbr void @llvm.amdgcn.kill(i1 [[C]])
; CHECK-NEXT:            to label %[[CONT:.*]] [label %kill]
; CHECK:       [[KILL:.*:]]
; CHECK-NEXT:    unreachable
; CHECK:       [[CONT]]:
; CHECK-NEXT:    ret void
;
  callbr void @llvm.amdgcn.kill(i1 %c) to label %cont [label %kill]
kill:
  unreachable
cont:
  ret void
}

define void @test_kill_unreachable_not_first(i1 %c, ptr %p) {
; CHECK-LABEL: define void @test_kill_unreachable_not_first(
; CHECK-SAME: i1 [[C:%.*]], ptr [[P:%.*]]) {
; CHECK-NEXT:    callbr void @llvm.amdgcn.kill(i1 [[C]])
; CHECK-NEXT:            to label %[[CONT:.*]] [label %kill]
; CHECK:       [[KILL:.*:]]
; CHECK-NEXT:    store i32 0, ptr [[P]], align 4
; CHECK-NEXT:    unreachable
; CHECK:       [[CONT]]:
; CHECK-NEXT:    ret void
;
  callbr void @llvm.amdgcn.kill(i1 %c) to label %cont [label %kill]
kill:
  store i32 0, ptr %p, align 4
  unreachable
cont:
  ret void
}

define void @test_kill_amdgcn_unreachable_not_first(i1 %c, ptr %p) {
; CHECK-LABEL: define void @test_kill_amdgcn_unreachable_not_first(
; CHECK-SAME: i1 [[C:%.*]], ptr [[P:%.*]]) {
; CHECK-NEXT:    callbr void @llvm.amdgcn.kill(i1 [[C]])
; CHECK-NEXT:            to label %[[CONT:.*]] [label %kill]
; CHECK:       [[KILL:.*:]]
; CHECK-NEXT:    store i32 0, ptr [[P]], align 4
; CHECK-NEXT:    call void @llvm.amdgcn.unreachable()
; CHECK-NEXT:    ret void
; CHECK:       [[CONT]]:
; CHECK-NEXT:    ret void
;
  callbr void @llvm.amdgcn.kill(i1 %c) to label %cont [label %kill]
kill:
  store i32 0, ptr %p, align 4
  call void @llvm.amdgcn.unreachable()
  ret void
cont:
  ret void
}
