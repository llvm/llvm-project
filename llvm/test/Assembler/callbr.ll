; RUN: llvm-as < %s | llvm-dis | FileCheck %s

declare void @llvm.amdgcn.kill(i1)

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
