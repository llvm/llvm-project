; RUN: opt -S -passes=kcfi %s | FileCheck %s

; CHECK-LABEL: define void @f1(
define void @f1(ptr noundef %x) {
  ; CHECK:      %[[#GEPI:]] = getelementptr inbounds i32, ptr %x, i32 -1
  ; CHECK-NEXT: %[[#LOAD:]] = load i32, ptr %[[#GEPI]], align 4
  ; CHECK-NEXT: %[[#ICMP:]] = icmp ne i32 %[[#LOAD]], 12345678
  ; CHECK-NEXT: br i1 %[[#ICMP]], label %[[#TRAP:]], label %[[#CALL:]], !prof ![[#WEIGHTS:]]
  ; CHECK:      [[#TRAP]]:
  ; CHECK-NEXT: call void @llvm.trap()
  ; CHECK-NEXT: br label %[[#CALL]]
  ; CHECK:      [[#CALL]]:
  ; CHECK-NEXT: call void %x()
  ; CHECK-NOT:  [ "kcfi"(i32 12345678) ]
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
; CHECK: ![[#WEIGHTS]] = !{!"branch_weights", i32 1, i32 1048575}
