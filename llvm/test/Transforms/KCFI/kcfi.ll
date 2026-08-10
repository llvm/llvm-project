; RUN: opt -S -passes=kcfi %s | FileCheck --check-prefixes=CHECK,NOARM %s
; RUN: %if arm-registered-target %{ opt -S -passes=kcfi -mtriple=thumbv7m-unknown-linux-gnu %s | FileCheck --check-prefixes=CHECK,ARM %s %}

; CHECK-LABEL: define void @f1(
define void @f1(ptr noundef %x) {
  ; ARM:        %[[#FINT:]] = ptrtoint ptr %x to i32
  ; ARM-NEXT:   %[[#FAND:]] = and i32 %[[#FINT]], -2
  ; ARM-NEXT:   %[[#FPTR:]] = inttoptr i32 %[[#FAND]] to ptr
  ; ARM-NEXT:   %[[#GEPI:]] = getelementptr inbounds i32, ptr %[[#FPTR]], i32 -1
  ; NOARM:      %[[#GEPI:]] = getelementptr inbounds i32, ptr %x, i32 -1
  ; CHECK-NEXT: %[[#LOAD:]] = load i32, ptr %[[#GEPI]], align 4
  ; CHECK-NEXT: %[[#ICMP:]] = icmp ne i32 %[[#LOAD]], 12345678
  ; CHECK-NEXT: br i1 %[[#ICMP]], label %[[#TRAP:]], label %[[#CALL:]], !prof ![[#WEIGHTS:]]
  ; CHECK:      [[#TRAP]]:
  ; CHECK-NEXT: call void @llvm.debugtrap()
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
