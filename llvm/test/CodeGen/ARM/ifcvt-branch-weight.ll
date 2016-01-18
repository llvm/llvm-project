; RUN: llc < %s -mtriple=thumbv8 -print-machineinstrs=if-converter -arm-atomic-cfg-tidy=0 -o /dev/null 2>&1 | FileCheck %s

%struct.S = type { i8* (i8*)*, [1 x i8] }
define internal zeroext i8 @bar(%struct.S* %x, %struct.S* nocapture %y) nounwind readonly {
entry:
  %0 = getelementptr inbounds %struct.S, %struct.S* %x, i32 0, i32 1, i32 0
  %1 = load i8, i8* %0, align 1
  %2 = zext i8 %1 to i32
  %3 = and i32 %2, 112
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %return, label %bb

bb:
  %5 = getelementptr inbounds %struct.S, %struct.S* %y, i32 0, i32 1, i32 0
  %6 = load i8, i8* %5, align 1
  %7 = zext i8 %6 to i32
  %8 = and i32 %7, 112
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %return, label %bb2

; CHECK: BB#2: derived from LLVM BB %bb2
; CHECK: Successors according to CFG: BB#3(192) BB#4(192)

bb2:
  %v10 = icmp eq i32 %3, 16
  br i1 %v10, label %bb4, label %bb3, !prof !0

bb3:
  %v11 = icmp eq i32 %8, 16
  br i1 %v11, label %bb4, label %return, !prof !1

bb4:
  %v12 = ptrtoint %struct.S* %x to i32
  %phitmp = trunc i32 %v12 to i8
  ret i8 %phitmp

return:
  ret i8 1
}

!0 = !{!"branch_weights", i32 4, i32 12}
!1 = !{!"branch_weights", i32 8, i32 16}
