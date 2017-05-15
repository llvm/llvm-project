; RUN: llc -mtriple=armv7a   %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7m %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv6m %s -o - | FileCheck %s

@arr = internal unnamed_addr constant [4 x i32] [i32 305419896, i32 -1430532899, i32 -2023406815, i32 -573785174], align 4

; Check that the adr of arr is hoisted out of the loop
; CHECK: adr [[REG:r[0-9]+]], .LCP
; CHECK: .LBB
; CHECK-NOT adr
; CHECK: ldr{{(.w)?}} {{r[0-9]+}}, {{\[}}[[REG]],

define void @fn(i32 %n, i32* %p) {
entry:
  %cmp8 = icmp sgt i32 %n, 0
  br i1 %cmp8, label %for.body, label %for.cond.cleanup

for.body:
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds [4 x i32], [4 x i32]* @arr, i32 0, i32 %0
  %1 = load i32, i32* %arrayidx1, align 4
  store i32 %1, i32* %arrayidx, align 4
  %inc = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}
