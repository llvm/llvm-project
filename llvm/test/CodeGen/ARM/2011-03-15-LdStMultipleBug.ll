; RUN: llc < %s -mtriple=thumbv7-apple-darwin10 -relocation-model=pic -frame-pointer=all -mcpu=cortex-a8 | FileCheck %s

; Do not form Thumb2 ldrd / strd if the offset is not multiple of 4.
; rdar://9133587

%struct.Outer = type { i32, [2 x %"struct.Outer::Inner"] }
%"struct.Outer::Inner" = type { i32, i32, i8, i8 }

@oStruct = external global %struct.Outer, align 4

define void @main(i8 %val8) nounwind {
; CHECK-LABEL: main:
; CHECK-NOT: ldrd
; CHECK: mul
for.body.lr.ph:
  br label %for.body

for.body:                                         ; preds = %_Z14printIsNotZeroi.exit17.for.body_crit_edge, %for.body.lr.ph
  %tmp3 = phi i1 [ false, %for.body.lr.ph ], [ %phitmp27, %_Z14printIsNotZeroi.exit17.for.body_crit_edge ]
  %i.022 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %_Z14printIsNotZeroi.exit17.for.body_crit_edge ]
  %x = getelementptr %struct.Outer, %struct.Outer* @oStruct, i32 0, i32 1, i32 %i.022, i32 0
  %y = getelementptr %struct.Outer, %struct.Outer* @oStruct, i32 0, i32 1, i32 %i.022, i32 1
  %inc = add i32 %i.022, 1
  %tmp8 = load i32, i32* %x, align 4
  %tmp11 = load i32, i32* %y, align 4
  %mul = mul nsw i32 %tmp11, %tmp8
  %tobool.i14 = icmp eq i32 %mul, 0
  br i1 %tobool.i14, label %_Z14printIsNotZeroi.exit17, label %if.then.i16

if.then.i16:                                      ; preds = %_Z14printIsNotZeroi.exit
  ret void

_Z14printIsNotZeroi.exit17:                       ; preds = %_Z14printIsNotZeroi.exit
  br label %_Z14printIsNotZeroi.exit17.for.body_crit_edge

_Z14printIsNotZeroi.exit17.for.body_crit_edge:    ; preds = %_Z14printIsNotZeroi.exit17
  %b.phi.trans.insert = getelementptr %struct.Outer, %struct.Outer* @oStruct, i32 0, i32 1, i32 %inc, i32 3
  %tmp3.pre = load i8, i8* %b.phi.trans.insert, align 1
  %phitmp27 = icmp eq i8 %val8, 0
  br label %for.body

for.end:                                          ; preds = %_Z14printIsNotZeroi.exit17
  ret void
}
