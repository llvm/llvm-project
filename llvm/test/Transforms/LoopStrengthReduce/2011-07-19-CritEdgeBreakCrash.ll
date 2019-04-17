; ModuleID = '<stdin>'
; RUN: opt < %s -loop-reduce -S | FileCheck %s
; PR10386

declare i1 @foo()
declare i8* @bar(i8*,i8*,i8*,i8*)

define void @f(i64* %a,i64* %b,i64* %c,i64* %d,i64* %e,i64* %f,i64* %g) nounwind uwtable {
entry:
  br label %b_throw.preheader

D_BREAK_LBL:                                      ; preds = %indirectgoto
  call i1 @foo()
  br label %indirectgoto

H_CONST_LBL:                                      ; preds = %indirectgoto
  call i1 @foo()
  br label %body_failed

H_MPZ_LBL:                                        ; preds = %indirectgoto
  %boo3 = call i1 @foo()
  br i1 %boo3, label %body_failed, label %while.cond.i

while.cond.i:                                     ; preds = %while.body.i15795, %if.then.i15791
  %phi = phi i64 [ %tmp20916, %while.body.i15795 ], [ 0, %H_MPZ_LBL ]
  %tmp20916 = add i64 %phi, 1
  %incdec.ptr.i15793 = getelementptr i64, i64* %pc.0.lcssa.i1610719352, i64 %tmp20916
  %boo2 = call i1 @foo()
  br i1 %boo2, label %indirectgoto, label %while.body.i15795

while.body.i15795:                                ; preds = %while.cond.i
  %tmp20.i = load i64, i64* %incdec.ptr.i15793, align 8
  %boo1 = call i1 @foo()
  br i1 %boo1, label %while.cond.i, label %body_failed

b_throw.preheader:                                ; preds = %body_failed, %entry
  call i1 @foo()
  br label %indirectgoto

body_failed:
  %pc.0.lcssa.i1610719364 = phi i64* [ %pc.0.lcssa.i1610719352, %indirectgoto ], [ %pc.0.lcssa.i1610719352, %H_MPZ_LBL ], [ %b, %H_CONST_LBL ], [ %pc.0.lcssa.i1610719352, %while.body.i15795 ]
  call i1 @foo()
  br label %b_throw.preheader

indirectgoto:
  %pc.0.lcssa.i1610719352 = phi i64* [ %pc.0.lcssa.i1610719352, %D_BREAK_LBL ], [ %a, %b_throw.preheader ], [ %d, %while.cond.i ]
  %p = call i8* @bar(i8* blockaddress(@f, %D_BREAK_LBL), i8* blockaddress(@f, %H_CONST_LBL), i8* blockaddress(@f, %H_MPZ_LBL), i8* blockaddress(@f, %body_failed) )
  indirectbr i8* %p, [label %D_BREAK_LBL, label %H_CONST_LBL, label %H_MPZ_LBL, label %body_failed]
}

; CHECK: %p = call i8* @bar(i8* blockaddress(@f, %D_BREAK_LBL), i8* blockaddress(@f, %H_CONST_LBL), i8* blockaddress(@f, %H_MPZ_LBL), i8* blockaddress(@f, %body_failed))
; CHECK: indirectbr i8* %p, [label %D_BREAK_LBL, label %H_CONST_LBL, label %H_MPZ_LBL, label %body_failed]
