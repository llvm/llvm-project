; RUN: llc -filetype=obj -o /dev/null < %s
; RUN: llc -filetype=asm < %s | FileCheck %s

; ModuleID = 'bugpoint-reduced-simplified.bc'
source_filename = "bugpoint-output-39ed676.bc"
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8m.base-arm-none-eabi"

@crc32_tab = external unnamed_addr global [256 x i32], align 4
@g_566 = external global i32**, align 4

define void @main() {
entry:
  %0 = load volatile i32**, i32*** @g_566, align 4
  br label %func_16.exit.i.i.i

lbl_1394.i.i.i.loopexit:                          ; preds = %for.cond14.preheader.us.i.i.i
  unreachable

func_16.exit.i.i.i:                               ; preds = %entry
  br i1 undef, label %for.cond7.preheader.i.lr.ph.i.i, label %for.end476.i.i.i.loopexit

for.cond7.preheader.i.lr.ph.i.i:                  ; preds = %func_16.exit.i.i.i
  br i1 undef, label %for.end476.i.i.i.loopexit, label %for.cond7.preheader.i.i.preheader.i

for.cond7.preheader.i.i.preheader.i:              ; preds = %for.cond7.preheader.i.lr.ph.i.i
  br label %for.cond14.preheader.us.i.i.i

for.cond7.preheader.i.us.i.i:                     ; preds = %for.cond7.preheader.i.lr.ph.i.i
  unreachable

for.cond14.preheader.us.i.i.i:                    ; preds = %for.inc459.us.i.i.i, %for.cond7.preheader.i.i.preheader.i
; CHECK: @ %bb.4
; CHECK-NEXT: .p2align 2
  switch i4 undef, label %func_1.exit.loopexit [
    i4 0, label %for.inc459.us.i.i.i
    i4 -5, label %for.inc459.us.i.i.i
    i4 2, label %lbl_1394.i.i.i.loopexit
    i4 3, label %for.end476.i.i.i.loopexit
  ]

for.inc459.us.i.i.i:                              ; preds = %for.cond14.preheader.us.i.i.i, %for.cond14.preheader.us.i.i.i
  br label %for.cond14.preheader.us.i.i.i

for.end476.i.i.i.loopexit:                        ; preds = %for.cond14.preheader.us.i.i.i
  unreachable

func_1.exit.loopexit:                             ; preds = %for.cond14.preheader.us.i.i.i
  %arrayidx.i63.i.i5252 = getelementptr inbounds [256 x i32], [256 x i32]* @crc32_tab, i32 0, i32 undef
  unreachable
}
