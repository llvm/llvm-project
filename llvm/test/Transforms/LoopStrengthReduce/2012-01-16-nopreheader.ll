; RUN: opt -loop-reduce -S < %s | FileCheck %s
;
; <rdar://10701050> "Cannot split an edge from an IndirectBrInst" assert.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; while.cond197 is a dominates the simplified loop while.cond238 but
; has no with no preheader.
;
; CHECK-LABEL: @nopreheader(
; CHECK: %while.cond238
; CHECK: phi i64
; CHECK-NOT: phi
; CHECK: indirectbr
define void @nopreheader(i8* %end) nounwind {
entry:
  br label %while.cond179

while.cond179:                                    ; preds = %if.end434, %if.end369, %if.end277, %if.end165
  %s.1 = phi i8* [ undef, %if.end434 ], [ %incdec.ptr356, %if.end348 ], [ undef, %entry ]
  indirectbr i8* undef, [label %land.rhs184, label %while.end453]

land.rhs184:                                      ; preds = %while.cond179
  indirectbr i8* undef, [label %while.end453, label %while.cond197]

while.cond197:                                    ; preds = %land.rhs202, %land.rhs184
  %0 = phi i64 [ %indvar.next11, %land.rhs202 ], [ 0, %land.rhs184 ]
  indirectbr i8* undef, [label %land.rhs202, label %while.end215]

land.rhs202:                                      ; preds = %while.cond197
  %indvar.next11 = add i64 %0, 1
  indirectbr i8* undef, [label %while.end215, label %while.cond197]

while.end215:                                     ; preds = %land.rhs202, %while.cond197
  indirectbr i8* undef, [label %PREMATURE, label %if.end221]

if.end221:                                        ; preds = %while.end215
  indirectbr i8* undef, [label %while.cond238.preheader, label %lor.lhs.false227]

lor.lhs.false227:                                 ; preds = %if.end221
  indirectbr i8* undef, [label %while.cond238.preheader, label %if.else]

while.cond238.preheader:                          ; preds = %lor.lhs.false227, %if.end221
  %tmp16 = add i64 %0, 2
  indirectbr i8* undef, [label %while.cond238]

while.cond238:                                    ; preds = %land.rhs243, %while.cond238.preheader
  %1 = phi i64 [ %indvar.next15, %land.rhs243 ], [ 0, %while.cond238.preheader ]
  %tmp36 = add i64 %tmp16, %1
  %s.3 = getelementptr i8, i8* %s.1, i64 %tmp36
  %cmp241 = icmp ult i8* %s.3, %end
  indirectbr i8* undef, [label %land.rhs243, label %while.end256]

land.rhs243:                                      ; preds = %while.cond238
  %indvar.next15 = add i64 %1, 1
  indirectbr i8* undef, [label %while.end256, label %while.cond238]

while.end256:                                     ; preds = %land.rhs243, %while.cond238
  indirectbr i8* undef, [label %PREMATURE]

if.else:                                          ; preds = %lor.lhs.false227
  indirectbr i8* undef, [label %if.then297, label %if.else386]

if.then297:                                       ; preds = %if.else
  indirectbr i8* undef, [label %PREMATURE, label %if.end307]

if.end307:                                        ; preds = %if.then297
  indirectbr i8* undef, [label %if.end314, label %FAIL]

if.end314:                                        ; preds = %if.end307
  indirectbr i8* undef, [label %if.end340]

if.end340:                                        ; preds = %while.end334
  indirectbr i8* undef, [label %PREMATURE, label %if.end348]

if.end348:                                        ; preds = %if.end340
  %incdec.ptr356 = getelementptr inbounds i8, i8* undef, i64 2
  indirectbr i8* undef, [label %while.cond179]

if.else386:                                       ; preds = %if.else
  indirectbr i8* undef, [label %while.end453, label %if.end434]

if.end434:                                        ; preds = %if.then428, %if.end421
  indirectbr i8* undef, [label %while.cond179]

while.end453:                                     ; preds = %if.else386, %land.rhs184, %while.cond179
  indirectbr i8* undef, [label %PREMATURE, label %if.end459]

if.end459:                                        ; preds = %while.end453
  indirectbr i8* undef, [label %if.then465, label %FAIL]

if.then465:                                       ; preds = %if.end459
  indirectbr i8* undef, [label %return, label %if.then479]

if.then479:                                       ; preds = %if.then465
  indirectbr i8* undef, [label %return]

FAIL:                                             ; preds = %if.end459, %if.end307, %land.lhs.true142, %land.lhs.true131, %while.end
  indirectbr i8* undef, [label %DECL_FAIL]

PREMATURE:                                        ; preds = %while.end453, %while.end415, %if.end340, %while.end334, %if.then297, %while.end256, %while.end215
  indirectbr i8* undef, [label %return, label %if.then495]

if.then495:                                       ; preds = %PREMATURE
  indirectbr i8* undef, [label %return]

DECL_FAIL:                                        ; preds = %if.then488, %FAIL, %land.lhs.true99, %lor.lhs.false, %if.end83, %if.then39, %if.end
  indirectbr i8* undef, [label %return]

return:                                           ; preds = %if.then512, %if.end504, %DECL_FAIL, %if.then495, %PREMATURE, %if.then479, %if.then465, %if.then69, %if.end52, %if.end19, %if.then
  ret void
}
