; RUN: llc < %s -verify-coalescing
; PR10046
;
; PHI elimination splits the critical edge from %while.end415 to %if.end427.
; This requires updating the BNE-J terminators to a BEQ. The BNE instruction
; kills a virtual register, and LiveVariables must be updated with the new kill
; instruction.

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-n32"
target triple = "mips-ellcc-linux"

define i32 @mergesort(i8* %base, i32 %nmemb, i32 %size, i32 (i8*, i8*)* nocapture %cmp) nounwind {
entry:
  br i1 undef, label %return, label %if.end13

if.end13:                                         ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body, %if.end13
  %list1.0482 = phi i8* [ %base, %if.end13 ], [ null, %while.body ]
  br i1 undef, label %while.end415, label %while.body

while.end415:                                     ; preds = %while.body
  br i1 undef, label %if.then419, label %if.end427

if.then419:                                       ; preds = %while.end415
  %call425 = tail call i8* @memmove(i8* %list1.0482, i8* undef, i32 undef) nounwind
  br label %if.end427

if.end427:                                        ; preds = %if.then419, %while.end415
  %list2.1 = phi i8* [ undef, %if.then419 ], [ %list1.0482, %while.end415 ]
  tail call void @free(i8* %list2.1)
  unreachable

return:                                           ; preds = %entry
  ret i32 -1
}


declare i8* @memmove(i8*, i8*, i32)

declare void @free(i8*)

