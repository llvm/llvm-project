; RUN: llc < %s -verify-machineinstrs | FileCheck %s
target triple = "x86_64-apple-macosx10.8.0"

; The critical edge from for.cond to if.end2 should be split to avoid injecting
; copies into the loop. The use of %b after the loop causes interference that
; makes a copy necessary.
; <rdar://problem/11561842>
;
; CHECK: split_loop_exit
; CHECK: %for.cond
; CHECK-NOT: mov
; CHECK: je

define i32 @split_loop_exit(i32 %a, i32 %b, i8* nocapture %p) nounwind uwtable readonly ssp {
entry:
  %cmp = icmp sgt i32 %a, 10
  br i1 %cmp, label %for.cond, label %if.end2

for.cond:                                         ; preds = %entry, %for.cond
  %p.addr.0 = phi i8* [ %incdec.ptr, %for.cond ], [ %p, %entry ]
  %incdec.ptr = getelementptr inbounds i8* %p.addr.0, i64 1
  %0 = load i8* %p.addr.0, align 1
  %tobool = icmp eq i8 %0, 0
  br i1 %tobool, label %for.cond, label %if.end2

if.end2:                                          ; preds = %for.cond, %entry
  %r.0 = phi i32 [ %a, %entry ], [ %b, %for.cond ]
  %add = add nsw i32 %r.0, %b
  ret i32 %add
}
