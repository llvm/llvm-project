; REQUIRES: asserts
; RUN: opt -S -passes=dfa-jump-threading -dfa-max-path-length=3 %s \
; RUN:  2>&1 -disable-output -debug-only=dfa-jump-threading | FileCheck %s
; RUN: opt -S -passes=dfa-jump-threading -dfa-max-num-visited-paths=3 %s \
; RUN:  2>&1 -disable-output -debug-only=dfa-jump-threading | FileCheck %s

; Make the path
;   < case1 case1.1 case1.2 case1.3 case1.4 for.inc for.body > [ 3, case1 ]
; too long so that it is not jump-threaded.
define i32 @max_path_length(i32 %num) {
; CHECK-NOT: 3, case1
; CHECK: < case2 for.inc for.body > [ 1, for.inc ]
; CHECK-NEXT: < for.inc for.body > [ 1, for.inc ]
; CHECK-NEXT: < case2 sel.si.unfold.false for.inc for.body > [ 2, sel.si.unfold.false ]
; CHECK-NEXT: DFA-JT: Renaming non-local uses of: 
entry:
  br label %for.body

for.body:
  %count = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %state = phi i32 [ 1, %entry ], [ %state.next, %for.inc ]
  switch i32 %state, label %for.inc [
  i32 1, label %case1
  i32 2, label %case2
  ]

case1:
  %case1.state.next = phi i32 [ 3, %for.body ]
  br label %case1.1

case1.1:
  br label %case1.2

case1.2:
  br label %case1.3

case1.3:
  br label %case1.4

case1.4:
  br label %for.inc

case2:
  %cmp = icmp eq i32 %count, 50
  %sel = select i1 %cmp, i32 1, i32 2
  br label %for.inc

for.inc:
  %state.next = phi i32 [ %sel, %case2 ], [ 1, %for.body ], [ %case1.state.next, %case1.4 ]
  %inc = add nsw i32 %count, 1
  %cmp.exit = icmp slt i32 %inc, %num
  br i1 %cmp.exit, label %for.body, label %for.end

for.end:
  ret i32 0
}
