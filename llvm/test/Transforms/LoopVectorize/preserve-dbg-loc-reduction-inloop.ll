; RUN: opt < %s -passes=debugify,loop-vectorize -force-vector-width=4 -prefer-inloop-reductions -S | FileCheck %s -check-prefix DEBUGLOC

; Testing the debug locations of the generated vector intstructions are same as
; their scalar counterpart.

define i32 @reduction_sum(ptr %A, ptr %B) {
; DEBUGLOC-LABEL: define i32 @reduction_sum(
; DEBUGLOC: vector.body:
; DEBUGLOC:   = load <4 x i32>, ptr %{{.+}}, align 4, !dbg ![[LOADLOC:[0-9]+]]
; DEBUGLOC:   = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %{{.+}}), !dbg ![[REDLOC:[0-9]+]]
; DEBUGLOC: loop:
; DEBUGLOC:   %[[LOAD:.+]] = load i32, ptr %{{.+}}, align 4, !dbg ![[LOADLOC]]
; DEBUGLOC:   = add i32 %{{.+}}, %[[LOAD]], !dbg ![[REDLOC]]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %red = phi i32 [ 0, %entry ], [ %red.next, %loop ]
  %gep = getelementptr inbounds i32, ptr %A, i64 %iv
  %load = load i32, ptr %gep, align 4
  %red.next = add i32 %red, %load
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 256
  br i1 %exitcond, label %exit, label %loop

exit:
  %red.lcssa = phi i32 [ %red.next, %loop ]
  ret i32 %red.lcssa
}

; DEBUGLOC: ![[LOADLOC]] = !DILocation(line: 5
; DEBUGLOC: ![[REDLOC]] = !DILocation(line: 6
