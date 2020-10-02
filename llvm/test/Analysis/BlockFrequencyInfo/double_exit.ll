; RUN: opt < %s -analyze -block-freq -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s

; CHECK-LABEL: Printing analysis {{.*}} for function 'double_exit':
; CHECK-NEXT: block-frequency-info: double_exit
define i32 @double_exit(i32 %N) {
; Mass = 1
; Frequency = 1
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  br label %outer

; Mass = 1
; Backedge mass = 1/3, exit mass = 2/3
; Loop scale = 3/2
; Pseudo-edges = exit
; Pseudo-mass = 1
; Frequency = 1*3/2*1 = 3/2
; CHECK-NEXT: outer: float = 1.5,
outer:
  %I.0 = phi i32 [ 0, %entry ], [ %inc6, %outer.inc ]
  %Return.0 = phi i32 [ 0, %entry ], [ %Return.1, %outer.inc ]
  %cmp = icmp slt i32 %I.0, %N
  br i1 %cmp, label %inner, label %exit, !prof !2 ; 2:1

; Mass = 1
; Backedge mass = 3/5, exit mass = 2/5
; Loop scale = 5/2
; Pseudo-edges = outer.inc @ 1/5, exit @ 1/5
; Pseudo-mass = 2/3
; Frequency = 3/2*1*5/2*2/3 = 5/2
; CHECK-NEXT: inner: float = 2.5,
inner:
  %Return.1 = phi i32 [ %Return.0, %outer ], [ %call4, %inner.inc ]
  %J.0 = phi i32 [ %I.0, %outer ], [ %inc, %inner.inc ]
  %cmp2 = icmp slt i32 %J.0, %N
  br i1 %cmp2, label %inner.body, label %outer.inc, !prof !1 ; 4:1

; Mass = 4/5
; Frequency = 5/2*4/5 = 2
; CHECK-NEXT: inner.body: float = 2.0,
inner.body:
  %call = call i32 @c2(i32 %I.0, i32 %J.0)
  %tobool = icmp ne i32 %call, 0
  br i1 %tobool, label %exit, label %inner.inc, !prof !0 ; 3:1

; Mass = 3/5
; Frequency = 5/2*3/5 = 3/2
; CHECK-NEXT: inner.inc: float = 1.5,
inner.inc:
  %call4 = call i32 @logic2(i32 %Return.1, i32 %I.0, i32 %J.0)
  %inc = add nsw i32 %J.0, 1
  br label %inner

; Mass = 1/3
; Frequency = 3/2*1/3 = 1/2
; CHECK-NEXT: outer.inc: float = 0.5,
outer.inc:
  %inc6 = add nsw i32 %I.0, 1
  br label %outer

; Mass = 1
; Frequency = 1
; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
exit:
  %Return.2 = phi i32 [ %Return.1, %inner.body ], [ %Return.0, %outer ]
  ret i32 %Return.2
}

!0 = !{!"branch_weights", i32 1, i32 3}
!1 = !{!"branch_weights", i32 4, i32 1}
!2 = !{!"branch_weights", i32 2, i32 1}

declare i32 @c2(i32, i32)
declare i32 @logic2(i32, i32, i32)

; CHECK-LABEL: Printing analysis {{.*}} for function 'double_exit_in_loop':
; CHECK-NEXT: block-frequency-info: double_exit_in_loop
define i32 @double_exit_in_loop(i32 %N) {
; Mass = 1
; Frequency = 1
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  br label %outer

; Mass = 1
; Backedge mass = 1/2, exit mass = 1/2
; Loop scale = 2
; Pseudo-edges = exit
; Pseudo-mass = 1
; Frequency = 1*2*1 = 2
; CHECK-NEXT: outer: float = 2.0,
outer:
  %I.0 = phi i32 [ 0, %entry ], [ %inc12, %outer.inc ]
  %Return.0 = phi i32 [ 0, %entry ], [ %Return.3, %outer.inc ]
  %cmp = icmp slt i32 %I.0, %N
  br i1 %cmp, label %middle, label %exit, !prof !3 ; 1:1

; Mass = 1
; Backedge mass = 1/3, exit mass = 2/3
; Loop scale = 3/2
; Pseudo-edges = outer.inc
; Pseudo-mass = 1/2
; Frequency = 2*1*3/2*1/2 = 3/2
; CHECK-NEXT: middle: float = 1.5,
middle:
  %J.0 = phi i32 [ %I.0, %outer ], [ %inc9, %middle.inc ]
  %Return.1 = phi i32 [ %Return.0, %outer ], [ %Return.2, %middle.inc ]
  %cmp2 = icmp slt i32 %J.0, %N
  br i1 %cmp2, label %inner, label %outer.inc, !prof !2 ; 2:1

; Mass = 1
; Backedge mass = 3/5, exit mass = 2/5
; Loop scale = 5/2
; Pseudo-edges = middle.inc @ 1/5, outer.inc @ 1/5
; Pseudo-mass = 2/3
; Frequency = 3/2*1*5/2*2/3 = 5/2
; CHECK-NEXT: inner: float = 2.5,
inner:
  %Return.2 = phi i32 [ %Return.1, %middle ], [ %call7, %inner.inc ]
  %K.0 = phi i32 [ %J.0, %middle ], [ %inc, %inner.inc ]
  %cmp5 = icmp slt i32 %K.0, %N
  br i1 %cmp5, label %inner.body, label %middle.inc, !prof !1 ; 4:1

; Mass = 4/5
; Frequency = 5/2*4/5 = 2
; CHECK-NEXT: inner.body: float = 2.0,
inner.body:
  %call = call i32 @c3(i32 %I.0, i32 %J.0, i32 %K.0)
  %tobool = icmp ne i32 %call, 0
  br i1 %tobool, label %outer.inc, label %inner.inc, !prof !0 ; 3:1

; Mass = 3/5
; Frequency = 5/2*3/5 = 3/2
; CHECK-NEXT: inner.inc: float = 1.5,
inner.inc:
  %call7 = call i32 @logic3(i32 %Return.2, i32 %I.0, i32 %J.0, i32 %K.0)
  %inc = add nsw i32 %K.0, 1
  br label %inner

; Mass = 1/3
; Frequency = 3/2*1/3 = 1/2
; CHECK-NEXT: middle.inc: float = 0.5,
middle.inc:
  %inc9 = add nsw i32 %J.0, 1
  br label %middle

; Mass = 1/2
; Frequency = 2*1/2 = 1
; CHECK-NEXT: outer.inc: float = 1.0,
outer.inc:
  %Return.3 = phi i32 [ %Return.2, %inner.body ], [ %Return.1, %middle ]
  %inc12 = add nsw i32 %I.0, 1
  br label %outer

; Mass = 1
; Frequency = 1
; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
exit:
  ret i32 %Return.0
}

!3 = !{!"branch_weights", i32 1, i32 1}

declare i32 @c3(i32, i32, i32)
declare i32 @logic3(i32, i32, i32, i32)
