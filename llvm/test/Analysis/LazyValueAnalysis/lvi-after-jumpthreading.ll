; RUN: opt < %s -jump-threading -print-lazy-value-info -disable-output 2>&1 | FileCheck %s

; Testing LVI cache after jump-threading

; Jump-threading transforms the IR below to one where
; loop and backedge basic blocks are merged into one.
; basic block (named backedge) with the branch being:
; %cont = icmp slt i32 %iv.next, 400
; br i1 %cont, label %backedge, label %exit
define i8 @test1(i32 %a, i32 %length) {
; CHECK-LABEL: LVI for function 'test1':
entry:
; CHECK-LABEL: entry:
; CHECK-NEXT:    ; LatticeVal for: 'i32 %a' is: overdefined
; CHECK-NEXT:    ; LatticeVal for: 'i32 %length' is: overdefined
  br label %loop

; CHECK-LABEL: backedge:
; CHECK-NEXT:     ; LatticeVal for: 'i32 %a' is: overdefined
; CHECK-NEXT:     ; LatticeVal for: 'i32 %length' is: overdefined
; CHECK-NEXT:     ; LatticeVal for: '  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]' in BB: '%backedge' is: constantrange<0, 400>
; CHECK-NEXT:     ; LatticeVal for: '  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]' in BB: '%exit' is: constantrange<399, 400>
; CHECK-NEXT:  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
; CHECK-NEXT:     ; LatticeVal for: '  %iv.next = add nsw i32 %iv, 1' in BB: '%backedge' is: constantrange<1, 401>
; CHECK-NEXT:     ; LatticeVal for: '  %iv.next = add nsw i32 %iv, 1' in BB: '%exit' is: constantrange<400, 401>
; CHECK-NEXT:  %iv.next = add nsw i32 %iv, 1
; CHECK-NEXT:     ; LatticeVal for: '  %cont = icmp slt i32 %iv.next, 400' in BB: '%backedge' is: overdefined
; CHECK-NEXT:     ; LatticeVal for: '  %cont = icmp slt i32 %iv.next, 400' in BB: '%exit' is: constantrange<0, -1>
; CHECK-NEXT:  %cont = icmp slt i32 %iv.next, 400
; CHECK-NOT: loop
loop:
  %iv = phi i32 [0, %entry], [%iv.next, %backedge]
  %cnd = icmp sge i32 %iv, 0
  br i1 %cnd, label %backedge, label %exit

backedge:
  %iv.next = add nsw i32 %iv, 1
  %cont = icmp slt i32 %iv.next, 400
  br i1 %cont, label %loop, label %exit

exit:
  ret i8 0
}

; Here JT does not transform the code, but LVICache is populated during the processing of blocks.
define i8 @test2(i32 %n) {
; CHECK-LABEL: LVI for function 'test2':
; CHECK-LABEL: entry:
; CHECK-NEXT:    ; LatticeVal for: 'i32 %n' is: overdefined
; CHECK-NEXT: br label %loop
entry:
  br label %loop

; CHECK-LABEL: loop:
; CHECK-NEXT:    ; LatticeVal for: 'i32 %n' is: overdefined
; CHECK-NEXT:    ; LatticeVal for: '  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]' in BB: '%loop' is: constantrange<0, -2147483647>
; CHECK-DAG:     ; LatticeVal for: '  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]' in BB: '%backedge' is: constantrange<0, -2147483648>
; CHECK-DAG:     ; LatticeVal for: '  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]' in BB: '%exit' is: constantrange<0, -2147483647>
; CHECK-NEXT:  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
loop:
  %iv = phi i32 [0, %entry], [%iv.next, %backedge]
; CHECK-NEXT:    ; LatticeVal for: '  %iv2 = phi i32 [ %n, %entry ], [ %iv2.next, %backedge ]' in BB: '%loop' is: overdefined
; CHECK-DAG:     ; LatticeVal for: '  %iv2 = phi i32 [ %n, %entry ], [ %iv2.next, %backedge ]' in BB: '%backedge' is: constantrange<1, -2147483648>
; CHECK-DAG:     ; LatticeVal for: '  %iv2 = phi i32 [ %n, %entry ], [ %iv2.next, %backedge ]' in BB: '%exit' is: overdefined
; CHECK-NEXT:  %iv2 = phi i32 [ %n, %entry ], [ %iv2.next, %backedge ]
  %iv2 = phi i32 [%n, %entry], [%iv2.next, %backedge]

; CHECK-NEXT:    ; LatticeVal for: '  %cnd1 = icmp sge i32 %iv, 0' in BB: '%loop' is: overdefined
; CHECK-DAG:     ; LatticeVal for: '  %cnd1 = icmp sge i32 %iv, 0' in BB: '%backedge' is: overdefined
; CHECK-DAG:     ; LatticeVal for: '  %cnd1 = icmp sge i32 %iv, 0' in BB: '%exit' is: overdefined
; CHECK-NEXT:  %cnd1 = icmp sge i32 %iv, 0
  %cnd1 = icmp sge i32 %iv, 0
  %cnd2 = icmp sgt i32 %iv2, 0
; CHECK:       %cnd2 = icmp sgt i32 %iv2, 0
; CHECK:         ; LatticeVal for: '  %cnd = and i1 %cnd1, %cnd2' in BB: '%loop' is: overdefined
; CHECK-DAG:     ; LatticeVal for: '  %cnd = and i1 %cnd1, %cnd2' in BB: '%backedge' is: constantrange<-1, 0>
; CHECK-DAG:     ; LatticeVal for: '  %cnd = and i1 %cnd1, %cnd2' in BB: '%exit' is: overdefined
; CHECK-NEXT:  %cnd = and i1 %cnd1, %cnd2
  %cnd = and i1 %cnd1, %cnd2
  br i1 %cnd, label %backedge, label %exit

; CHECK-LABEL: backedge:
; CHECK-NEXT:    ; LatticeVal for: 'i32 %n' is: overdefined
; CHECK-NEXT:    ; LatticeVal for: '  %iv.next = add nsw i32 %iv, 1' in BB: '%backedge' is: constantrange<1, -2147483647>
; CHECK-NEXT:  %iv.next = add nsw i32 %iv, 1
backedge:
  %iv.next = add nsw i32 %iv, 1
  %iv2.next = sub nsw i32 %iv2, 1
; CHECK:         ; LatticeVal for: '  %cont1 = icmp slt i32 %iv.next, 400' in BB: '%backedge' is: overdefined
; CHECK-NEXT:  %cont1 = icmp slt i32 %iv.next, 400
  %cont1 = icmp slt i32 %iv.next, 400
; CHECK-NEXT:    ; LatticeVal for: '  %cont2 = icmp sgt i32 %iv2.next, 0' in BB: '%backedge' is: overdefined
; CHECK-NEXT:  %cont2 = icmp sgt i32 %iv2.next, 0
  %cont2 = icmp sgt i32 %iv2.next, 0
; CHECK-NEXT:    ; LatticeVal for: '  %cont = and i1 %cont1, %cont2' in BB: '%backedge' is: overdefined
; CHECK-NEXT:  %cont = and i1 %cont1, %cont2
  %cont = and i1 %cont1, %cont2
  br i1 %cont, label %loop, label %exit

exit:
  ret i8 0
}
