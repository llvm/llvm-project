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
  br label %loop
; CHECK-LABEL: backedge:
; CHECK-NEXT: ; CachedLatticeValues for: '  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]'
; CHECK-DAG: ; at beginning of BasicBlock: '%backedge' LatticeVal: 'constantrange<0, 400>'
; CHECK-NEXT: %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
; CHECK-NEXT: ; CachedLatticeValues for: '  %iv.next = add nsw i32 %iv, 1'
; CHECK-NEXT: ; at beginning of BasicBlock: '%backedge' LatticeVal: 'constantrange<1, 401>'
; CHECK-NEXT: %iv.next = add nsw i32 %iv, 1
; CHECK-NEXT:  %cont = icmp slt i32 %iv.next, 400
; CHECK-NEXT: br i1 %cont, label %backedge, label %exit

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
; CHECK-LABEL: ; OverDefined values for block are:
; CHECK-NEXT: ;i32 %n
; CHECK-NEXT: br label %loop
entry:
  br label %loop

; CHECK-LABEL: loop:
; CHECK-LABEL: ; OverDefined values for block are:
; CHECK-NEXT: ; %iv2 = phi i32 [ %n, %entry ], [ %iv2.next, %backedge ]
; CHECK-NEXT: ; CachedLatticeValues for: '  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]'
; CHECK-DAG: ; at beginning of BasicBlock: '%loop' LatticeVal: 'constantrange<0, -2147483647>'
; CHECK-DAG: ; at beginning of BasicBlock: '%backedge' LatticeVal: 'constantrange<0, -2147483648>'
; CHECK-NEXT: %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
; CHECK: %cnd = and i1 %cnd1, %cnd2
; CHECK: br i1 %cnd, label %backedge, label %exit
loop:
  %iv = phi i32 [0, %entry], [%iv.next, %backedge]
  %iv2 = phi i32 [%n, %entry], [%iv2.next, %backedge]
  %cnd1 = icmp sge i32 %iv, 0
  %cnd2 = icmp sgt i32 %iv2, 0
  %cnd = and i1 %cnd1, %cnd2
  br i1 %cnd, label %backedge, label %exit

; CHECK-LABEL: backedge:
; CHECK-NEXT: ; CachedLatticeValues for: '  %iv.next = add nsw i32 %iv, 1'
; CHECK-NEXT: ; at beginning of BasicBlock: '%backedge' LatticeVal: 'constantrange<1, -2147483647>'
; CHECK-NEXT: %iv.next = add nsw i32 %iv, 1
; CHECK-NEXT: %iv2.next = sub nsw i32 %iv2, 1
; CHECK: %cont = and i1 %cont1, %cont2
; CHECK: br i1 %cont, label %loop, label %exit
backedge:
  %iv.next = add nsw i32 %iv, 1
  %iv2.next = sub nsw i32 %iv2, 1
  %cont1 = icmp slt i32 %iv.next, 400
  %cont2 = icmp sgt i32 %iv2.next, 0
  %cont = and i1 %cont1, %cont2
  br i1 %cont, label %loop, label %exit

exit:
  ret i8 0
}
