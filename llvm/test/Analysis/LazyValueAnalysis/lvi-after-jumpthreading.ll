; RUN: opt < %s -jump-threading -print-lvi-after-jump-threading -disable-output 2>&1 | FileCheck %s

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

; Merging cont block into do block. Make sure that we do not incorrectly have the cont
; LVI info as LVI info for the beginning of do block. LVI info for %i is Range[0,1)
; at beginning of cont Block, which is incorrect at the beginning of do block.
define i32 @test3(i32 %i, i1 %f, i32 %n) {
; CHECK-LABEL: LVI for function 'test3':
; CHECK-LABEL: entry
; CHECK:  ; LatticeVal for: 'i32 %i' is: overdefined
; CHECK: %c = icmp ne i32 %i, -2134 
; CHECK: br i1 %c, label %cont, label %exit
entry:
  %c = icmp ne i32 %i, -2134
  br i1 %c, label %do, label %exit

exit:
  %c1 = icmp ne i32 %i, -42
  br i1 %c1, label %exit2, label %exit

; CHECK-LABEL: cont:
; Here cont is merged to do and i is any value except -2134.
; i is not the single value: zero.
; CHECK-NOT:  ; LatticeVal for: 'i32 %i' is: constantrange<0, 1>
; CHECK:      ; LatticeVal for: 'i32 %i' is: constantrange<-2133, -2134>
; CHECK:      ; LatticeVal for: '  %cond.0 = icmp sgt i32 %i, 0' in BB: '%cont' is: overdefined
; CHECK:   %cond.0 = icmp sgt i32 %i, 0
; CHECK:   %consume = call i32 @consume
; CHECK:   %cond = icmp eq i32 %i, 0
; CHECK:   call void (i1, ...) @llvm.experimental.guard(i1 %cond)
; CHECK:   %cond.3 = icmp sgt i32 %i, %n
; CHECK:   br i1 %cond.3, label %exit2, label %exit
cont:
  %cond.3 = icmp sgt i32 %i, %n
  br i1 %cond.3, label %exit2, label %exit

do:
  %cond.0 = icmp sgt i32 %i, 0
  %consume = call i32 @consume(i1 %cond.0)
  %cond = icmp eq i32 %i, 0
  call void (i1, ...) @llvm.experimental.guard(i1 %cond) [ "deopt"() ]
  %cond.2 = icmp sgt i32 %i, 0
  br i1 %cond.2, label %exit, label %cont
  
exit2:
; CHECK-LABEL: exit2:
; LatticeVal for: 'i32 %i' is: constantrange<-2134, 1>
  ret i32 30
}

; FIXME: We should be able to merge cont into do.
; When we do so, LVI for cont cannot be the one for the merged do block.
define i32 @test4(i32 %i, i1 %f, i32 %n) {
; CHECK-LABEL: LVI for function 'test4':
entry:
  %c = icmp ne i32 %i, -2134
  br i1 %c, label %do, label %exit

exit:                                             ; preds = %do, %cont, %exit, %entry
  %c1 = icmp ne i32 %i, -42
  br i1 %c1, label %exit2, label %exit

cont:                                             ; preds = %do
; CHECK-LABEL: cont:
; CHECK:  ; LatticeVal for: 'i1 %f' is: constantrange<-1, 0>
; CHECK: call void @dummy(i1 %f)
  call void @dummy(i1 %f)
  br label %exit2

do:                                               ; preds = %entry
; CHECK-LABEL: do:
; CHECK:  ; LatticeVal for: 'i1 %f' is: overdefined
; CHECK: call void @dummy(i1 %f)
; CHECK: br i1 %cond, label %exit, label %cont
  call void @dummy(i1 %f)
  %consume = call i32 @exit()
  call void @llvm.assume(i1 %f)
  %cond = icmp eq i1 %f, false
  br i1 %cond, label %exit, label %cont

exit2:                                            ; preds = %cont, %exit
  ret i32 30
}

declare i32 @exit()
declare i32 @consume(i1)
declare void @llvm.assume(i1) nounwind
declare void @dummy(i1) nounwind
declare void @llvm.experimental.guard(i1, ...)
