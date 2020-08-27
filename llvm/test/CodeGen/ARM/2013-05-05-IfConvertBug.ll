; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-a8 | FileCheck %s
; RUN: llc < %s -mtriple=thumbv8 | FileCheck -check-prefix=CHECK-V8 %s
; RUN: llc < %s -mtriple=thumbv7 -arm-restrict-it | FileCheck -check-prefix=CHECK-V8 %s

define i32 @t1(i32 %a, i32 %b, i8** %retaddr) {
; CHECK-LABEL: t1:
; CHECK: Block address taken
; CHECK-NOT: Address of block that was removed by CodeGen
  store i8* blockaddress(@t1, %cond_true), i8** %retaddr
  %tmp2 = icmp eq i32 %a, 0
  br i1 %tmp2, label %cond_false, label %cond_true

cond_true:
  %tmp5 = add i32 %b, 1
  ret i32 %tmp5

cond_false:
  %tmp7 = add i32 %b, -1
  ret i32 %tmp7
}

define i32 @t2(i32 %a, i32 %b, i32 %c, i32 %d, i8** %retaddr) {
; CHECK-LABEL: t2:
; CHECK: Block address taken
; CHECK: %cond_true
; CHECK: add
; CHECK: bx lr
  store i8* blockaddress(@t2, %cond_true), i8** %retaddr
  %tmp2 = icmp sgt i32 %c, 10
  %tmp5 = icmp slt i32 %d, 4
  %tmp8 = and i1 %tmp5, %tmp2
  %tmp13 = add i32 %b, %a
  br i1 %tmp8, label %cond_true, label %UnifiedReturnBlock

cond_true:
  %tmp15 = add i32 %tmp13, %c
  %tmp1821 = sub i32 %tmp15, %d
  ret i32 %tmp1821

UnifiedReturnBlock:
  ret i32 %tmp13
}

define hidden fastcc void @t3(i8** %retaddr, i1 %tst, i8* %p8) {
; CHECK-LABEL: t3:
; CHECK: Block address taken
; CHECK-NOT: Address of block that was removed by CodeGen
bb:
  store i8* blockaddress(@t3, %KBBlockZero_return_1), i8** %retaddr
  br i1 %tst, label %bb77, label %bb7.i

bb7.i:                                            ; preds = %bb35
  br label %bb2.i

KBBlockZero_return_1:                             ; preds = %KBBlockZero.exit
  ret void

KBBlockZero_return_0:                             ; preds = %KBBlockZero.exit
  ret void

bb77:                                             ; preds = %bb26, %bb12, %bb
  ret void

bb2.i:                                            ; preds = %bb6.i350, %bb7.i
  br i1 %tst, label %bb6.i350, label %KBBlockZero.exit

bb6.i350:                                         ; preds = %bb2.i
  br label %bb2.i

KBBlockZero.exit:                                 ; preds = %bb2.i
  indirectbr i8* %p8, [label %KBBlockZero_return_1, label %KBBlockZero_return_0]
}

@foo = global i32 ()* null
define i32 @t4(i32 %x, i32 ()* %p_foo) {
entry:
;CHECK-LABEL: t4:
;CHECK-V8-LABEL: t4:
  %cmp = icmp slt i32 %x, 60
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %tmp.2 = call i32 %p_foo()
  %sub = add nsw i32 %x, -1
  br label %return

if.else:                                          ; preds = %entry
  %sub1 = add nsw i32 %x, -120
  br label %return

return:                                           ; preds = %if.end5, %if.then4, %if.then
  %retval.0 = phi i32 [ %sub, %if.then ], [ %sub1, %if.else ]
  ret i32 %retval.0
}

; If-converter was checking for the wrong predicate subsumes pattern when doing
; nested predicates.
; E.g., Let A be a basic block that flows conditionally into B and B be a
; predicated block.
; B can be predicated with A.BrToBPredicate into A iff B.Predicate is less
; "permissive" than A.BrToBPredicate, i.e., iff A.BrToBPredicate subsumes
; B.Predicate. 

; Hard-coded registers comes from the ABI.
; CHECK-LABEL: wrapDistance:
; CHECK: cmp r1, #59
; CHECK-NEXT: itt le
; CHECK-NEXT: suble r0, r2, #1
; CHECK-NEXT: bxle lr
; CHECK-NEXT: LBB{{.*}}:
; CHECK-NEXT: subs [[REG:r[0-9]+]], #120
; CHECK-NEXT: cmp [[REG]], r1
; CHECK-NOT: it lt
; CHECK-NEXT: bge [[LABEL:.+]]
; Next BB
; CHECK-NOT: cmplt
; CHECK: cmp r0, #119
; CHECK-NEXT: itt le
; CHECK-NEXT: addle r0, r1, #1
; CHECK-NEXT: bxle lr
; Next BB
; CHECK: [[LABEL]]:
; CHECK-NEXT: subs r0, r1, r0
; CHECK-NEXT: bx lr

; CHECK-V8-LABEL: wrapDistance:
; CHECK-V8: cmp r1, #59
; CHECK-V8-NEXT: bgt
; CHECK-V8-NEXT: %if.then
; CHECK-V8-NEXT: subs r0, r2, #1
; CHECK-V8-NEXT: bx lr
; CHECK-V8-NEXT: %if.else
; CHECK-V8-NEXT: subs [[REG:r[0-9]+]], #120
; CHECK-V8-NEXT: cmp [[REG]], r1
; CHECK-V8-NEXT: bge
; CHECK-V8-NEXT: %if.else
; CHECK-V8-NEXT: cmp r0, #119
; CHECK-V8-NEXT: bgt
; CHECK-V8-NEXT: %if.then4
; CHECK-V8-NEXT: adds r0, r1, #1
; CHECK-V8-NEXT: bx lr
; CHECK-V8-NEXT: %if.end5
; CHECK-V8-NEXT: subs r0, r1, r0
; CHECK-V8-NEXT: bx lr

define i32 @wrapDistance(i32 %tx, i32 %sx, i32 %w) {
entry:
  %cmp = icmp slt i32 %sx, 60
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %sub = add nsw i32 %w, -1
  br label %return

if.else:                                          ; preds = %entry
  %sub1 = add nsw i32 %w, -120
  %cmp2 = icmp slt i32 %sub1, %sx
  %cmp3 = icmp slt i32 %tx, 120
  %or.cond = and i1 %cmp2, %cmp3
  br i1 %or.cond, label %if.then4, label %if.end5

if.then4:                                         ; preds = %if.else
  %add = add nsw i32 %sx, 1
  br label %return

if.end5:                                          ; preds = %if.else
  %sub6 = sub nsw i32 %sx, %tx
  br label %return

return:                                           ; preds = %if.end5, %if.then4, %if.then
  %retval.0 = phi i32 [ %sub, %if.then ], [ %add, %if.then4 ], [ %sub6, %if.end5 ]
  ret i32 %retval.0
}
