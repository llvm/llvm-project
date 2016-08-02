; RUN: llc -mtriple=aarch64-apple-darwin -aarch64-bcc-offset-bits=3 < %s | FileCheck %s

; CHECK-LABEL: invert_bcc:
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  b.ne [[BB1:LBB[0-9]+_[0-9]+]]
; CHECK-NEXT:  b.vs [[BB2:LBB[0-9]+_[0-9]+]]
; CHECK-NEXT:  b [[BB2]]

; CHECK: [[BB1]]:
; CHECK: mov w{{[0-9]+}}, #9
; CHECK: ret

; CHECK: [[BB2]]:
; CHECK: mov w{{[0-9]+}}, #42
; CHECK: ret

define i32 @invert_bcc(float %x, float %y) #0 {
  %1 = fcmp ueq float %x, %y
  br i1 %1, label %bb1, label %bb2

bb2:
  call void asm sideeffect
    "nop
     nop",
    ""() #0
  store volatile i32 9, i32* undef
  ret i32 1

bb1:
  store volatile i32 42, i32* undef
  ret i32 0
}

declare i32 @foo() #0

; CHECK-LABEL: _block_split:
; CHECK: cmp w0, #5
; CHECK-NEXT: b.eq [[LONG_BR_BB:LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: b [[LOR_LHS_FALSE_BB:LBB[0-9]+_[0-9]+]]

; CHECK: [[LONG_BR_BB]]:
; CHECK-NEXT: b [[IF_THEN_BB:LBB[0-9]+_[0-9]+]]

; CHECK: [[LOR_LHS_FALSE_BB]]:
; CHECK: cmp w{{[0-9]+}}, #16
; CHECK-NEXT: b.le [[IF_THEN_BB]]
; CHECK-NEXT: b [[IF_END_BB:LBB[0-9]+_[0-9]+]]

; CHECK: [[IF_THEN_BB]]:
; CHECK: bl _foo
; CHECK-NOT: b L

; CHECK: [[IF_END_BB]]:
; CHECK: #0x7
; CHECK: ret
define i32 @block_split(i32 %a, i32 %b) #0 {
entry:
  %cmp = icmp eq i32 %a, 5
  br i1 %cmp, label %if.then, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  %cmp1 = icmp slt i32 %b, 7
  %mul = shl nsw i32 %b, 1
  %add = add nsw i32 %b, 1
  %cond = select i1 %cmp1, i32 %mul, i32 %add
  %cmp2 = icmp slt i32 %cond, 17
  br i1 %cmp2, label %if.then, label %if.end

if.then:                                          ; preds = %lor.lhs.false, %entry
  %call = tail call i32 @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %lor.lhs.false
  ret i32 7
}

attributes #0 = { nounwind }
