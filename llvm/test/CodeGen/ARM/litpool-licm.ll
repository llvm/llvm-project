; RUN: llc -mtriple=thumbv7-linux-gnueabihf -relocation-model=pic %s -o - | FileCheck %s

@var = thread_local global i32 0, align 4

define void @func(i32 %n) {
; CHECK-LABEL: func:
; CHECK: ldr [[REF1:r[0-9]+]], [[CP1:.LCPI[0-9]+_[0-9]+]]
; CHECK: ldr [[REF2:r[0-9]+]], [[CP2:.LCPI[0-9]+_[0-9]+]]

; CHECK: [[PCPOS1:.LPC[0-9]+_[0-9]+]]:
; CHECK-NEXT: add [[REF1]], pc

; CHECK: [[PCPOS2:.LPC[0-9]+_[0-9]+]]:
; CHECK-NEXT: add [[REF2]], pc

; CHECK: [[CP1]]:
; CHECK-NEXT: [[CP1_TMP:.Ltmp[0-9]+]]:
; CHECK-NEXT:     .long var(TLSGD)-(([[PCPOS1]]+4)-[[CP1_TMP]])

; CHECK: [[CP2]]:
; CHECK-NEXT: [[CP2_TMP:.Ltmp[0-9]+]]:
; CHECK-NEXT:     .long var(TLSGD)-(([[PCPOS2]]+4)-[[CP2_TMP]])

entry:
  br label %loop

loop:
  %i = phi i32 [ %inc, %next ], [ 0, %entry ]
  %val = load i32, i32* @var
  %tst = icmp eq i32 %val, 0
  br i1 %tst, label %next, label %call

call:
  tail call void @foo(i32* nonnull @var) #2
  br label %next

next:
  %inc = add i32 %i, 1
  %stop = icmp eq i32 %inc, %n
  br i1 %stop, label %done, label %loop

done:
  ret void
}

declare void @foo(i32*)
