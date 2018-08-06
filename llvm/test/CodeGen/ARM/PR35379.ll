; RUN: llc -mtriple=armv7a-eabi < %s | FileCheck %s --check-prefix=CHECK-ARM
; RUN: llc -mtriple=armv6m-eabi < %s | FileCheck %s --check-prefix=CHECK-THM

; Function Attrs: minsize optsize
declare void @g(i32*) local_unnamed_addr #0

; Function Attrs: minsize optsize
define void @f() local_unnamed_addr #0 {
entry:
  %i = alloca i32, align 4
  %0 = bitcast i32* %i to i8*
  store i32 1, i32* %i, align 4
  call void @g(i32* nonnull %i)
  ret void
}

; Check unwind info does not mention the registers used for padding, and
; the amount of stack adjustment is the same as in the actual
; instructions.

; CHECK-ARM:      .save {r11, lr}
; CHECK-ARM-NEXT: .pad #8
; CHECK-ARM-NEXT: push {r9, r10, r11, lr}
; CHECK-ARM:      pop {r2, r3, r11, pc}

; CHECK-THM:      .save {r7, lr}
; CHECK-THM-NEXT: .pad #8
; CHECK-THM-NEXT: push {r5, r6, r7, lr}
; CHECK-THM:      pop {r2, r3, r7, pc}


define void @f1() local_unnamed_addr #1 {
entry:
  %i = alloca i32, align 4
  %0 = bitcast i32* %i to i8*
  store i32 1, i32* %i, align 4
  call void @g(i32* nonnull %i)
  ret void
}

; Check that unwind info is the same whether or not using -Os (minsize attr)

; CHECK-ARM:      .save {r11, lr}
; CHECK-ARM-NEXT: push {r11, lr}
; CHECK-ARM-NEXT: .pad #8

; CHECK-THM:      .save {r7, lr}
; CHECK-THM-NEXT: push {r7, lr}
; CHECK-THM-NEXT: .pad #8

attributes #0 = { minsize optsize }
attributes #1 = { optsize }
