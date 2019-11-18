; Using VLAs(Variable Length Arrays) in a function will use R6 to keep track
; of the stack frame, and also spill/restore R6 to the stack.
; This tests that using -ffixed-r6 (-mattr=+reserve-r6) will stop R6
; being used and also stop it being spilled/restored to the stack.
; RUN: llc < %s -mcpu=cortex-m0 -mtriple=thumbv7-arm-none-eabi  | FileCheck %s --check-prefix=CHECK-STATIC --check-prefix=CHECK-R6
; RUN: llc < %s -mcpu=cortex-m0 -mtriple=thumbv7-arm-none-eabi -mattr=+reserve-r6  | FileCheck %s --check-prefix=CHECK-STATIC --check-prefix=CHECK-NO-R6

define void @f() #0 {
entry:
  %i = alloca i32, align 4
  store i32 0, i32* %i, align 4

  %saved_stack = alloca i8*, align 4
  %0 = call i8* @llvm.stacksave()
  store i8* %0, i8** %saved_stack, align 4

  %__vla_expr0 = alloca i32, align 4
  %1 = load i32, i32* %i, align 4
  %vla = alloca double, i32 %1, align 8
  store i32 %1, i32* %__vla_expr0, align 4

  %2 = load i8*, i8** %saved_stack, align 4
  call void @llvm.stackrestore(i8* %2)

  ret void
}

declare i8* @llvm.stacksave() #1
declare void @llvm.stackrestore(i8* %ptr) #1

attributes #0 = { noinline nounwind "stackrealign" }
attributes #1 = { nounwind }

; CHECK-STATIC: push {r4,
; CHECK-R6: r6
; CHECK-NO-R6-NOT: r6
; CHECK-STATIC: lr}
; CHECK-R6: r6
; CHECK-NO-R6-NOT: r6
; CHECK-STATIC: pop {r4,
; CHECK-R6: r6
; CHECK-NO-R6-NOT: r6
; CHECK-STATIC: pc}

