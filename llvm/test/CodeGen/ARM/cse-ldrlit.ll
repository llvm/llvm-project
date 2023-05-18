; RUN: llc -mtriple=thumbv6m-apple-none-macho -relocation-model=pic -o -  %s | FileCheck %s --check-prefix=CHECK-THUMB-PIC
; RUN: llc -mtriple=arm-apple-none-macho -relocation-model=pic -o -  %s | FileCheck %s --check-prefix=CHECK-ARM-PIC
; RUN: llc -mtriple=thumbv6m-apple-none-macho -relocation-model=dynamic-no-pic -o -  %s | FileCheck %s --check-prefix=CHECK-DYNAMIC
; RUN: llc -mtriple=arm-apple-none-macho -relocation-model=dynamic-no-pic -o -  %s | FileCheck %s --check-prefix=CHECK-DYNAMIC
; RUN: llc -mtriple=thumbv6m-apple-none-macho -relocation-model=static -o -  %s | FileCheck %s --check-prefix=CHECK-STATIC
; RUN: llc -mtriple=arm-apple-none-macho -relocation-model=static -o -  %s | FileCheck %s --check-prefix=CHECK-STATIC
@var = global [16 x i32] zeroinitializer

declare void @bar(ptr)

define void @foo() {
  %flag = load i32, ptr getelementptr inbounds([16 x i32], ptr @var, i32 0, i32 1)
  %tst = icmp eq i32 %flag, 0
  br i1 %tst, label %true, label %false
true:
  tail call void @bar(ptr getelementptr inbounds([16 x i32], ptr @var, i32 0, i32 4))
  ret void
false:
  ret void
}

; CHECK-THUMB-PIC-LABEL: foo:
; CHECK-THUMB-PIC: ldr r0, LCPI0_0
; CHECK-THUMB-PIC: LPC0_0:
; CHECK-THUMB-PIC-NEXT: add r0, pc
; CHECK-THUMB-PIC: ldr {{r[1-9][0-9]?}}, [r0, #4]

; CHECK-THUMB-PIC: LCPI0_0:
; CHECK-THUMB-PIC-NEXT: .long _var-(LPC0_0+4)
; CHECK-THUMB-PIC-NOT: LCPI0_1


; CHECK-ARM-PIC-LABEL: foo:
; CHECK-ARM-PIC: ldr [[VAR_OFFSET:r[0-9]+]], LCPI0_0
; CHECK-ARM-PIC: LPC0_0:
; CHECK-ARM-PIC-NEXT: add r0, pc, [[VAR_OFFSET]]
; CHECK-ARM-PIC: ldr {{r[0-9]+}}, [r0, #4]

; CHECK-ARM-PIC: LCPI0_0:
; CHECK-ARM-PIC-NEXT: .long _var-(LPC0_0+8)
; CHECK-ARM-PIC-NOT: LCPI0_1


; CHECK-DYNAMIC-LABEL: foo:
; CHECK-DYNAMIC: ldr r0, LCPI0_0
; CHECK-DYNAMIC: ldr {{r[1-9][0-9]?}}, [r0, #4]

; CHECK-DYNAMIC: LCPI0_0:
; CHECK-DYNAMIC-NEXT: .long _var
; CHECK-DYNAMIC-NOT: LCPI0_1


; CHECK-STATIC-LABEL: foo:
; CHECK-STATIC: ldr r0, LCPI0_0
; CHECK-STATIC: ldr {{r[1-9][0-9]?}}, [r0, #4]

; CHECK-STATIC: LCPI0_0:
; CHECK-STATIC-NEXT: .long _var{{$}}
; CHECK-STATIC-NOT: LCPI0_1


