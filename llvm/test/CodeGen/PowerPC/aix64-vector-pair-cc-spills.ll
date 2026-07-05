; RUN: llc -O0 -mtriple=powerpc64-ibm-aix-xcoff -mcpu=pwr10 -stop-after=prologepilog -verify-machineinstrs < %s | \
; RUN: FileCheck --check-prefix=CHECK %s
; RUN: llc -O0 -mtriple=powerpc64-ibm-aix-xcoff -mcpu=pwr10 -vec-extabi -stop-after=prologepilog -verify-machineinstrs < %s | \
; RUN: FileCheck --check-prefix=CHECK-VEXT %s

; CHECK-LABEL: name: foo
; CHECK-NOT: spill-slot
; CHECK-NOT: callee-saved-register: '$v31'
; CHECK-NOT: callee-saved-register: '$v30'
; CHECK-NOT: callee-saved-register: '$v29'
; CHECK-NOT: callee-saved-register: '$v28'
; CHECK-NOT: callee-saved-register: '$v27'
; CHECK-NOT: callee-saved-register: '$v26'
; CHECK-NOT: callee-saved-register: '$v25'
; CHECK-NOT: callee-saved-register: '$v24'
; CHECK-NOT: callee-saved-register: '$v23'
; CHECK-NOT: callee-saved-register: '$v22'
; CHECK-NOT: callee-saved-register: '$v21'
; CHECK-NOT: callee-saved-register: '$v20'

; CHECK-VEXT-LABEL: name: foo
; CHECK-VEXT-NOT: spill-slot
; CHECK-VEXT-NOT: callee-saved-register: '$v31'
; CHECK-VEXT-NOT: callee-saved-register: '$v30'
; CHECK-VEXT-NOT: callee-saved-register: '$v29'
; CHECK-VEXT-NOT: callee-saved-register: '$v28'
; CHECK-VEXT-NOT: callee-saved-register: '$v27'
; CHECK-VEXT-NOT: callee-saved-register: '$v26'
; CHECK-VEXT-NOT: callee-saved-register: '$v25'
; CHECK-VEXT-NOT: callee-saved-register: '$v24'
; CHECK-VEXT-NOT: callee-saved-register: '$v23'
; CHECK-VEXT-NOT: callee-saved-register: '$v22'
; CHECK-VEXT-NOT: callee-saved-register: '$v21'
; CHECK-VEXT-NOT: callee-saved-register: '$v20'
define void @foo() {
entry:
  call void @bar(i32 0)
  ret void
}

; CHECK-LABEL: name: spill
; CHECK-NOT: spill-slot
; CHECK-NOT: callee-saved-register: '$v31'
; CHECK-NOT: callee-saved-register: '$v30'
; CHECK-NOT: callee-saved-register: '$v29'
; CHECK-NOT: callee-saved-register: '$v28'
; CHECK-NOT: callee-saved-register: '$v27'
; CHECK-NOT: callee-saved-register: '$v26'
; CHECK-NOT: callee-saved-register: '$v25'
; CHECK-NOT: callee-saved-register: '$v24'
; CHECK-NOT: callee-saved-register: '$v23'
; CHECK-NOT: callee-saved-register: '$v22'
; CHECK-NOT: callee-saved-register: '$v21'
; CHECK-NOT: callee-saved-register: '$v20'

; CHECK-VEXT-LABEL: name: spill
; CHECK-VEXT: spill-slot
; CHECK-VEXT-NEXT: callee-saved-register: '$v31'
; CHECK-VEXT: spill-slot
; CHECK-VEXT-NEXT: callee-saved-register: '$v30'
; CHECK-VEXT: spill-slot
; CHECK-VEXT-NEXT: callee-saved-register: '$v29'
; CHECK-VEXT: spill-slot
; CHECK-VEXT-NEXT: callee-saved-register: '$v28'
; CHECK-VEXT: spill-slot
; CHECK-VEXT-NEXT: callee-saved-register: '$v27'
; CHECK-VEXT: spill-slot
; CHECK-VEXT-NEXT: callee-saved-register: '$v26'
; CHECK-VEXT: spill-slot
; CHECK-VEXT-NEXT: callee-saved-register: '$v25'
; CHECK-VEXT: spill-slot
; CHECK-VEXT-NEXT: callee-saved-register: '$v24'
; CHECK-VEXT: spill-slot
; CHECK-VEXT-NEXT: callee-saved-register: '$v23'
; CHECK-VEXT: spill-slot
; CHECK-VEXT-NEXT: callee-saved-register: '$v22'
; CHECK-VEXT: spill-slot
; CHECK-VEXT-NEXT: callee-saved-register: '$v21'
; CHECK-VEXT: spill-slot
; CHECK-VEXT-NEXT: callee-saved-register: '$v20'
define void @spill() {
entry:
  call void asm sideeffect "nop", "~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
  call void @bar(i32 0)
  ret void
}

declare void @bar(i32)
