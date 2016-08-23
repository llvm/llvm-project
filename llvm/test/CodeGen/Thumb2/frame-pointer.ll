; RUN: llc -mtriple=thumbv7m-none-eabi -o - %s | FileCheck %s

declare void @foo()

; Leaf function, no frame so no need for a frame pointer.
define void @leaf() {
; CHECK-LABEL: leaf:
; CHECK-NOT: push
; CHECK-NOT: sp
; CHECK-NOT: pop
; CHECK: bx lr
  ret void
}

; Leaf function, frame pointer is requested but we don't need any stack frame,
; so don't create a frame pointer.
define void @leaf_nofpelim() "no-frame-pointer-elim"="true" {
; CHECK-LABEL: leaf_nofpelim:
; CHECK-NOT: push
; CHECK-NOT: sp
; CHECK-NOT: pop
; CHECK: bx lr
  ret void
}

; Leaf function, frame pointer is requested and we need a stack frame, so we
; need to use a frame pointer.
define void @leaf_lowreg_nofpelim() "no-frame-pointer-elim"="true" {
; CHECK-LABEL: leaf_lowreg_nofpelim:
; CHECK: push {r4, r7, lr}
; CHECK: add r7, sp, #4
; CHECK: pop {r4, r7, pc}
  call void asm sideeffect "", "~{r4}" ()
  ret void
}

; Leaf function, frame pointer is requested and we need a stack frame, so we
; need to use a frame pointer. A high register is pushed to the stack, so we
; must use two push/pop instructions to ensure that fp and sp are adjacent on
; the stack.
define void @leaf_highreg_nofpelim() "no-frame-pointer-elim"="true" {
; CHECK-LABEL: leaf_highreg_nofpelim:
; CHECK: push {r7, lr}
; CHECK: mov r7, sp
; CHECK: str r8, [sp, #-4]!
; CHECK: ldr r8, [sp], #4
; CHECK: pop {r7, pc}
  call void asm sideeffect "", "~{r8}" ()
  ret void
}

; Leaf function, frame pointer requested for non-leaf functions only, so no
; need for a stack frame.
define void @leaf_nononleaffpelim() "no-frame-pointer-elim-non-leaf" {
; CHECK-LABEL: leaf_nononleaffpelim:
; CHECK-NOT: push
; CHECK-NOT: sp
; CHECK-NOT: pop
; CHECK: bx lr
  ret void
}

; Has a call, but still no need for a frame pointer.
define void @call() {
; CHECK-LABEL: call:
; CHECK: push {[[DUMMYREG:r[0-9]+]], lr}
; CHECK-NOT: sp
; CHECK: bl foo
; CHECK: pop {[[DUMMYREG]], pc}
  call void @foo()
  ret void
}

; Has a call, and frame pointer requested.
define void @call_nofpelim() "no-frame-pointer-elim"="true" {
; CHECK-LABEL: call_nofpelim:
; CHECK: push {r7, lr}
; CHECK: mov r7, sp
; CHECK: bl foo
; CHECK: pop {r7, pc}
  call void @foo()
  ret void
}

; Has a call, and frame pointer requested for non-leaf function.
define void @call_nononleaffpelim() "no-frame-pointer-elim-non-leaf" {
; CHECK-LABEL: call_nononleaffpelim:
; CHECK: push {r7, lr}
; CHECK: mov r7, sp
; CHECK: bl foo
; CHECK: pop {r7, pc}
  call void @foo()
  ret void
}

; Has a high register clobbered, no need for a frame pointer.
define void @highreg() {
; CHECK-LABEL: highreg:
; CHECK: push.w {r8, lr}
; CHECK-NOT: sp
; CHECK: bl foo
; CHECK: pop.w {r8, pc}
  call void asm sideeffect "", "~{r8}" ()
  call void @foo()
  ret void
}

; Has a high register clobbered, frame pointer requested. We need to split the
; push into two, to ensure that r7 and sp are adjacent on the stack.
define void @highreg_nofpelim() "no-frame-pointer-elim"="true" {
; CHECK-LABEL: highreg_nofpelim:
; CHECK: push {[[DUMMYREG:r[0-9]+]], r7, lr}
; CHECK: add r7, sp, #4
; CHECK: str r8, [sp, #-4]!
; CHECK: bl foo
; CHECK: ldr r8, [sp], #4
; CHECK: pop {[[DUMMYREG]], r7, pc}
  call void asm sideeffect "", "~{r8}" ()
  call void @foo()
  ret void
}

; Has a high register clobbered, frame required due to variable-sized alloca.
; We need a frame pointer to correctly restore the stack, but don't need to
; split the push/pop here, because the frame pointer not required by the ABI.
define void @highreg_alloca(i32 %a) {
; CHECK-LABEL: highreg_alloca:
; CHECK: push.w {[[SOMEREGS:.*]], r7, r8, lr}
; CHECK: add r7, sp, #{{[0-9]+}}
; CHECK: bl foo
; CHECK: pop.w {[[SOMEREGS]], r7, r8, pc}
  %alloca = alloca i32, i32 %a, align 4
  call void @foo()
  call void asm sideeffect "", "~{r8}" ()
  ret void
}

; Has a high register clobbered, frame required due to both variable-sized
; alloca and ABI. We do need to split the push/pop here.
define void @highreg_alloca_nofpelim(i32 %a) "no-frame-pointer-elim"="true" {
; CHECK-LABEL: highreg_alloca_nofpelim:
; CHECK: push {[[SOMEREGS:.*]], r7, lr}
; CHECK: add r7, sp, #{{[0-9]+}}
; CHECK: str r8, [sp, #-4]!
; CHECK: bl foo
; CHECK: ldr r8, [sp], #4
; CHECK: pop {[[SOMEREGS]], r7, pc}
  %alloca = alloca i32, i32 %a, align 4
  call void @foo()
  call void asm sideeffect "", "~{r8}" ()
  ret void
}
