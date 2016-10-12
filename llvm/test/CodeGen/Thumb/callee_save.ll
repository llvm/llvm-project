; RUN: llc -mtriple=thumbv6m-none-eabi < %s | FileCheck %s

declare i8* @llvm.returnaddress(i32)

; We don't allocate high registers, so any function not using inline asm will
; only need to save the low registers.
define void @low_regs_only() {
; CHECK-LABEL: low_regs_only:
entry:
; CHECK: push {r4, r5, r6, r7, lr}
  tail call void asm sideeffect "", "~{r4},~{r5},~{r6},~{r7}"()
; CHECK: pop {r4, r5, r6, r7, pc}
  ret void
}

; One high reg clobbered, but no low regs, args or returns. We can use an
; argument/return register to help save/restore it.
define void @one_high() {
; CHECK-LABEL: one_high:
entry:
; CHECK: mov [[SAVEREG:r[0-3]]], r8
; CHECK: push {[[SAVEREG]]}
  tail call void asm sideeffect "", "~{r8}"()
; CHECK: pop {[[RESTOREREG:r[0-3]]]}
; CHECK: mov r8, [[RESTOREREG]]
  ret void
}

; 4 high regs clobbered, but still no low regs, args or returns. We can use all
; 4 arg/return regs for the save/restore.
define void @four_high() {
; CHECK-LABEL: four_high:
entry:
; CHECK: mov r3, r11
; CHECK: mov r2, r10
; CHECK: mov r1, r9
; CHECK: mov r0, r8
; CHECK: push {r0, r1, r2, r3}
  tail call void asm sideeffect "", "~{r8},~{r9},~{r10},~{r11}"()
; CHECK: pop {r0, r1, r2, r3}
; CHECK: mov r8, r0
; CHECK: mov r9, r1
; CHECK: mov r10, r2
; CHECK: mov r11, r3
  ret void
}

; One high and one low register clobbered. lr also gets pushed to simplify the
; return, and r7 to keep the stack aligned. Here, we could use r0-r3, r4, r7 or
; lr to save/restore r8.
define void @one_high_one_low() {
; CHECK-LABEL: one_high_one_low:
entry:
; CHECK: push {r4, r7, lr}
; CHECK: mov [[SAVEREG:r0|r1|r2|r3|r4|r7|lr]], r8
; CHECK: push {[[SAVEREG]]}
  tail call void asm sideeffect "", "~{r4},~{r8}"()
; CHECK: pop {[[RESTOREREG:r0|r1|r2|r3|r4|r7]]}
; CHECK: mov r8, [[RESTOREREG]]
; CHECK: pop {r4, r7, pc}
  ret void
}

; All callee-saved registers clobbered, r4-r7 and lr are not live after the
; first push so can be used for pushing the high registers.
define void @four_high_four_low() {
; CHECK-LABEL: four_high_four_low:
entry:
; CHECK: push {r4, r5, r6, r7, lr}
; CHECK: mov lr, r11
; CHECK: mov r7, r10
; CHECK: mov r6, r9
; CHECK: mov r5, r8
; CHECK: push {r5, r6, r7, lr}
  tail call void asm sideeffect "", "~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11}"()
; CHECK: pop {r0, r1, r2, r3}
; CHECK: mov r8, r0
; CHECK: mov r9, r1
; CHECK: mov r10, r2
; CHECK: mov r11, r3
; CHECK: pop {r4, r5, r6, r7, pc}
  ret void
}


; All callee-saved registers clobbered, and frame pointer is requested. r7 now
; cannot be used while saving/restoring the high regs.
define void @four_high_four_low_frame_ptr() "no-frame-pointer-elim"="true" {
; CHECK-LABEL: four_high_four_low_frame_ptr:
entry:
; CHECK: push {r4, r5, r6, r7, lr}
; CHECK: add r7, sp, #12
; CHECK: mov lr, r11
; CHECK: mov r6, r10
; CHECK: mov r5, r9
; CHECK: mov r4, r8
; CHECK: push {r4, r5, r6, lr}
  tail call void asm sideeffect "", "~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11}"()
; CHECK: pop {r0, r1, r2, r3}
; CHECK: mov r8, r0
; CHECK: mov r9, r1
; CHECK: mov r10, r2
; CHECK: mov r11, r3
; CHECK: pop {r4, r5, r6, r7, pc}
  ret void
}

; All callee-saved registers clobbered, frame pointer is requested and
; llvm.returnaddress used. r7 and lr now cannot be used while saving/restoring
; the high regs.
define void @four_high_four_low_frame_ptr_ret_addr() "no-frame-pointer-elim"="true" {
; CHECK-LABEL: four_high_four_low_frame_ptr_ret_addr:
entry:
; CHECK: push {r4, r5, r6, r7, lr}
; CHECK: mov r6, r11
; CHECK: mov r5, r10
; CHECK: mov r4, r9
; CHECK: mov r3, r8
; CHECK: push {r3, r4, r5, r6}
  %a = tail call i8* @llvm.returnaddress(i32 0)
  tail call void asm sideeffect "", "r,~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11}"(i8* %a)
; CHECK: pop {r0, r1, r2, r3}
; CHECK: mov r8, r0
; CHECK: mov r9, r1
; CHECK: mov r10, r2
; CHECK: mov r11, r3
; CHECK: pop {r4, r5, r6, r7, pc}
  ret void
}

; 4 high regs clobbered, all 4 argument registers used. We push an extra 4 low
; registers, so that we can use them for saving the high regs.
define void @four_high_four_arg(i32 %a, i32 %b, i32 %c, i32 %d) {
; CHECK-LABEL: four_high_four_arg:
entry:
; CHECK: push    {r5, r6, r7, lr}
; CHECK: mov     lr, r11
; CHECK: mov     r7, r10
; CHECK: mov     r6, r9
; CHECK: mov     r5, r8
; CHECK: push    {r5, r6, r7, lr}
  tail call void asm sideeffect "", "r,r,r,r,~{r8},~{r9},~{r10},~{r11}"(i32 %a, i32 %b, i32 %c, i32 %d)
; CHECK: pop     {r0, r1, r2, r3}
; CHECK: mov     r8, r0
; CHECK: mov     r9, r1
; CHECK: mov     r10, r2
; CHECK: mov     r11, r3
; CHECK: pop     {r5, r6, r7, pc}
  ret void
}

; 4 high regs clobbered, all 4 return registers used. We push an extra 4 low
; registers, so that we can use them for restoring the high regs.
define <4 x i32> @four_high_four_return() {
; CHECK-LABEL: four_high_four_return:
entry:
; CHECK: push    {r4, r5, r6, r7, lr}
; CHECK: mov     lr, r11
; CHECK: mov     r7, r10
; CHECK: mov     r6, r9
; CHECK: mov     r5, r8
; CHECK: push    {r5, r6, r7, lr}
  tail call void asm sideeffect "", "~{r8},~{r9},~{r10},~{r11}"()
  %vecinit = insertelement <4 x i32> undef, i32 1, i32 0
  %vecinit11 = insertelement <4 x i32> %vecinit, i32 2, i32 1
  %vecinit12 = insertelement <4 x i32> %vecinit11, i32 3, i32 2
  %vecinit13 = insertelement <4 x i32> %vecinit12, i32 4, i32 3
; CHECK: pop     {r4, r5, r6, r7}
; CHECK: mov     r8, r4
; CHECK: mov     r9, r5
; CHECK: mov     r10, r6
; CHECK: mov     r11, r7
; CHECK: pop     {r4, r5, r6, r7, pc}
  ret <4 x i32> %vecinit13
}

; 4 high regs clobbered, all args & returns used, frame pointer requested and
; llvm.returnaddress called. This leaves us with 3 low registers available (r4,
; r5, r6), with which to save 4 high registers, so we have to use two pushes
; and pops.
define <4 x i32> @all_of_the_above(i32 %a, i32 %b, i32 %c, i32 %d) "no-frame-pointer-elim"="true" {
; CHECK-LABEL: all_of_the_above
entry:
; CHECK: push    {r4, r5, r6, r7, lr}
; CHECK: add     r7, sp, #12
; CHECK: mov     r6, r11
; CHECK: mov     r5, r10
; CHECK: mov     r4, r9
; CHECK: push    {r4, r5, r6}
; CHECK: mov     r6, r8
; CHECK: push    {r6}
  tail call void asm sideeffect "", "r,r,r,r,~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11}"(i32 %a, i32 %b, i32 %c, i32 %d)
  %e = tail call i8* @llvm.returnaddress(i32 0)
  %f = ptrtoint i8* %e to i32
  %vecinit = insertelement <4 x i32> undef, i32 %f, i32 0
  %vecinit11 = insertelement <4 x i32> %vecinit, i32 2, i32 1
  %vecinit12 = insertelement <4 x i32> %vecinit11, i32 3, i32 2
  %vecinit13 = insertelement <4 x i32> %vecinit12, i32 4, i32 3
; CHECK: pop     {r4, r5, r6}
; CHECK: mov     r8, r4
; CHECK: mov     r9, r5
; CHECK: mov     r10, r6
; CHECK: pop     {r4}
; CHECK: mov     r11, r4
; CHECK: pop     {r4, r5, r6, r7, pc}
  ret <4 x i32> %vecinit13
}

; When a base pointer is being used, we can safely use it for saving/restoring
; the high regs because it is set after the last push, and not used at all in the
; epliogue. We can also use r4 for restoring the registers despite it also being
; used when restoring sp from fp, as that happens before the first pop.
define <4 x i32> @base_pointer(i32 %a) {
; CHECK-LABEL: base_pointer:
entry:
; CHECK: push    {r4, r6, r7, lr}
; CHECK: add     r7, sp, #8
; CHECK: mov     lr, r9
; CHECK: mov     r6, r8
; CHECK: push    {r6, lr}
; CHECK: mov     r6, sp
  %b = alloca i32, i32 %a
  call void asm sideeffect "", "r,~{r8},~{r9}"(i32* %b)
  %vecinit = insertelement <4 x i32> undef, i32 1, i32 0
  %vecinit11 = insertelement <4 x i32> %vecinit, i32 2, i32 1
  %vecinit12 = insertelement <4 x i32> %vecinit11, i32 3, i32 2
  %vecinit13 = insertelement <4 x i32> %vecinit12, i32 4, i32 3
; CHECK: subs    r4, r7, #7
; CHECK: subs    r4, #9
; CHECK: mov     sp, r4
; CHECK: pop     {r4, r6}
; CHECK: mov     r8, r4
; CHECK: mov     r9, r6
; CHECK: pop     {r4, r6, r7, pc}
  ret <4 x i32> %vecinit13
}
