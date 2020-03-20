; RUN: llc -mtriple armv7a--none-eabi   -enable-ipra=true -arm-extra-spills -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK --check-prefix=ARM
; RUN: llc -mtriple thumbv7a--none-eabi -enable-ipra=true -arm-extra-spills -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB2
; RUN: llc -mtriple thumbv6m--none-eabi -enable-ipra=true -arm-extra-spills -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB1

; This clobbers r0, and already needs a push/pop, so we also save and restore
; r0. The push of r11 is to maintain stack alignment (though that isn't
; technically needed in this example).
define void @test_r0_r4() minsize nounwind {
; CHECK-LABEL: test_r0_r4:
; ARM: .save   {r0, r4, r11, lr}
; ARM: push    {r0, r4, r11, lr}
; ARM: pop     {r0, r4, r11, pc}
; THUMB1: .save   {r0, r4, r7, lr}
; THUMB1: push    {r0, r4, r7, lr}
; THUMB1: pop     {r0, r4, r7, pc}
; THUMB2: .save   {r0, r4, r7, lr}
; THUMB2: push    {r0, r4, r7, lr}
; THUMB2: pop     {r0, r4, r7, pc}
  call void asm sideeffect "", "~{r0},~{r4}"()
  ret void
}

; This clobbers r0-r3, and already needs a push/pop, so we also save and
; restore all of them.
define void @test_r0_r1_r2_r3_r4() minsize nounwind {
; CHECK-LABEL: test_r0_r1_r2_r3_r4:
; CHECK: .save   {r0, r1, r2, r3, r4, lr}
; CHECK: push    {r0, r1, r2, r3, r4, lr}
; CHECK: pop     {r0, r1, r2, r3, r4, pc}
  call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r4}"()
  ret void
}

; Check that IPRA does make use of the extra saved registers.
define void @test_ipra() nounwind {
; CHECK-LABEL: test_ipra:
; CHECK: ASM1: r0, r1, r2, r3
; CHECK-NOT: r0
; CHECK-NOT: r1
; CHECK-NOT: r2
; CHECK-NOT: r3
; CHECK: bl      test_r0_r1_r2_r3_r4
; CHECK-NOT: r0
; CHECK-NOT: r1
; CHECK-NOT: r2
; CHECK-NOT: r3
; CHECK: ASM2: r0, r1, r2, r3
  %regs = call { i32, i32, i32, i32 } asm sideeffect "// ASM1: $0, $1, $2, $3", "={r0},={r1},={r2},={r3}"() 
  %r0 = extractvalue { i32, i32, i32, i32 } %regs, 0
  %r1 = extractvalue { i32, i32, i32, i32 } %regs, 1
  %r2 = extractvalue { i32, i32, i32, i32 } %regs, 2
  %r3 = extractvalue { i32, i32, i32, i32 } %regs, 3
  call void @test_r0_r1_r2_r3_r4()
  call void asm sideeffect "// ASM2: $0, $1, $2, $3", "{r0},{r1},{r2},{r3}"(i32 %r0, i32 %r1, i32 %r2, i32 %r3)
  ret void
}

; This clobbers r0-r3, but doesn't otherwise need a push/pop, so we don't add
; them.
define void @test_r0_r1_r2_r3() minsize nounwind {
; CHECK-LABEL: test_r0_r1_r2_r3:
; CHECK-NOT: push
; CHECK-NOT: pop
  call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3}"()
  ret void
}

; This isn't called in this function, so we don't push any extra registers.
define void @test_r0_r4_not_called() minsize nounwind {
; CHECK-LABEL: test_r0_r4_not_called:
; CHECK: .save   {r4, lr}
; CHECK: push    {r4, lr}
; CHECK: pop     {r4, pc}
; CHECK-NOT: push
; CHECK-NOT: pop
  call void asm sideeffect "", "~{r0},~{r4}"()
  ret void
}

; This function is only optsize, not minsize, so we don't add any extra saves.
define void @test_r0_r4_not_minsize() optsize nounwind {
; CHECK-LABEL: test_r0_r4_not_minsize:
; CHECK: .save   {r4, lr}
; CHECK: push    {r4, lr}
; CHECK: pop     {r4, pc}
; CHECK-NOT: push
; CHECK-NOT: pop
  call void asm sideeffect "", "~{r0},~{r4}"()
  ret void
}

; This function is not an exact definition (the linker could pick an
; alternative version of it), so we don't add any extra saves.
define linkonce_odr void @test_r0_r4_not_exact() minsize nounwind {
; CHECK-LABEL: test_r0_r4_not_exact:
; CHECK: .save   {r4, lr}
; CHECK: push    {r4, lr}
; CHECK: pop     {r4, pc}
; CHECK-NOT: push
; CHECK-NOT: pop
  call void asm sideeffect "", "~{r0},~{r4}"()
  ret void
}

; This clobbers r0-r3, but returns a value in r0, so only r1-r3 are saved.
define i32 @test_r0_r1_r2_r3_r4_return_1() minsize nounwind {
; CHECK-LABEL: test_r0_r1_r2_r3_r4_return_1:
; ARM: .save   {r1, r2, r3, r4, r11, lr}
; ARM: push    {r1, r2, r3, r4, r11, lr}
; ARM: pop     {r1, r2, r3, r4, r11, pc}
; THUMB1: .save   {r1, r2, r3, r4, r7, lr}
; THUMB1: push    {r1, r2, r3, r4, r7, lr}
; THUMB1: pop     {r1, r2, r3, r4, r7, pc}
; THUMB2: .save   {r1, r2, r3, r4, r7, lr}
; THUMB2: push    {r1, r2, r3, r4, r7, lr}
; THUMB2: pop     {r1, r2, r3, r4, r7, pc}
  call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r4}"()
  ret i32 42
}

; This clobbers r0-r3, but returns a value in r0 and r1, so only r2-r3 are
; saved.
define i64 @test_r0_r1_r2_r3_r4_return_2() minsize nounwind {
; CHECK-LABEL: test_r0_r1_r2_r3_r4_return_2:
; CHECK: .save   {r2, r3, r4, lr}
; CHECK: push    {r2, r3, r4, lr}
; CHECK: pop     {r2, r3, r4, pc}
  call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r4}"()
  ret i64 42
}

; This clobbers r0-r3, but returns a value in all of r0-r3, so none of them can
; be saved.
define i128 @test_r0_r1_r2_r3_r4_return_4() minsize nounwind {
; CHECK-LABEL: test_r0_r1_r2_r3_r4_return_4:
; CHECK: .save   {r4, lr}
; CHECK: push    {r4, lr}
; CHECK: pop     {r4, pc}
  call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r4}"()
  ret i128 42
}

; This clobbers r0-r3, and returns a value in s0, so all of r0-r3 are saved (we
; previously only checked the number of return registers, ignoring their
; class).
define arm_aapcs_vfpcc float @test_r0_r1_r2_r3_r4_return_float() minsize nounwind {
; CHECK-LABEL: test_r0_r1_r2_r3_r4_return_float:
; ARM: .save   {r0, r1, r2, r3, r4, lr}
; ARM: push    {r0, r1, r2, r3, r4, lr}
; ARM: pop     {r0, r1, r2, r3, r4, pc}
; THUMB1: .save   {r1, r2, r3, r4, r7, lr}
; THUMB1: push    {r1, r2, r3, r4, r7, lr}
; THUMB1: pop     {r1, r2, r3, r4, r7, pc}
; THUMB2: .save   {r0, r1, r2, r3, r4, lr}
; THUMB2: push    {r0, r1, r2, r3, r4, lr}
; THUMB2: pop     {r0, r1, r2, r3, r4, pc}
  call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r4}"()
  ret float 42.0
}

; Saving of high registers in thumb1 is more complicated, because they need to
; be copied down to low registers to use push/pop instructions. Luckily, the
; extra registers we are preserving are low registers, which are handled by the
; outer-most push/pop pair, so this doesn't interact badly.
define void @test_save_high_regs() minsize nounwind {
; CHECK-LABEL: test_save_high_regs:
; ARM: .save   {r0, r1, r2, r3, r7, r8, r9, r10, r11, lr}
; ARM: push    {r0, r1, r2, r3, r7, r8, r9, r10, r11, lr}
; ARM: pop     {r0, r1, r2, r3, r7, r8, r9, r10, r11, pc}
; THUMB1:      .save   {r0, r1, r2, r3, r7, lr}
; THUMB1-NEXT: push    {r0, r1, r2, r3, r7, lr}
; THUMB1-NEXT: mov     lr, r11
; THUMB1-NEXT: mov     r7, r10
; THUMB1-NEXT: mov     r3, r9
; THUMB1-NEXT: mov     r2, r8
; THUMB1-NEXT: .save   {r8, r9, r10, r11}
; THUMB1-NEXT: push    {r2, r3, r7, lr}
; THUMB1:      pop     {r0, r1, r2, r3}
; THUMB1-NEXT: mov     r8, r0
; THUMB1-NEXT: mov     r9, r1
; THUMB1-NEXT: mov     r10, r2
; THUMB1-NEXT: mov     r11, r3
; THUMB1-NEXT: pop     {r0, r1, r2, r3, r7, pc}
; THUMB2: .save   {r0, r1, r2, r3, r7, r8, r9, r10, r11, lr}
; THUMB2: push.w  {r0, r1, r2, r3, r7, r8, r9, r10, r11, lr}
; THUMB2: pop.w   {r0, r1, r2, r3, r7, r8, r9, r10, r11, pc}
  call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r8},~{r9},~{r10},~{r11}"()
  ret void
}

; We can also use extra registers in the PUSH/POP instructions to move the SP
; to make space for local variables. These registers aren't preserved, because
; the space they are saved in is used for the local variable. We try to back
; off the extra-CSRs optimisation to allow this to still happen. In this case,
; there are 8 bytes of stack space needed, so we preserve two argument
; registers and use the other two for the SP update.
define void @test_r0_r1_r2_r3_r4_stack8() minsize nounwind {
; CHECK-LABEL: test_r0_r1_r2_r3_r4_stack8:
; CHECK: .save   {r2, r3, r4, lr}
; CHECK: push    {r0, r1, r2, r3, r4, lr}
; CHECK: pop     {r0, r1, r2, r3, r4, pc}
  %a = alloca [2 x i32], align 4
  call void asm sideeffect "str $1, [$0]; str $1, [$0, #4]", "{r0},{r1},~{r2},~{r3},~{r4}"([2 x i32]* %a, i32 42)
  ret void
}

; Check that, when the above function is called, r0 and r1 (used for the SP
; updates) are considered clobbered, and r2 and r3 are preserved.
define void @test_r0_r1_r2_r3_r4_stack8_caller() nounwind {
; CHECK-LABEL: test_r0_r1_r2_r3_r4_stack8_caller:
; CHECK:      ASM1: r0, r1, r2, r3
; CHECK-NEXT: @NO_APP
; CHECK-NEXT: mov     r4, r0
; CHECK-NEXT: mov     r5, r1
; CHECK-NEXT: bl      test_r0_r1_r2_r3_r4
; CHECK-NEXT: mov     r0, r4
; CHECK-NEXT: mov     r1, r5
; CHECK-NEXT: @APP
; CHECK-NEXT: ASM2: r0, r1, r2, r3
  %regs = call { i32, i32, i32, i32 } asm sideeffect "// ASM1: $0, $1, $2, $3", "={r0},={r1},={r2},={r3}"() 
  %r0 = extractvalue { i32, i32, i32, i32 } %regs, 0
  %r1 = extractvalue { i32, i32, i32, i32 } %regs, 1
  %r2 = extractvalue { i32, i32, i32, i32 } %regs, 2
  %r3 = extractvalue { i32, i32, i32, i32 } %regs, 3
  call void @test_r0_r1_r2_r3_r4_stack8()
  call void asm sideeffect "// ASM2: $0, $1, $2, $3", "{r0},{r1},{r2},{r3}"(i32 %r0, i32 %r1, i32 %r2, i32 %r3)
  ret void
}

; Like @test_r0_r1_r2_r3_r4_stack8, but 16 bytes of stack space are needed, so
; all of r0-r3 are used for the SP update, and not preserved.
define void @test_r0_r1_r2_r3_r4_stack16() minsize nounwind {
; CHECK-LABEL: test_r0_r1_r2_r3_r4_stack16:
; CHECK: .save   {r4, lr}
; CHECK: push    {r0, r1, r2, r3, r4, lr}
; CHECK: pop     {r0, r1, r2, r3, r4, pc}
  %a = alloca [4 x i32], align 4
  call void asm sideeffect "str $1, [$0]; str $1, [$0, #4]", "{r0},{r1},~{r2},~{r3},~{r4}"([4 x i32]* %a, i32 42)
  ret void
}

; If more than 16 bytes of stack space are needed, it's unlikely that the
; SP-update folding optimisation will succeed, so we revert back to preserving
; r0-r3 for use in our callers.
define void @test_r0_r1_r2_r3_r4_stack24() minsize nounwind {
; CHECK-LABEL: test_r0_r1_r2_r3_r4_stack24:
; CHECK: .save   {r0, r1, r2, r3, r4, lr}
; CHECK: push    {r0, r1, r2, r3, r4, lr}
; CHECK: pop     {r0, r1, r2, r3, r4, pc}
  %a = alloca [6 x i32], align 4
  call void asm sideeffect "str $1, [$0]; str $1, [$0, #4]", "{r0},{r1},~{r2},~{r3},~{r4}"([6 x i32]* %a, i32 42)
  ret void
}

define i32 @tail_callee(i32 %a, i32 %b) minsize nounwind {
entry:
  tail call void asm sideeffect "", "~{r2}"()
  ret i32 %a
}

; The tail call happens outside the save/restore region, so prevents us from
; preserving some registers. r0 and r1 are outgoing arguments to the tail-call,
; so can't be preserved. r2 is modified inside the tail-called function, so
; can't be presrved. r3 is known to be preserved by the callee, so can be
; presrved. For Thumb1, we can't (efficiently) use a tail-call here, so r1-r3
; are all preserved, with r0 being the return value.
define i32 @test_tail_call() minsize nounwind {
entry:
; CHECK-LABEL: test_tail_call:
; ARM: .save   {r3, lr}
; ARM: push    {r3, lr}
; ARM: pop     {r3, lr}
; ARM: b       tail_callee
; THUMB2: .save   {r3, lr}
; THUMB2: push    {r3, lr}
; THUMB2: pop.w   {r3, lr}
; THUMB2: b       tail_callee
; THUMB1: .save   {r1, r2, r3, lr}
; THUMB1: push    {r1, r2, r3, lr}
; THUMB1: bl      tail_callee
; THUMB1: pop     {r1, r2, r3, pc}
  tail call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{lr}"()
  %call = tail call i32 @tail_callee(i32 3, i32 4)
  ret i32 %call
}

declare i32 @tail_callee_external(i32 %a, i32 %b)

; If we tail-call an external function, it could clobber any of r0-r3.
define i32 @test_tail_call_external() minsize nounwind {
entry:
; CHECK-LABEL: test_tail_call_external:
; ARM: .save   {r11, lr}
; ARM: push    {r11, lr}
; ARM: pop     {r11, lr}
; ARM: b       tail_callee_external
; THUMB2: .save   {r7, lr}
; THUMB2: push    {r7, lr}
; THUMB2: pop.w   {r7, lr}
; THUMB2: b       tail_callee_external
; THUMB1: .save   {r1, r2, r3, lr}
; THUMB1: push    {r1, r2, r3, lr}
; THUMB1: bl      tail_callee_external
; THUMB1: pop     {r1, r2, r3, pc}
  tail call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{lr}"()
  %call = tail call i32 @tail_callee_external(i32 3, i32 4)
  ret i32 %call
}

define linkonce_odr i32 @tail_callee_linkonce_odr(i32 %a, i32 %b) minsize nounwind {
entry:
  tail call void asm sideeffect "", "~{r2}"()
  ret i32 %a
}

; If a tail-callee has an interposable linkage type (such as linkonce_odr), we
; can't assume the linker will pick the definition we can see, so must assume
; it clobbers all of r0-r3.
define i32 @test_tail_call_linkonce_odr() minsize nounwind {
entry:
; CHECK-LABEL: test_tail_call_linkonce_odr:
; ARM: .save   {r11, lr}
; ARM: push    {r11, lr}
; ARM: pop     {r11, lr}
; ARM: b       tail_callee_linkonce_odr
; THUMB2: .save   {r7, lr}
; THUMB2: push    {r7, lr}
; THUMB2: pop.w   {r7, lr}
; THUMB2: b       tail_callee_linkonce_odr
; THUMB1: .save   {r1, r2, r3, lr}
; THUMB1: push    {r1, r2, r3, lr}
; THUMB1: bl      tail_callee_linkonce_odr
; THUMB1: pop     {r1, r2, r3, pc}
  tail call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{lr}"()
  %call = tail call i32 @tail_callee_linkonce_odr(i32 3, i32 4)
  ret i32 %call
}

; This function doesn't have the nounwind attribute, so unwind tables will be
; emitted. Saving r0-r3 requires a longer unwind instruction sequence, which
; results in an increase in total code size if there are few callers to make
; use of the extra registers.
define void @test_unwind_tables() minsize {
; CHECK-LABEL: test_unwind_tables:
; ARM: .save   {r4, lr}
; ARM: push    {r4, lr}
; ARM: pop     {r4, pc}
; THUMB1: .save   {r4, lr}
; THUMB1: push    {r4, lr}
; THUMB1: pop     {r4, pc}
; THUMB2: .save   {r4, lr}
; THUMB2: push    {r4, lr}
; THUMB2: pop     {r4, pc}
  call void asm sideeffect "", "~{r0},~{r4}"()
  ret void
}

; This requires an unwind table, but has many call sites, so overall we expect
; the benefits to outweigh the size increase of the unwind table.
define void @test_unwind_tables_many_calls() minsize {
; CHECK-LABEL: test_unwind_tables_many_calls:
; ARM: .save   {r0, r4, r11, lr}
; ARM: push    {r0, r4, r11, lr}
; ARM: pop     {r0, r4, r11, pc}
; THUMB1: .save   {r0, r4, r7, lr}
; THUMB1: push    {r0, r4, r7, lr}
; THUMB1: pop     {r0, r4, r7, pc}
; THUMB2: .save   {r0, r4, r7, lr}
; THUMB2: push    {r0, r4, r7, lr}
; THUMB2: pop     {r0, r4, r7, pc}
  call void asm sideeffect "", "~{r0},~{r4}"()
  ret void
}

; We don't do this optimisation is there are no callers in the same translation
; unit (otherwise IPRA wouldn't be able to take advantage of the extra saved
; registers), so most functions in this file are called here.
define void @caller() {
; CHECK-LABEL: caller:
  call void @test_r0_r4()
  call void @test_r0_r1_r2_r3_r4()
  call void @test_r0_r1_r2_r3()
  call void @test_r0_r4_not_minsize()
  call void @test_r0_r4_not_exact()
  %t1 = call i32 @test_r0_r1_r2_r3_r4_return_1()
  %t2 = call i64 @test_r0_r1_r2_r3_r4_return_2()
  %t3 = call i128 @test_r0_r1_r2_r3_r4_return_4()
  %t4 = call float @test_r0_r1_r2_r3_r4_return_float()
  call void @test_save_high_regs()
  call void @test_r0_r1_r2_r3_r4_stack16()
  call void @test_r0_r1_r2_r3_r4_stack24()
  %t5 = call i32 @test_tail_call()
  %t6 = call i32 @test_tail_call_external()
  %t7 = call i32 @test_tail_call_linkonce_odr()
  call void @test_unwind_tables()
  call void @test_unwind_tables_many_calls()
  call void @test_unwind_tables_many_calls()
  call void @test_unwind_tables_many_calls()
  call void @test_unwind_tables_many_calls()
  call void @test_unwind_tables_many_calls()
  call void @test_unwind_tables_many_calls()
  call void @test_unwind_tables_many_calls()
  call void @test_unwind_tables_many_calls()
  call void @test_unwind_tables_many_calls()
  ret void
}
