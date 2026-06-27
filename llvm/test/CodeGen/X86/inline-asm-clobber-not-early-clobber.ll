; RUN: llc < %s -mtriple=x86_64-linux-gnu 2>&1 | FileCheck %s
;
; Verify that clobber-list registers (~{reg}) are NOT treated as early-clobber
; by the register allocator.
;
; Before the fix, Kind::Clobber fell through into the Kind::RegDefEarlyClobber
; case in InstrEmitter::EmitSpecialNode(), causing every clobbered register to
; receive the EarlyClobber MachineOperand flag.  With 11 clobbered registers
; and -frame-pointer=all (rbp reserved), only rsi/rdi/r12 remained usable, but
; the asm needed 3 operand registers plus a live-across pointer — causing:
;   "inline assembly requires more registers than available"
;
; GCC semantics: clobber-list registers are destroyed *during* the asm body,
; not before inputs are read, so inputs are allowed to use them.

@ctr0 = internal global [1 x i64] zeroinitializer
@ctr1 = internal global [1 x i64] zeroinitializer

; CHECK-NOT: error:
; CHECK-LABEL: test3:
; CHECK:         nop
; CHECK:         nop
define dso_local void @test3(ptr noundef %q) local_unnamed_addr #0 {
  ; Simulate gcov counter increment between the two asm calls — this is what
  ; creates register pressure and triggers the bug in practice.
  %c0 = load i64, ptr @ctr0, align 8
  %c1 = add i64 %c0, 1
  store i64 %c1, ptr @ctr0, align 8

  call { ptr, ptr } asm sideeffect "nop",
      "=&r,=&r,r,0,1,~{rax},~{rbx},~{rcx},~{rdx},~{r8},~{r9},~{r10},~{r11},~{r13},~{r14},~{r15},~{memory},~{cc},~{dirflag},~{fpsr},~{flags}"
      (ptr nonnull inttoptr (i64 4660 to ptr),
       ptr nonnull inttoptr (i64 43981 to ptr),
       ptr nonnull inttoptr (i64 239 to ptr)) #1

  %c2 = load i64, ptr @ctr1, align 8
  %c3 = add i64 %c2, 1
  store i64 %c3, ptr @ctr1, align 8

  call { ptr, ptr } asm sideeffect "nop",
      "=&r,=&r,r,0,1,~{rax},~{rbx},~{rcx},~{rdx},~{r8},~{r9},~{r10},~{r11},~{r13},~{r14},~{r15},~{memory},~{cc},~{dirflag},~{fpsr},~{flags}"
      (ptr %q,
       ptr nonnull inttoptr (i64 6699 to ptr),
       ptr nonnull inttoptr (i64 15437 to ptr)) #1

  ret void
}

attributes #0 = { nounwind uwtable "frame-pointer"="all" "target-cpu"="x86-64" }
attributes #1 = { nounwind }
