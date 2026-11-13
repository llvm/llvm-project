; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64-v3 -enable-post-misched -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs -stop-after=kcfi < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs -stop-after=kcfi -enable-new-pm < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs -function-splitting=all < %s | FileCheck %s --check-prefix=SPLIT

;; With the kcfi-out-of-line-traps module flag, a KCFI check ends in a branch
;; that's taken if the check fails, to a block that only has the trap, and the
;; call starts the block that the check falls through to.

define void @f1(ptr noundef %x) {
; ASM-LABEL: f1:
; ASM:         movl $4282621618, %r10d # imm = 0xFF439EB2
; ASM-NEXT:    addl -4(%rdi), %r10d
; ASM-NEXT:    jne [[TRAP:.LBB[0-9_]+]]
; ASM-NEXT:  # %bb.{{[0-9]+}}:
; ASM-NEXT:    callq *%rdi
; ASM:         retq
;; The trap is laid out with the cold blocks, and has the right CFA.
; ASM-NEXT:  [[TRAP]]:
; ASM-NEXT:    .cfi_def_cfa_offset 16
; ASM-NEXT:  [[UD2:.Ltmp[0-9]+]]:
; ASM-NEXT:    ud2
; ASM-NEXT:    .section .kcfi_traps,"ao",@progbits,.text
; ASM-NEXT:  [[ENTRY:.Ltmp[0-9]+]]:
; ASM-NEXT:    .long [[UD2]]-[[ENTRY]]
; ASM-NEXT:    .text

; MIR-LABEL: name: f1
; MIR:       bb.0 (%ir-block.0):
; MIR-NEXT:    successors: %bb.2(0x80000000), %bb.1(0x00000000)
; MIR:         KCFI_CMP $rdi, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; MIR-NEXT:    JCC_1 %bb.1, 5, implicit $eflags
; MIR:       bb.2 (%ir-block.0):
; MIR-NEXT:    liveins: {{.*}}$rdi
; MIR:         CALL64r killed $rdi, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp
; MIR:       bb.1 (%ir-block.0):
; MIR-NEXT:    nomerge KCFI_TRAP
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

define void @f2(ptr noundef %x) {
; ASM-LABEL: f2:
; ASM:         addl -4(%rdi), %r10d
; ASM-NEXT:    jne [[TRAP:.LBB[0-9_]+]]
; ASM-NEXT:  # %bb.{{[0-9]+}}:
; ASM-NEXT:    jmpq *%rdi # TAILCALL
; ASM-NEXT:  [[TRAP]]:
; ASM-NEXT:  .Ltmp{{[0-9]+}}:
; ASM-NEXT:    ud2

; MIR-LABEL: name: f2
; MIR:         KCFI_CMP $rdi, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; MIR-NEXT:    JCC_1 %bb.1, 5, implicit $eflags
; MIR:       bb.2 (%ir-block.0):
; MIR:         TAILJMPr64 killed $rdi, csr_64, implicit $rsp, implicit $ssp, implicit $rsp, implicit $ssp
; MIR:       bb.1 (%ir-block.0):
; MIR-NEXT:    nomerge KCFI_TRAP
  tail call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

;; A call through an indirect thunk has its target in r11.
define void @f3(ptr noundef %x) #0 {
; ASM-LABEL: f3:
; ASM:         addl -4(%r11), %r10d
; ASM-NEXT:    jne [[TRAP:.LBB[0-9_]+]]
; ASM-NEXT:  # %bb.{{[0-9]+}}:
; ASM-NEXT:    jmp __llvm_retpoline_r11 # TAILCALL
; ASM-NEXT:  [[TRAP]]:
; ASM-NEXT:  .Ltmp{{[0-9]+}}:
; ASM-NEXT:    ud2

; MIR-LABEL: name: f3
; MIR:         KCFI_CMP $r11, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; MIR-NEXT:    JCC_1 %bb.1, 5, implicit $eflags
; MIR:       bb.2 (%ir-block.0):
; MIR-NEXT:    liveins: {{.*}}$r11
; MIR:         TAILJMPd64 &__llvm_retpoline_r11, csr_64, implicit $rsp, implicit $ssp, implicit $rsp, implicit $ssp, implicit killed $r11
; MIR:       bb.1 (%ir-block.0):
; MIR-NEXT:    nomerge KCFI_TRAP
  tail call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

@g = external local_unnamed_addr global ptr, align 8

;; So does one that had its memory operand unfolded.
define void @f4() {
; ASM-LABEL: f4:
; ASM:         movq g@GOTPCREL(%rip), [[PTR:%r[a-z0-9]+]]
; ASM-NEXT:    movq ([[PTR]]), [[CALLEE:%r[a-z0-9]+]]
; ASM:         addl -4([[CALLEE]]), %r10d
; ASM-NEXT:    jne [[TRAP:\.LBB[0-9_]+]]
; ASM:         callq *[[CALLEE]]
; ASM:       [[TRAP]]:
; ASM:         ud2

; MIR-LABEL: name: f4
; MIR:         $r11 = MOV64rm killed renamable $rax, 1, $noreg, 0, $noreg
; MIR-NEXT:    KCFI_CMP $r11, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; MIR-NEXT:    JCC_1 %bb.1, 5, implicit $eflags
; MIR:       bb.2 (%ir-block.0):
; MIR-NEXT:    liveins: {{.*}}$r11
; MIR:         CALL64r $r11, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp
; MIR:       bb.1 (%ir-block.0):
; MIR-NEXT:    nomerge KCFI_TRAP
  %1 = load ptr, ptr @g, align 8
  call void %1() [ "kcfi"(i32 12345678) ]
  ret void
}

;; Each check has a trap of its own.
define void @f5(ptr noundef %x, ptr noundef %y) {
; ASM-LABEL: f5:
; ASM:         addl -4(%rdi), %r10d
; ASM-NEXT:    jne [[TRAP1:.LBB[0-9_]+]]
; ASM-NEXT:  # %bb.{{[0-9]+}}:
; ASM-NEXT:    callq *%rdi
; ASM-NEXT:    movl $4282621618, %r10d # imm = 0xFF439EB2
; ASM-NEXT:    addl -4(%rbx), %r10d
; ASM-NEXT:    jne [[TRAP2:.LBB[0-9_]+]]
; ASM-NEXT:  # %bb.{{[0-9]+}}:
; ASM-NEXT:    callq *%rbx
; ASM:         retq
; ASM-NEXT:  [[TRAP1]]:
; ASM-NEXT:    .cfi_def_cfa_offset 16
; ASM-NEXT:  .Ltmp{{[0-9]+}}:
; ASM-NEXT:    ud2
; ASM:       [[TRAP2]]:
; ASM-NEXT:  .Ltmp{{[0-9]+}}:
; ASM-NEXT:    ud2

; MIR-LABEL: name: f5
; MIR:       bb.0 (%ir-block.0):
; MIR-NEXT:    successors: %bb.4(0x80000000), %bb.3(0x00000000)
; MIR:         KCFI_CMP $rdi, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; MIR-NEXT:    JCC_1 %bb.3, 5, implicit $eflags
; MIR:       bb.4 (%ir-block.0):
; MIR-NEXT:    successors: %bb.2(0x80000000), %bb.1(0x00000000)
; MIR:         CALL64r killed $rdi, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp
; MIR-NEXT:    KCFI_CMP $rbx, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; MIR-NEXT:    JCC_1 %bb.1, 5, implicit $eflags
; MIR:       bb.2 (%ir-block.0):
; MIR:         CALL64r killed $rbx, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp
; MIR:       bb.1 (%ir-block.0):
; MIR:         nomerge KCFI_TRAP
; MIR:       bb.3 (%ir-block.0):
; MIR-NEXT:    nomerge KCFI_TRAP
  call void %x() [ "kcfi"(i32 12345678) ]
  call void %y() [ "kcfi"(i32 12345678) ]
  ret void
}

declare i32 @__gxx_personality_v0(...)
declare void @cleanup()

;; A non-invoke call preceding another call/invoke in a block with an EH pad
;; successor keeps an inline bundled check.
define void @f6(ptr noundef %x) personality ptr @__gxx_personality_v0 {
; ASM-LABEL: f6:
; ASM:         addl -4(%rdi), %r10d
; ASM-NEXT:    je [[PASS:\.Ltmp[0-9]+]]
; ASM-NEXT:  .Ltmp{{[0-9]+}}:
; ASM-NEXT:    ud2
; ASM:       [[PASS]]:
; ASM-NEXT:    callq *%rdi
; ASM-NEXT:  .Ltmp{{[0-9]+}}:
; ASM-NEXT:    callq cleanup@PLT

; MIR-LABEL: name: f6
; MIR:       BUNDLE{{.*}} {
; MIR-NEXT:    KCFI_CHECK $rdi, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; MIR-NEXT:    CALL64r killed $rdi, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp
; MIR-NEXT:  }
; MIR-NEXT:  EH_LABEL
; MIR-NEXT:  CALL64pcrel32 target-flags(x86-plt) @cleanup
; MIR-NEXT:  EH_LABEL
  call void %x() [ "kcfi"(i32 12345678) ]
  invoke void @cleanup()
    to label %cont
    unwind label %err
cont:
  ret void
err:
  %exn = landingpad { ptr, i32 }
          cleanup
  call void @cleanup()
  resume { ptr, i32 } %exn
}

;; With a profile, the trap is cold, so it's split from the rest of the
;; function.
define void @f7(ptr noundef %x) !prof !16 {
; SPLIT-LABEL: f7:
; SPLIT:         addl -4(%rdi), %r10d
; SPLIT-NEXT:    jne f7.cold
; SPLIT-NEXT:  # %bb.{{[0-9]+}}:
; SPLIT-NEXT:    callq *%rdi
; SPLIT:         .section .text.split.f7,"ax",@progbits
; SPLIT-NEXT:  f7.cold:
; SPLIT-NEXT:    .cfi_startproc
; SPLIT-NEXT:    .cfi_def_cfa %rsp, 16
; SPLIT-NEXT:  .Ltmp{{[0-9]+}}:
; SPLIT-NEXT:    ud2
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

attributes #0 = { "target-features"="+retpoline-indirect-branches,+retpoline-indirect-calls" }

!llvm.module.flags = !{!0, !1, !2}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 8, !"kcfi-out-of-line-traps", i32 1}
!2 = !{i32 1, !"ProfileSummary", !3}
!3 = !{!4, !5, !6, !7, !8, !9, !10, !11}
!4 = !{!"ProfileFormat", !"InstrProf"}
!5 = !{!"TotalCount", i64 10000}
!6 = !{!"MaxCount", i64 10}
!7 = !{!"MaxInternalCount", i64 1}
!8 = !{!"MaxFunctionCount", i64 1000}
!9 = !{!"NumCounts", i64 3}
!10 = !{!"NumFunctions", i64 5}
!11 = !{!"DetailedSummary", !12}
!12 = !{!13, !14, !15}
!13 = !{i32 10000, i64 100, i32 1}
!14 = !{i32 999900, i64 100, i32 1}
!15 = !{i32 999999, i64 1, i32 2}
!16 = !{!"function_entry_count", i64 7000}
