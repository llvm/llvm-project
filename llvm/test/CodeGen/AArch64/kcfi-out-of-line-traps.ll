; RUN: llc -mtriple=aarch64-- -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=aarch64-- -verify-machineinstrs -stop-after=kcfi < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=aarch64-- -verify-machineinstrs -function-splitting=all < %s | FileCheck %s --check-prefix=SPLIT
; RUN: llc -mtriple=aarch64-- -verify-machineinstrs -basic-block-sections=all < %s | FileCheck %s --check-prefix=BBS

;; With the kcfi-out-of-line-traps module flag, a KCFI check ends in a branch
;; that's taken if the check fails, to a block that only has the trap, and the
;; call starts the block that the check falls through to. The trap has the ESR
;; that an inline one has.

define void @f1(ptr noundef %x) {
; ASM-LABEL: f1:
; ASM:         ldur w16, [x0, #-4]
; ASM-NEXT:    movk w17, #24910
; ASM-NEXT:    movk w17, #188, lsl #16
; ASM-NEXT:    cmp w16, w17
; ASM-NEXT:    b.ne [[TRAP:.LBB[0-9_]+]]
; ASM-NEXT:  // %bb.{{[0-9]+}}:
; ASM-NEXT:    blr x0
; ASM:         ret
; ASM-NEXT:  [[TRAP]]:
; ASM-NEXT:    brk #0x8220

; MIR-LABEL: name: f1
; MIR:       bb.0 (%ir-block.0):
; MIR-NEXT:    successors: %bb.2(0x80000000), %bb.1(0x00000000)
; MIR:         KCFI_CMP $x0, 12345678, implicit-def $x9, implicit-def $x16, implicit-def $x17, implicit-def $nzcv
; MIR-NEXT:    Bcc 1, %bb.1, implicit $nzcv
; MIR:       bb.2 (%ir-block.0):
; MIR-NEXT:    liveins: {{.*}}$x0
; MIR:         BLR killed $x0, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit-def $sp
; MIR:       bb.1 (%ir-block.0):
; MIR-NEXT:    liveins: $x0, $x17
; MIR:         nomerge KCFI_TRAP $x0, $w17
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

define void @f2(ptr noundef %x) {
; ASM-LABEL: f2:
; ASM:         cmp w16, w17
; ASM-NEXT:    b.ne [[TRAP:.LBB[0-9_]+]]
; ASM-NEXT:  // %bb.{{[0-9]+}}:
; ASM-NEXT:    br x0
; ASM-NEXT:  [[TRAP]]:
; ASM-NEXT:    brk #0x8220

; MIR-LABEL: name: f2
; MIR:         KCFI_CMP $x0, 12345678, implicit-def $x9, implicit-def $x16, implicit-def $x17, implicit-def $nzcv
; MIR-NEXT:    Bcc 1, %bb.1, implicit $nzcv
; MIR:       bb.2 (%ir-block.0):
; MIR:         TCRETURNri killed $x0, 0, csr_aarch64_aapcs, implicit $sp
; MIR:       bb.1 (%ir-block.0):
; MIR-NEXT:    liveins: $x0, $x17
; MIR:         nomerge KCFI_TRAP $x0, $w17
  tail call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

;; If the target is in one of the registers that a check compares the types
;; in, the check uses w9 instead, and the trap says so.
define void @f3(ptr noundef %x) #1 {
; ASM-LABEL: f3:
; ASM:         ldur w9, [x16, #-4]
; ASM-NEXT:    movk w17, #24910
; ASM-NEXT:    movk w17, #188, lsl #16
; ASM-NEXT:    cmp w9, w17
; ASM-NEXT:    b.ne [[TRAP:.LBB[0-9_]+]]
; ASM-NEXT:  // %bb.{{[0-9]+}}:
; ASM-NEXT:    br x16
; ASM-NEXT:  [[TRAP]]:
; ASM-NEXT:    brk #0x8230

; MIR-LABEL: name: f3
; MIR:         KCFI_CMP $x16, 12345678, implicit-def $x9, implicit-def $x16, implicit-def $x17, implicit-def $nzcv
; MIR-NEXT:    Bcc 1, %bb.1, implicit $nzcv
; MIR:       bb.2 (%ir-block.0):
; MIR-NEXT:    liveins: {{.*}}$x16
; MIR:         TCRETURNrix16x17 killed $x16, 0, csr_aarch64_aapcs, implicit $sp
; MIR:       bb.1 (%ir-block.0):
; MIR-NEXT:    liveins: $x16, $x17
; MIR:         nomerge KCFI_TRAP $x16, $w17

;; The registers that the trap reports are live into its block. A branch to
;; another section can get a veneer from the linker, which clobbers x16, so
;; one to the trap saves x16 if the target is in it.
; BBS-LABEL: f3:
; BBS:         cmp w9, w17
; BBS-NEXT:    b.eq [[PASS:.LBB[0-9_]+]]
; BBS-NEXT:  // %bb.{{[0-9]+}}:
; BBS-NEXT:    str x16, [sp, #-16]!
; BBS-NEXT:    b [[TRAP:f3.__part.[0-9]+]]
; BBS:       [[TRAP]]:
; BBS-NEXT:    .cfi_startproc
; BBS-NEXT:    ldr x16, [sp], #16
; BBS-NEXT:  // %bb.{{[0-9]+}}:
; BBS-NEXT:    brk #0x8230
  tail call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

define void @f4(ptr noundef %x) #2 {
; ASM-LABEL: f4:
; ASM:         ldur w16, [x17, #-4]
; ASM-NEXT:    movk w9, #24910
; ASM-NEXT:    movk w9, #188, lsl #16
; ASM-NEXT:    cmp w16, w9
; ASM-NEXT:    b.ne [[TRAP:.LBB[0-9_]+]]
; ASM-NEXT:  // %bb.{{[0-9]+}}:
; ASM-NEXT:    br x17
; ASM-NEXT:  [[TRAP]]:
; ASM-NEXT:    brk #0x8131

; MIR-LABEL: name: f4
; MIR:         KCFI_CMP $x17, 12345678, implicit-def $x9, implicit-def $x16, implicit-def $x17, implicit-def $nzcv
; MIR-NEXT:    Bcc 1, %bb.1, implicit $nzcv
; MIR:       bb.2 (%ir-block.0):
; MIR-NEXT:    liveins: {{.*}}$x17
; MIR:         TCRETURNrix17 killed $x17, 0, csr_aarch64_aapcs, implicit $sp
; MIR:       bb.1 (%ir-block.0):
; MIR-NEXT:    liveins: $x17, $x9
; MIR:         nomerge KCFI_TRAP $x17, $w9
  tail call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

;; Each check has a trap of its own.
define void @f5(ptr noundef %x, ptr noundef %y) {
; ASM-LABEL: f5:
; ASM:         cmp w16, w17
; ASM-NEXT:    b.ne [[TRAP1:.LBB[0-9_]+]]
; ASM-NEXT:  // %bb.{{[0-9]+}}:
; ASM-NEXT:    blr x0
; ASM-NEXT:    ldur w16, [x19, #-4]
; ASM-NEXT:    movk w17, #24910
; ASM-NEXT:    movk w17, #188, lsl #16
; ASM-NEXT:    cmp w16, w17
; ASM-NEXT:    b.ne [[TRAP2:.LBB[0-9_]+]]
; ASM-NEXT:  // %bb.{{[0-9]+}}:
; ASM-NEXT:    blr x19
; ASM:         ret
; ASM-NEXT:  [[TRAP1]]:
; ASM-NEXT:    brk #0x8220
; ASM-NEXT:  [[TRAP2]]:
; ASM-NEXT:    brk #0x8233

; MIR-LABEL: name: f5
; MIR:       bb.0 (%ir-block.0):
; MIR-NEXT:    successors: %bb.4(0x80000000), %bb.3(0x00000000)
; MIR:         KCFI_CMP $x0, 12345678, implicit-def $x9, implicit-def $x16, implicit-def $x17, implicit-def $nzcv
; MIR-NEXT:    Bcc 1, %bb.3, implicit $nzcv
; MIR:       bb.4 (%ir-block.0):
; MIR-NEXT:    successors: %bb.2(0x80000000), %bb.1(0x00000000)
; MIR:         BLR killed $x0, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit-def $sp
; MIR-NEXT:    KCFI_CMP $x19, 12345678, implicit-def $x9, implicit-def $x16, implicit-def $x17, implicit-def $nzcv
; MIR-NEXT:    Bcc 1, %bb.1, implicit $nzcv
; MIR:       bb.2 (%ir-block.0):
; MIR:         BLR killed $x19, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit-def $sp
; MIR:       bb.1 (%ir-block.0):
; MIR:         nomerge KCFI_TRAP $x19, $w17
; MIR:       bb.3 (%ir-block.0):
; MIR:         nomerge KCFI_TRAP $x0, $w17
  call void %x() [ "kcfi"(i32 12345678) ]
  call void %y() [ "kcfi"(i32 12345678) ]
  ret void
}

;; A call that's already bundled, here with the BTI that follows it, stays
;; bundled with its check, and keeps an inline trap.
define void @f6(ptr noundef %x) #1 {
; ASM-LABEL: f6:
; ASM:         cmp w16, w17
; ASM-NEXT:    b.eq [[PASS:.Ltmp[0-9]+]]
; ASM-NEXT:    brk #0x8220
; ASM-NEXT:  [[PASS]]:
; ASM-NEXT:    blr x0
; ASM-NEXT:    hint #36

; MIR-LABEL: name: f6
; MIR:       BUNDLE{{.*}} {
; MIR-NEXT:    KCFI_CHECK $x0, 12345678, implicit-def $x9, implicit-def $x16, implicit-def $x17, implicit-def $nzcv
; MIR-NEXT:    BLR killed $x0, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit-def $sp
; MIR-NEXT:    HINT 36
; MIR-NEXT:  }
  call void %x() #0 [ "kcfi"(i32 12345678) ]
  ret void
}

declare i32 @__gxx_personality_v0(...)
declare void @cleanup()

;; A call in a block that has an EH pad successor can be an invoke, which has
;; to stay next to the EH label that starts its try-range. It keeps an inline
;; trap.
define void @f7(ptr noundef %x) personality ptr @__gxx_personality_v0 {
; ASM-LABEL: f7:
; ASM:         cmp w16, w17
; ASM-NEXT:    b.eq [[PASS:.Ltmp[0-9]+]]
; ASM-NEXT:    brk #0x8220
; ASM-NEXT:  [[PASS]]:
; ASM-NEXT:    blr x0

; MIR-LABEL: name: f7
; MIR:       BUNDLE{{.*}} {
; MIR-NEXT:    KCFI_CHECK $x0, 12345678, implicit-def $x9, implicit-def $x16, implicit-def $x17, implicit-def $nzcv
; MIR-NEXT:    BLR killed $x0, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit-def $sp
; MIR-NEXT:  }
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

;; A conditional branch can't go to another section, so the trap isn't split
;; from the rest of the function with the cold blocks. That would have every
;; check that passes take a branch over an unconditional one.
define void @f8(ptr noundef %x, i1 %c) !prof !17 {
; SPLIT-LABEL: f8:
; SPLIT:         cmp w16, w17
; SPLIT-NEXT:    b.ne [[TRAP:.LBB[0-9_]+]]
; SPLIT-NEXT:  // %bb.{{[0-9]+}}:
; SPLIT-NEXT:    blr x0
; SPLIT:         ret
; SPLIT-NEXT:  [[TRAP]]:
; SPLIT-NEXT:    brk #0x8220
; SPLIT:         .section .text.split.f8,"ax",@progbits
; SPLIT-NEXT:  f8.cold:
; SPLIT:         bl cleanup
; SPLIT-NOT:     brk
; SPLIT:         .size f8.cold
  br i1 %c, label %cold, label %hot, !prof !18
cold:
  call void @cleanup()
  br label %hot
hot:
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

;; The trap has the CFA of the block that branches to it.
define void @f9(ptr noundef %x) uwtable(async) {
; ASM-LABEL: f9:
; ASM:         .cfi_remember_state
; ASM:         cmp w16, w17
; ASM-NEXT:    b.ne [[TRAP:.LBB[0-9_]+]]
; ASM:         .cfi_def_cfa_offset 0
; ASM:         ret
; ASM-NEXT:  [[TRAP]]:
; ASM-NEXT:    .cfi_restore_state
; ASM-NEXT:    brk #0x8220
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

attributes #0 = { returns_twice }
attributes #1 = { "branch-target-enforcement" }
attributes #2 = { "branch-target-enforcement" "branch-protection-pauth-lr" "sign-return-address"="all" "sign-return-address-key"="a_key" }

!llvm.module.flags = !{!0, !1, !2, !3}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 4, !"kcfi-out-of-line-traps", i32 1}
!2 = !{i32 8, !"branch-target-enforcement", i32 1}
!3 = !{i32 1, !"ProfileSummary", !4}
!4 = !{!5, !6, !7, !8, !9, !10, !11, !12}
!5 = !{!"ProfileFormat", !"InstrProf"}
!6 = !{!"TotalCount", i64 10000}
!7 = !{!"MaxCount", i64 10}
!8 = !{!"MaxInternalCount", i64 1}
!9 = !{!"MaxFunctionCount", i64 1000}
!10 = !{!"NumCounts", i64 3}
!11 = !{!"NumFunctions", i64 5}
!12 = !{!"DetailedSummary", !13}
!13 = !{!14, !15, !16}
!14 = !{i32 10000, i64 100, i32 1}
!15 = !{i32 999900, i64 100, i32 1}
!16 = !{i32 999999, i64 1, i32 2}
!17 = !{!"function_entry_count", i64 7000}
!18 = !{!"branch_weights", i32 0, i32 7000}
