; RUN: llc -mtriple=aarch64-- -verify-machineinstrs < %s | FileCheck %s --check-prefixes=ASM,OUTLINED
; RUN: llc -mtriple=aarch64-- -mcpu=cortex-a53 -enable-post-misched -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=aarch64-- -verify-machineinstrs -stop-after=kcfi-split < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=aarch64-- -verify-machineinstrs -verify-cfiinstrs -function-splitting=all < %s | FileCheck %s --check-prefix=SPLIT
; RUN: llc -mtriple=aarch64-- -verify-machineinstrs -verify-cfiinstrs -basic-block-sections=all < %s | FileCheck %s --check-prefix=BBS

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
; MIR-NOT:     liveins:
; MIR:         BRK 33312
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
; MIR-NOT:     liveins:
; MIR:         BRK 33312
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
; MIR:         KCFI_CMP killed $x16, 12345678, implicit-def $x9, implicit-def $x16, implicit-def $x17, implicit-def $nzcv
; MIR-NEXT:    Bcc 1, %bb.1, implicit $nzcv
; MIR:       bb.2 (%ir-block.0):
; MIR-NEXT:    liveins: {{.*}}$x16
; MIR:         TCRETURNrix16x17 killed $x16, 0, csr_aarch64_aapcs, implicit $sp
; MIR:       bb.1 (%ir-block.0):
; MIR-NOT:     liveins:
; MIR:         BRK 33328

;; With basic block sections enabled on AArch64, the call stays paired in the
;; entry section while BranchRelaxation routes the split trap section without
;; spilling x16.
; BBS-LABEL: f3:
; BBS-NOT:     {{str|stp|ldr|ldp}} {{.*}}x16
; BBS:         ldur w9, [x16, #-4]
; BBS:         cmp w9, w17
; BBS-NEXT:    b.eq [[BBS_CALL:\.LBB[0-9_]+]]
; BBS:         b [[BBS_TRAP:f3\.__part\.[0-9]+]]
; BBS:       [[BBS_CALL]]:
; BBS-NEXT:    br x16
; BBS:       [[BBS_TRAP]]:
; BBS:         brk #0x8230
; BBS:         .size f3,
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
; MIR:         KCFI_CMP killed $x17, 12345678, implicit-def $x9, implicit-def $x16, implicit-def $x17, implicit-def $nzcv
; MIR-NEXT:    Bcc 1, %bb.1, implicit $nzcv
; MIR:       bb.2 (%ir-block.0):
; MIR-NEXT:    liveins: {{.*}}$x17
; MIR:         TCRETURNrix17 killed $x17, 0, csr_aarch64_aapcs, implicit $sp
; MIR:       bb.1 (%ir-block.0):
; MIR-NOT:     liveins:
; MIR:         BRK 33073
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
; MIR:       bb.3 (%ir-block.0):
; MIR-NOT:     liveins:
; MIR:         BRK 33312
; MIR:       bb.1 (%ir-block.0):
; MIR-NOT:     liveins:
; MIR:         BRK 33331
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

;; A non-invoke call preceding another call/invoke in a block with an EH pad
;; successor keeps an inline bundled check.
define void @f7(ptr noundef %x) personality ptr @__gxx_personality_v0 {
; ASM-LABEL: f7:
; ASM:         cmp w16, w17
; ASM-NEXT:    b.eq [[PASS:\.Ltmp[0-9]+]]
; ASM-NEXT:    brk #0x8220
; ASM-NEXT:  [[PASS]]:
; ASM-NEXT:    blr x0
; ASM-NEXT:  .Ltmp{{[0-9]+}}:
; ASM-NEXT:    bl cleanup

; MIR-LABEL: name: f7
; MIR:       BUNDLE{{.*}} {
; MIR-NEXT:    KCFI_CHECK $x0, 12345678, implicit-def $x9, implicit-def $x16, implicit-def $x17, implicit-def $nzcv
; MIR-NEXT:    BLR killed $x0, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit-def $sp
; MIR-NEXT:  }
; MIR-NEXT:  EH_LABEL
; MIR-NEXT:  BL @cleanup
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

;; With machine function splitting, a hot check's trap moves to .text.split
;; while BranchRelaxation keeps passing checks falling through cleanly.
define void @f8(ptr noundef %x, i1 %c) !prof !17 {
; SPLIT-LABEL: f8:
; SPLIT:         cmp w16, w17
; SPLIT-NEXT:    b.ne [[TRAMP:.LBB[0-9_]+]]
; SPLIT-NEXT:  // %bb.{{[0-9]+}}:
; SPLIT-NEXT:    blr x0
; SPLIT:         ret
; SPLIT:       [[TRAMP]]:
; SPLIT:         b [[TRAP:\.LBB[0-9_]+]]
; SPLIT:         .section .text.split.f8,"ax",@progbits
; SPLIT-NEXT:  f8.cold:
; SPLIT:         bl cleanup
; SPLIT:       [[TRAP]]:
; SPLIT-NEXT:    brk #0x8220
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

;; When the block containing the check is cold, BasicBlockSections moves the
;; bundled check to .text.split first, and KCFISplit then keeps KCFI_CMP, its
;; fallthrough call block, and its trap together inside .text.split.
define void @f10(ptr noundef %x, i1 %c1, i1 %c2) !prof !17 {
; SPLIT-LABEL: f10:
; SPLIT-NOT:     blr x0
; SPLIT:         .section .text.split.f10,"ax",@progbits
; SPLIT-NEXT:  f10.cold:
; SPLIT:         cmp w16, w17
; SPLIT-NEXT:    b.ne [[TRAP:\.LBB[0-9_]+]]
; SPLIT-NEXT:  // %bb.{{[0-9]+}}:
; SPLIT-NEXT:    blr x0
; SPLIT:       [[TRAP]]:
; SPLIT-NEXT:    brk #0x8220
; SPLIT:         .size f10.cold
entry:
  br i1 %c1, label %cold, label %check2, !prof !18
check2:
  br i1 %c2, label %cold2, label %hot, !prof !18
cold:
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
cold2:
  call void @cleanup()
  ret void
hot:
  ret void
}

;; When BranchFolder tail-merges post-call cleanup across cold checks with
;; different type hashes, KCFI bundles prevent inverting the checks, and
;; KCFISplit keeps both checks, calls, and traps out of line in .text.split.
define void @f10_tailmerge(ptr noundef %x, i1 %c1, i1 %c2, i1 %c3) !prof !17 {
; SPLIT-LABEL: f10_tailmerge:
; SPLIT-NOT:     blr x0
; SPLIT:         .section .text.split.f10_tailmerge,"ax",@progbits
; SPLIT-NEXT:  f10_tailmerge.cold:
; SPLIT:         cmp w16, w17
; SPLIT-NEXT:    b.ne [[TRAP1:\.LBB[0-9_]+]]
; SPLIT-NEXT:  // %bb.{{[0-9]+}}:
; SPLIT-NEXT:    blr x0
; SPLIT:         cmp w16, w17
; SPLIT-NEXT:    b.ne [[TRAP2:\.LBB[0-9_]+]]
; SPLIT-NEXT:  // %bb.{{[0-9]+}}:
; SPLIT-NEXT:    blr x0
; SPLIT:         bl cleanup
; SPLIT:       [[TRAP1]]:
; SPLIT-NEXT:    brk #0x8220
; SPLIT:       [[TRAP2]]:
; SPLIT-NEXT:    brk #0x8220
; SPLIT:         .size f10_tailmerge.cold
entry:
  br i1 %c1, label %cold1, label %check2, !prof !18
check2:
  br i1 %c2, label %cold2, label %check3, !prof !18
check3:
  br i1 %c3, label %cold3, label %hot, !prof !18
cold1:
  call void %x() [ "kcfi"(i32 12345678) ]
  call void @cleanup()
  ret void
cold2:
  call void %x() [ "kcfi"(i32 87654321) ]
  call void @cleanup()
  ret void
cold3:
  call void @cleanup()
  call void @cleanup()
  ret void
hot:
  ret void
}

;; With shrink-wrapping and machine function splitting, a cold prologue block
;; and cold trap blocks both move to .text.split with valid CFI state.
define void @f10_shrinkwrap(ptr noundef %x, i1 %c) uwtable(async) !prof !17 {
; SPLIT-LABEL: f10_shrinkwrap:
; SPLIT:         cmp w16, w17
; SPLIT-NEXT:    b.ne [[TRAMP:\.LBB[0-9_]+]]
; SPLIT:         blr x0
; SPLIT:       [[TRAMP]]:
; SPLIT:         b [[TRAP:\.LBB[0-9_]+]]
; SPLIT:         .section .text.split.f10_shrinkwrap,"ax",@progbits
; SPLIT-NEXT:  f10_shrinkwrap.cold:
; SPLIT:         bl cleanup
; SPLIT:       [[TRAP]]:
; SPLIT-NEXT:    .cfi_restore_state
; SPLIT-NEXT:    brk #0x8220
; SPLIT:         .size f10_shrinkwrap.cold
entry:
  br i1 %c, label %cold, label %hot, !prof !18
cold:
  call void @cleanup()
  ret void
hot:
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

define void @f10_shrinkwrap_tail(ptr noundef %x, i1 %c) uwtable(async) !prof !17 {
; SPLIT-LABEL: f10_shrinkwrap_tail:
; SPLIT:         cmp w16, w17
; SPLIT-NEXT:    b.ne [[TRAMP:\.LBB[0-9_]+]]
; SPLIT:         br x0
; SPLIT:       [[TRAMP]]:
; SPLIT:         b [[TRAP:\.LBB[0-9_]+]]
; SPLIT:         .section .text.split.f10_shrinkwrap_tail,"ax",@progbits
; SPLIT-NEXT:  f10_shrinkwrap_tail.cold:
; SPLIT:         bl cleanup
; SPLIT:       [[TRAP]]:
; SPLIT-NEXT:    brk #0x8220
; SPLIT:         .size f10_shrinkwrap_tail.cold
entry:
  br i1 %c, label %cold, label %hot, !prof !18
cold:
  call void @cleanup()
  ret void
hot:
  tail call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

;; KCFI checks and traps are never outlined by the machine outliner (including
;; bundled tail calls), preserving patchable-function-prefix offsets,
;; X16/X17/NZCV liveness, and paired check/call blocks.
define void @f11(ptr noundef %x, ptr %p) minsize "patchable-function-prefix"="2" {
; OUTLINED-LABEL: f11:
; OUTLINED:         bl OUTLINED_FUNCTION_[[FN:[0-9]+]]
; OUTLINED-NEXT:    ldur w16, [x0, #-12]
; OUTLINED:         b.ne [[TRAP:.LBB[0-9_]+]]
; OUTLINED:       [[TRAP]]:
; OUTLINED-NEXT:    brk #0x8220
  store i32 1, ptr %p, align 4
  %p1 = getelementptr i32, ptr %p, i64 1
  store i32 2, ptr %p1, align 4
  %p2 = getelementptr i32, ptr %p, i64 2
  store i32 3, ptr %p2, align 4
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

define void @f12(ptr noundef %x, ptr %p) minsize "patchable-function-prefix"="2" {
; OUTLINED-LABEL: f12:
; OUTLINED:         bl OUTLINED_FUNCTION_[[FN]]
; OUTLINED-NEXT:    ldur w16, [x0, #-12]
; OUTLINED:         b.ne [[TRAP:.LBB[0-9_]+]]
; OUTLINED:       [[TRAP]]:
; OUTLINED-NEXT:    brk #0x8220
  store i32 1, ptr %p, align 4
  %p1 = getelementptr i32, ptr %p, i64 1
  store i32 2, ptr %p1, align 4
  %p2 = getelementptr i32, ptr %p, i64 2
  store i32 3, ptr %p2, align 4
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

define void @f12_tail1(ptr noundef %x, ptr %p) minsize "patchable-function-prefix"="2" {
; OUTLINED-LABEL: f12_tail1:
; OUTLINED:         ldur w16, [{{x[0-9]+}}, #-12]
; OUTLINED:         b.ne [[TRAP:.LBB[0-9_]+]]
; OUTLINED:         br {{x[0-9]+}}
; OUTLINED:       [[TRAP]]:
; OUTLINED-NEXT:    brk #0x8220
  call void @cleanup()
  store i32 1, ptr %p, align 4
  %p1 = getelementptr i32, ptr %p, i64 1
  store i32 2, ptr %p1, align 4
  %p2 = getelementptr i32, ptr %p, i64 2
  store i32 3, ptr %p2, align 4
  tail call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

define void @f12_tail2(ptr noundef %x, ptr %p) minsize "patchable-function-prefix"="2" {
; OUTLINED-LABEL: f12_tail2:
; OUTLINED:         ldur w16, [{{x[0-9]+}}, #-12]
; OUTLINED:         b.ne [[TRAP:.LBB[0-9_]+]]
; OUTLINED:         br {{x[0-9]+}}
; OUTLINED:       [[TRAP]]:
; OUTLINED-NEXT:    brk #0x8220
  call void @cleanup()
  store i32 1, ptr %p, align 4
  %p1 = getelementptr i32, ptr %p, i64 1
  store i32 2, ptr %p1, align 4
  %p2 = getelementptr i32, ptr %p, i64 2
  store i32 3, ptr %p2, align 4
  tail call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

;; Functions with callee-pop stack arguments and PAC-RET move KCFI traps out of
;; line once epilogue/tail-call lowering finishes before KCFISplit.
define tailcc void @f13(ptr noundef %x, i64, i64, i64, i64, i64, i64, i64, i64) #2 {
; ASM-LABEL: f13:
; ASM:         hint #29
; ASM-NEXT:    ldur w16, [x17, #-4]
; ASM:         cmp w16, w9
; ASM-NEXT:    b.ne [[TRAP:\.LBB[0-9_]+]]
; ASM-NEXT:  // %bb.{{[0-9]+}}:
; ASM-NEXT:    br x17
; ASM-NEXT:  [[TRAP]]:
; ASM-NEXT:    brk #0x8131

; MIR-LABEL: name: f13
; MIR:         KCFI_CMP killed $x17, 12345678, implicit-def $x9, implicit-def $x16, implicit-def $x17, implicit-def $nzcv
; MIR-NEXT:    Bcc 1, %[[TRAP:bb\.[0-9]+]], implicit $nzcv
; MIR:       bb.{{[0-9]+}} (%ir-block.{{[0-9]+}}):
; MIR-NEXT:    liveins: {{.*}}$x17
; MIR:         TCRETURNrix17 killed $x17, 0, csr_aarch64_aapcs, implicit $sp
; MIR:       [[TRAP]] (%ir-block.{{[0-9]+}}):
; MIR-NOT:     liveins:
; MIR:         BRK 33073
  call void @cleanup()
  tail call tailcc void %x(ptr %x, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0) [ "kcfi"(i32 12345678) ]
  ret void
}

;; Functions with Speculative Load Hardening fall back to inline KCFI checks so
;; out-of-line branches do not interfere with X16 taint tracking or degrade
;; hardening to full barriers.
define void @f14(ptr noundef %x) speculative_load_hardening {
; ASM-LABEL: f14:
; ASM-NOT:     dsb sy
; ASM:         cmp w16, w17
; ASM-NEXT:    b.eq [[PASS:.Ltmp[0-9]+]]
; ASM-NEXT:    brk #0x8220
; ASM-NEXT:  [[PASS]]:
; ASM-NEXT:    blr x0
; ASM-NOT:     dsb sy
; ASM:         ret
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

;; Functions with XRay instrumentation move KCFI traps out of line once XRay
;; inserts PATCHABLE_TAIL_CALL before KCFISplit.
define void @f15(ptr noundef %x, ptr noundef %y) "function-instrument"="xray-always" {
; ASM-LABEL: f15:
; ASM:         cmp w16, w17
; ASM-NEXT:    b.ne [[TRAP1:\.LBB[0-9_]+]]
; ASM-NEXT:  // %bb.{{[0-9]+}}:
; ASM-NEXT:    blr x0
; ASM:       .Lxray_sled_{{[0-9]+}}:
; ASM:         ldur w16, [x0, #-4]
; ASM:         cmp w16, w17
; ASM-NEXT:    b.ne [[TRAP2:\.LBB[0-9_]+]]
; ASM-NEXT:  // %bb.{{[0-9]+}}:
; ASM-NEXT:    br x0
; ASM:       [[TRAP1]]:
; ASM-NEXT:    brk #0x8220
; ASM:       [[TRAP2]]:
; ASM-NEXT:    brk #0x8220
  call void %x() [ "kcfi"(i32 12345678) ]
  tail call void %y() [ "kcfi"(i32 12345678) ]
  ret void
}

;; Indirect invokes move their KCFI traps out of line on AArch64 while keeping
;; their BeginLabel/EndLabel EH_LABEL pair around the call.
define void @f16(ptr noundef %x) personality ptr @__gxx_personality_v0 {
; ASM-LABEL: f16:
; ASM:         ldur w16, [x0, #-4]
; ASM:         cmp w16, w17
; ASM-NEXT:    b.ne [[TRAP:\.LBB[0-9_]+]]
; ASM-NEXT:  // %bb.{{[0-9]+}}:
; ASM-NEXT:  .Ltmp{{[0-9]+}}: // EH_LABEL
; ASM-NEXT:    blr x0
; ASM-NEXT:  .Ltmp{{[0-9]+}}: // EH_LABEL
; ASM:       [[TRAP]]:
; ASM-NEXT:    brk #0x8220
  invoke void %x() [ "kcfi"(i32 12345678) ]
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

; OUTLINED-LABEL: OUTLINED_FUNCTION_0:
; OUTLINED-NOT:     ldur
; OUTLINED-NOT:     brk
; OUTLINED:         ret
; OUTLINED-LABEL: OUTLINED_FUNCTION_1:
; OUTLINED-NOT:     ldur
; OUTLINED-NOT:     brk
; OUTLINED:         ret

attributes #0 = { returns_twice }
attributes #1 = { "branch-target-enforcement" }
attributes #2 = { "branch-target-enforcement" "branch-protection-pauth-lr" "sign-return-address"="all" "sign-return-address-key"="a_key" }

!llvm.module.flags = !{!0, !1, !2, !3}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 7, !"kcfi-out-of-line-traps", i32 1}
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
