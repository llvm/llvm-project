; RUN: llc -mtriple=riscv32 -target-abi=ilp32 -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=RV32
; RUN: llc -mtriple=riscv64 -target-abi=lp64 -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=RV64

; PseudoCALLLpadAlign/PseudoCALLIndirectLpadAlign must carry the same
; call-preserved register mask as PseudoCALL/PseudoCALLIndirect, otherwise
; values live across a returns_twice call are not spilled/reloaded and get
; corrupted when the callee clobbers them.

declare i32 @setjmp(ptr) returns_twice
declare void @use(ptr, ptr)

define void @test_direct_call_preserved_mask(ptr %buf) {
  ; RV32-LABEL: name: test_direct_call_preserved_mask
  ; RV32: PseudoCALLLpadAlign target-flags(riscv-call) @setjmp, 0, csr_ilp32_lp64, implicit-def dead $x1, implicit $x10, implicit-def $x2, implicit-def $x10
  ; RV64-LABEL: name: test_direct_call_preserved_mask
  ; RV64: PseudoCALLLpadAlign target-flags(riscv-call) @setjmp, 0, csr_ilp32_lp64, implicit-def dead $x1, implicit $x10, implicit-def $x2, implicit-def $x10
  %call = call i32 @setjmp(ptr %buf)
  call void @use(ptr %buf, ptr %buf)
  ret void
}

define void @test_indirect_call_preserved_mask(ptr %fptr, ptr %buf) {
  ; RV32-LABEL: name: test_indirect_call_preserved_mask
  ; RV32: PseudoCALLIndirectLpadAlign %{{[0-9]+}}, 0, csr_ilp32_lp64, implicit-def dead $x1, implicit $x10, implicit-def $x2, implicit-def $x10
  ; RV64-LABEL: name: test_indirect_call_preserved_mask
  ; RV64: PseudoCALLIndirectLpadAlign %{{[0-9]+}}, 0, csr_ilp32_lp64, implicit-def dead $x1, implicit $x10, implicit-def $x2, implicit-def $x10
  %call = call i32 %fptr(ptr %buf) #0
  call void @use(ptr %buf, ptr %buf)
  ret void
}

attributes #0 = { returns_twice }

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
