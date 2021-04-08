; RUN: llc -mtriple=riscv32 -stop-after finalize-isel < %s | FileCheck %s -check-prefix=RV32
; RUN: llc -mtriple=riscv64 -stop-after finalize-isel < %s | FileCheck %s -check-prefix=RV64

; FIXME: The stack location used to pass the parameter to the function has the
; incorrect size and alignment for how we use it, and we clobber the stack.

declare void @callee(<4 x i8> %v)

define void @caller() {
  ; RV32-LABEL: name: caller
  ; RV32: stack:
  ; RV32:     - { id: 0, name: '', type: default, offset: 0, size: 4, alignment: 4,
  ; RV32-NEXT:    stack-id: default, callee-saved-register: '', callee-saved-restored: true,
  ; RV32-NEXT:    debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
  ; RV32: bb.0 (%ir-block.0):
  ; RV32:   ADJCALLSTACKDOWN 0, 0, implicit-def dead $x2, implicit $x2
  ; RV32:   [[ADDI:%[0-9]+]]:gpr = ADDI $x0, 7
  ; RV32:   SW killed [[ADDI]], %stack.0, 12 :: (store 4 into %stack.0)
  ; RV32:   [[ADDI1:%[0-9]+]]:gpr = ADDI $x0, 6
  ; RV32:   SW killed [[ADDI1]], %stack.0, 8 :: (store 4 into %stack.0)
  ; RV32:   [[ADDI2:%[0-9]+]]:gpr = ADDI $x0, 5
  ; RV32:   SW killed [[ADDI2]], %stack.0, 4 :: (store 4 into %stack.0)
  ; RV32:   [[ADDI3:%[0-9]+]]:gpr = ADDI $x0, 4
  ; RV32:   SW killed [[ADDI3]], %stack.0, 0 :: (store 4 into %stack.0)
  ; RV32:   [[ADDI4:%[0-9]+]]:gpr = ADDI %stack.0, 0
  ; RV32:   $x10 = COPY [[ADDI4]]
  ; RV32:   PseudoCALL target-flags(riscv-plt) @callee, csr_ilp32_lp64, implicit-def dead $x1, implicit $x10, implicit-def $x2
  ; RV32:   ADJCALLSTACKUP 0, 0, implicit-def dead $x2, implicit $x2
  ; RV32:   PseudoRET
  ; RV64-LABEL: name: caller
  ; RV64: stack:
  ; RV64:     - { id: 0, name: '', type: default, offset: 0, size: 4, alignment: 4,
  ; RV64-NEXT:    stack-id: default, callee-saved-register: '', callee-saved-restored: true,
  ; RV64-NEXT:    debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
  ; RV64: bb.0 (%ir-block.0):
  ; RV64:   ADJCALLSTACKDOWN 0, 0, implicit-def dead $x2, implicit $x2
  ; RV64:   [[ADDI:%[0-9]+]]:gpr = ADDI $x0, 7
  ; RV64:   SD killed [[ADDI]], %stack.0, 24 :: (store 8 into %stack.0)
  ; RV64:   [[ADDI1:%[0-9]+]]:gpr = ADDI $x0, 6
  ; RV64:   SD killed [[ADDI1]], %stack.0, 16 :: (store 8 into %stack.0)
  ; RV64:   [[ADDI2:%[0-9]+]]:gpr = ADDI $x0, 5
  ; RV64:   SD killed [[ADDI2]], %stack.0, 8 :: (store 8 into %stack.0)
  ; RV64:   [[ADDI3:%[0-9]+]]:gpr = ADDI $x0, 4
  ; RV64:   SD killed [[ADDI3]], %stack.0, 0 :: (store 8 into %stack.0)
  ; RV64:   [[ADDI4:%[0-9]+]]:gpr = ADDI %stack.0, 0
  ; RV64:   $x10 = COPY [[ADDI4]]
  ; RV64:   PseudoCALL target-flags(riscv-plt) @callee, csr_ilp32_lp64, implicit-def dead $x1, implicit $x10, implicit-def $x2
  ; RV64:   ADJCALLSTACKUP 0, 0, implicit-def dead $x2, implicit $x2
  ; RV64:   PseudoRET
  call void @callee(<4 x i8> <i8 4, i8 5, i8 6, i8 7>)
  ret void
}
