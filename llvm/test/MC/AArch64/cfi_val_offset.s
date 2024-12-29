// RUN: llvm-mc -triple aarch64-- -o - %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-- -filetype=obj -o - %s | llvm-dwarfdump --debug-frame - | FileCheck --check-prefix=DWARF %s

// This test just confirms the .cfi_val_offset directive emits a val_offset()
// rule. It's not testing anything AArch64 specific, it just needs a targets
// registers to be able to use the directive.
example:
// CHECK:      .cfi_startproc
  .cfi_startproc
  add wsp, wsp, 16
  .cfi_def_cfa wsp, -16
// CHECK: .cfi_def_cfa wsp, -16
// DWARF: DW_CFA_advance_loc: 4 to 0x4
// DWARF: DW_CFA_def_cfa: WSP -16
  .cfi_val_offset wsp, 0
// CHECK: .cfi_val_offset wsp, 0
// DWARF: DW_CFA_val_offset: WSP 0
  nop
  sub wsp, wsp, 16
  .cfi_def_cfa wsp, 0
// CHECK: .cfi_def_cfa wsp, 0
// DWARF: DW_CFA_advance_loc: 8 to 0xc
// DWARF: DW_CFA_def_cfa: WSP +0
  .cfi_register wsp, wsp
// CHECK: .cfi_register wsp, wsp
// DWARF: DW_CFA_register: WSP WSP
  ret
  .cfi_endproc
// CHECK: .cfi_endproc


// DWARF: 0x0: CFA=WSP
// DWARF: 0x4: CFA=WSP-16: WSP=CFA
// DWARF: 0xc: CFA=WSP: WSP=WSP
