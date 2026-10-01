// RUN: llvm-mc -triple=amdgpu12.51-amd-amdhsa -filetype=obj %s -o %t.o
// RUN: llvm-objdump -d --disassemble-symbols=fwd_short,fwd_max,fwd_over,fwd_cond,back_max,back_over,back_cond,ext_cond_global,ext_global,ext_weak,ext_other_section,ext_local %t.o | FileCheck %s
// RUN: llvm-readobj -r %t.o | FileCheck --check-prefix=RELOC %s
// RUN: llvm-mc -triple=amdgpu13.10-amd-amdhsa -mattr=+use-add-pc64-inst -filetype=obj %s -o %t13.o
// RUN: llvm-objdump -d --disassemble-symbols=fwd_short,fwd_max,fwd_over,fwd_cond,back_max,back_over,back_cond,ext_cond_global,ext_global,ext_weak,ext_other_section,ext_local %t13.o | FileCheck %s
// RUN: not llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o /dev/null 2>&1 | FileCheck --check-prefix=ERR %s
// RUN: not llvm-mc -triple=amdgpu12.51-amd-amdhsa -filetype=obj --defsym=UNDEF=1 %s -o /dev/null 2>&1 | FileCheck --check-prefix=UNDEF --implicit-check-not=error: %s

// Test that SOPP branches are relaxed into s_add_pc_i64 with a 32-bit literal
// when they are out of range, on subtargets with FeatureUseAddPC64Inst. The
// literal is sign-extended by the hardware, but the disassembler prints negative
// values zero-extended.

// CHECK-LABEL: <fwd_short>:
// CHECK-NEXT: s_branch
// CHECK-NEXT: s_nop 0
fwd_short:
  s_branch fwd_short_target
  s_nop 0
fwd_short_target:
  s_endpgm

// The maximum forward branch offset is 0x7fff dwords.
// CHECK-LABEL: <fwd_max>:
// CHECK-NEXT: s_branch {{.*}} // {{[0-9A-F]+}}: {{....}}7FFF
fwd_max:
  s_branch fwd_max_target
fwd_max_fill:
  .fill 0x7fff, 4, 0xbf800000
fwd_max_target:
  s_endpgm

// ERR: [[@LINE+5]]:12: error: branch size exceeds simm16
// CHECK-LABEL: <fwd_over>:
// CHECK-NEXT: s_add_pc_i64 0x20000 // {{[0-9A-F]+}}: BE804BFF 00020000
// CHECK-EMPTY:
fwd_over:
  s_branch fwd_over_target
fwd_over_fill:
  .fill 0x8000, 4, 0xbf800000
fwd_over_target:
  s_endpgm

// Conditional branches are relaxed into a short branch with the inverted
// condition that skips over the s_add_pc_i64.
// CHECK-LABEL: <fwd_cond>:
// CHECK-NEXT: s_cbranch_scc1 {{.*}} // {{[0-9A-F]+}}: {{....}}0002
// CHECK-NEXT: s_add_pc_i64 0x2003c
// CHECK-NEXT: s_cbranch_scc0 {{.*}} // {{[0-9A-F]+}}: {{....}}0002
// CHECK-NEXT: s_add_pc_i64 0x20030
// CHECK-NEXT: s_cbranch_vccnz {{.*}} // {{[0-9A-F]+}}: {{....}}0002
// CHECK-NEXT: s_add_pc_i64 0x20024
// CHECK-NEXT: s_cbranch_vccz {{.*}} // {{[0-9A-F]+}}: {{....}}0002
// CHECK-NEXT: s_add_pc_i64 0x20018
// CHECK-NEXT: s_cbranch_execnz {{.*}} // {{[0-9A-F]+}}: {{....}}0002
// CHECK-NEXT: s_add_pc_i64 0x2000c
// CHECK-NEXT: s_cbranch_execz {{.*}} // {{[0-9A-F]+}}: {{....}}0002
// CHECK-NEXT: s_add_pc_i64 0x20000
// CHECK-EMPTY:
fwd_cond:
  s_cbranch_scc0 fwd_cond_target
  s_cbranch_scc1 fwd_cond_target
  s_cbranch_vccz fwd_cond_target
  s_cbranch_vccnz fwd_cond_target
  s_cbranch_execz fwd_cond_target
  s_cbranch_execnz fwd_cond_target
fwd_cond_fill:
  .fill 0x8000, 4, 0xbf800000
fwd_cond_target:
  s_endpgm

// The maximum backward branch offset is -0x8000 dwords.
back_max_target:
  .fill 0x7fff, 4, 0xbf800000
// CHECK-LABEL: <back_max>:
// CHECK-NEXT: s_branch {{.*}} // {{[0-9A-F]+}}: {{....}}8000
back_max:
  s_branch back_max_target

back_over_target:
  .fill 0x8000, 4, 0xbf800000
// ERR: [[@LINE+4]]:12: error: branch size exceeds simm16
// CHECK-LABEL: <back_over>:
// CHECK-NEXT: s_add_pc_i64 0xfffdfff8 // {{[0-9A-F]+}}: BE804BFF FFFDFFF8
back_over:
  s_branch back_over_target

// CHECK-LABEL: <back_cond>:
// CHECK-NEXT: s_cbranch_execz {{.*}} // {{[0-9A-F]+}}: {{....}}0002
// CHECK-NEXT: s_add_pc_i64 0xfffdffec
back_cond:
  s_cbranch_execnz back_over_target

// Branches whose target cannot be resolved at assembly time are always relaxed
// and use a 32-bit PC-relative relocation, even if the target turns out to be
// nearby. This includes symbols in other sections, and global or weak symbols
// in the same section, because ELF keeps relocations for PC-relative references
// to non-local symbols. (Without FeatureUseAddPC64Inst these would use a short
// branch with an R_AMDGPU_REL16 relocation.)

.globl global_sym
global_sym:
  s_endpgm

.weak weak_sym
weak_sym:
  s_endpgm

local_sym:
  s_endpgm

// CHECK-LABEL: <ext_cond_global>:
// CHECK-NEXT: s_cbranch_scc0 {{.*}} // {{[0-9A-F]+}}: {{....}}0002
// CHECK-NEXT: s_add_pc_i64 lit(0x0)
// CHECK-EMPTY:
ext_cond_global:
  s_cbranch_scc1 global_sym

// CHECK-LABEL: <ext_global>:
// CHECK-NEXT: s_add_pc_i64 lit(0x0)
// CHECK-EMPTY:
ext_global:
  s_branch global_sym

// CHECK-LABEL: <ext_weak>:
// CHECK-NEXT: s_add_pc_i64 lit(0x0)
// CHECK-EMPTY:
ext_weak:
  s_branch weak_sym

// CHECK-LABEL: <ext_other_section>:
// CHECK-NEXT: s_add_pc_i64 lit(0x0)
// CHECK-EMPTY:
ext_other_section:
  s_branch other_section_sym

// A branch to a nearby local symbol is resolved at assembly time, so it is not
// relaxed and does not need a relocation.
// CHECK-LABEL: <ext_local>:
// CHECK-NEXT: s_branch local_sym
// CHECK-EMPTY:
ext_local:
  s_branch local_sym

// RELOC:      Section ({{[0-9]+}}) .rela.text {
// RELOC-NEXT:   R_AMDGPU_REL32 global_sym 0xFFFFFFFFFFFFFFFC
// RELOC-NEXT:   R_AMDGPU_REL32 global_sym 0xFFFFFFFFFFFFFFFC
// RELOC-NEXT:   R_AMDGPU_REL32 weak_sym 0xFFFFFFFFFFFFFFFC
// RELOC-NEXT:   R_AMDGPU_REL32 .text.other 0xFFFFFFFFFFFFFFFC
// RELOC-NEXT: }

// Branches to undefined symbols are not relaxed, so that they are still
// diagnosed, because they are most likely typos.
.ifdef UNDEF
// UNDEF: [[@LINE+1]]:12: error: undefined label 'undefined_sym'
  s_branch undefined_sym
// UNDEF: [[@LINE+1]]:18: error: undefined label 'undefined_sym'
  s_cbranch_scc0 undefined_sym
// UNDEF: [[@LINE+1]]:12: error: undefined label '.Lundefined_temp'
  s_branch .Lundefined_temp
.endif

.section .text.other,"ax",@progbits
other_section_sym:
  s_endpgm
