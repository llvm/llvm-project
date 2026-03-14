// RUN: llvm-mc -triple=amdgcn -mcpu=fiji -filetype=obj %s | llvm-objcopy -S -K keep_symbol - | llvm-objdump -d --mcpu=fiji - | FileCheck %s --check-prefix=BIN

// FIXME: Immediate operands to sopp_br instructions are currently scaled by a
// factor of 4, are unsigned, are always PC relative, don't accept most
// expressions, and are not range checked.

loop_start_nosym:
s_branch loop_start_nosym
// BIN-NOT: loop_start_nosym:
// BIN: s_branch 65535 // 000000000000: BF82FFFF <.text>

s_branch loop_end_nosym
// BIN: s_branch 0 // 000000000004: BF820000 <.text+0x8>
// BIN-NOT: loop_end_nosym:
loop_end_nosym:
  s_nop 0

keep_symbol:
  s_nop 0

loop_start_sym:
s_branch loop_start_sym
// BIN-NOT: loop_start_sym:
// BIN: s_branch 65535 // 000000000010: BF82FFFF <keep_symbol+0x4>

s_branch loop_end_sym
// BIN: s_branch 0 // 000000000014: BF820000 <keep_symbol+0xc>
// BIN-NOT: loop_end_sym:
loop_end_sym:
  s_nop 0

s_branch 65535
// BIN: s_branch 65535 // 00000000001C: BF82FFFF <keep_symbol+0x10>

s_branch 32768
// BIN: s_branch 32768 // 000000000020: BF828000 <keep_symbol+0xfffffffffffe0018>

s_branch 32767
// BIN: s_branch 32767 // 000000000024: BF827FFF <keep_symbol+0x20018>
