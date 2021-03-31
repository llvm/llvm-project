# RUN: llvm-mc %s -arch=mips -mcpu=mips32 | FileCheck %s --check-prefix=ASM

# RUN: llvm-mc %s -arch=mips -mcpu=mips32 -filetype=obj -o - | \
# RUN:   llvm-readobj --symbols - | FileCheck %s --check-prefix=OBJ

  .set micromips

  .global f_mm_insn_data
  .type f_mm_insn_data, @function
f_mm_insn_data:
  .insn
  .word 0x00e73910   # add $7, $7, $7

  .global f_mm_insn_instr
  .type f_mm_insn_instr, @function
f_mm_insn_instr:
  .insn
  add $7, $7, $7

  .global o_mm_insn_data
  .type o_mm_insn_data, @object
o_mm_insn_data:
  .insn
  .word 0x00e73910   # add $7, $7, $7

  .global o_mm_insn_instr
  .type o_mm_insn_instr, @object
o_mm_insn_instr:
  .insn
  add $7, $7, $7

  .set nomicromips

  .global f_normal_insn_data
  .type f_normal_insn_data, @function
f_normal_insn_data:
  .insn
  .word 0x00e73820   # add $7, $7, $7

  .global f_normal_insn_instr
  .type f_normal_insn_instr, @function
f_normal_insn_instr:
  .insn
  add $7, $7, $7

  .global o_normal_insn_data
  .type o_normal_insn_data, @object
o_normal_insn_data:
  .insn
  .word 0x00e73820   # add $7, $7, $7

  .global o_normal_insn_instr
  .type o_normal_insn_instr, @object
o_normal_insn_instr:
  .insn
  add $7, $7, $7

# Verify that .insn causes the currently saved labels to be cleared by checking
# that foo doesn't get marked.
  .set nomicromips
foo:
  .insn
  .word 0x00e73820   # add $7, $7, $7

  .set micromips
bar:
  add $7, $7, $7

# ASM: .insn

# OBJ: Symbols [
# OBJ: Name: foo
# OBJ: Other: 0

# OBJ: Name: f_mm_insn_data
# OBJ: Other [ (0x80)

# OBJ: Name: f_mm_insn_instr
# OBJ: Other [ (0x80)

# OBJ: Name: o_mm_insn_data
# OBJ: Other [ (0x80)

# OBJ: Name: o_mm_insn_instr
# OBJ: Other [ (0x80)

# OBJ: Name: f_normal_insn_data
# OBJ: Other: 0

# OBJ: Name: f_normal_insn_instr
# OBJ: Other: 0

# OBJ: Name: o_normal_insn_data
# OBJ: Other: 0

# OBJ: Name: o_normal_insn_instr
# OBJ: Other: 0
# OBJ: ]
