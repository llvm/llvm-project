# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=riscv64 -mattr=+c,+relax -filetype=obj \
# RUN:     -o %t/elf_riscv64_rvc.o %s
# RUN: llvm-mc -triple=riscv32 -mattr=+c,+relax -filetype=obj \
# RUN:     -o %t/elf_riscv32_rvc.o %s
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1ff00000 -slab-page-size 4096 \
# RUN:     -check %s %t/elf_riscv64_rvc.o
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1ff00000 -slab-page-size 4096 \
# RUN:     -check %s %t/elf_riscv32_rvc.o

.globl main
main:
  ret

# Test R_RISCV_RVC_BRANCH

# jitlink-check: *{2}(test_rvc_branch) = 0xC111
# jitlink-check: *{2}(test_rvc_branch+2) = 0xE109
.globl test_rvc_branch
.type test_rvc_branch,@function
test_rvc_branch:
  c.beqz a0, test_rvc_branch_ret
  c.bnez a0, test_rvc_branch_ret
test_rvc_branch_ret:
  ret
# jitlink-check: *{2}(test_rvc_branch+6) = 0xDD7D
# jitlink-check: *{2}(test_rvc_branch+8) = 0xFD75
test_rvc_branch2:
  c.beqz a0, test_rvc_branch_ret
  c.bnez a0, test_rvc_branch_ret

.size test_rvc_branch, .-test_rvc_branch

# Test R_RISCV_RVC_JUMP

# jitlink-check: *{2}(test_rvc_jump) = 0xA009
.globl test_rvc_jump
.type test_rvc_jump,@function
test_rvc_jump:
  c.j test_rvc_jump_ret
test_rvc_jump_ret:
  ret
# jitlink-check: *{2}(test_rvc_jump+4) = 0xBFFD
test_rvc_jump2:
  c.j test_rvc_jump_ret

.size test_rvc_jump, .-test_rvc_jump
