## Test that llvm-objdump accepts the subtarget spelled either as a legacy
## amdgcn triple plus an explicit --mcpu, or as an amdgpu subarch triple with
## no --mcpu. Both forms must select the same subtarget and disassemble
## identically.

# RUN: llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj -o %t.o %s

## Legacy amdgcn triple + explicit cpu.
# RUN: llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx1250 %t.o \
# RUN:   | FileCheck --check-prefixes=DIS,DIS-AMDGCN %s

## Subarch triple, no cpu: selects the same subtarget and decodes identically.
# RUN: llvm-objdump -d --triple=amdgpu12.50-amd-amdhsa %t.o \
# RUN:   | FileCheck --check-prefixes=DIS,DIS-SUBARCH %s

## No triple, just cpu: the triple is taken from the object.
# RUN: llvm-objdump -d --mcpu=gfx1250 %t.o \
# RUN:   | FileCheck --check-prefixes=DIS,DIS-DETECT %s

## No target arguments at all: both the triple and cpu are taken from the
## object's ELF header.
# RUN: llvm-objdump -d %t.o \
# RUN:   | FileCheck --check-prefixes=DIS,DIS-DETECT %s

# DIS-AMDGCN:  .amdgcn_target "amdgcn-amd-amdhsa{{ ?}}-unknown-gfx1250"
# DIS-SUBARCH: .amdgcn_target "amdgpu12.50-amd-amdhsa-unknown-gfx1250"
# DIS-DETECT:  .amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1250"

# DIS-LABEL: <kernel>:
# DIS: s_set_vgpr_msb 0x41
# DIS-NEXT: s_endpgm

.text
.globl kernel
kernel:
  s_set_vgpr_msb 0x41
  s_endpgm
