# RUN: llvm-mc -triple=amdgcn--amdhsa %s | FileCheck --check-prefix=PRINT %s

# RUN: llvm-mc -filetype=obj -triple=amdgcn--amdhsa %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

# PRINT:      .reloc 2, R_AMDGPU_NONE, .data
# PRINT-NEXT: .reloc 1, R_AMDGPU_NONE, foo+4
# PRINT-NEXT: .reloc 0, R_AMDGPU_NONE, 8
# PRINT-NEXT: .reloc 0, R_AMDGPU_ABS32_LO, .data
# PRINT-NEXT: .reloc 0, R_AMDGPU_ABS32_HI, .data
# PRINT-NEXT: .reloc 0, R_AMDGPU_ABS64, .data
# PRINT-NEXT: .reloc 0, R_AMDGPU_REL32, .data
# PRINT-NEXT: .reloc 0, R_AMDGPU_REL64, .data
# PRINT-NEXT: .reloc 0, R_AMDGPU_ABS32, .data
# PRINT-NEXT: .reloc 0, R_AMDGPU_GOTPCREL, .data
# PRINT-NEXT: .reloc 0, R_AMDGPU_GOTPCREL32_LO, .data
# PRINT-NEXT: .reloc 0, R_AMDGPU_GOTPCREL32_HI, .data
# PRINT-NEXT: .reloc 0, R_AMDGPU_REL32_LO, .data
# PRINT-NEXT: .reloc 0, R_AMDGPU_REL32_HI, .data
# PRINT-NEXT: .reloc 0, R_AMDGPU_RELATIVE64, .data
# PRINT-NEXT: .reloc 0, R_AMDGPU_REL16, .data

# CHECK:      0x2 R_AMDGPU_NONE .data
# CHECK-NEXT: 0x1 R_AMDGPU_NONE foo 0x4
# CHECK-NEXT: 0x0 R_AMDGPU_NONE - 0x8
# CHECK-NEXT: 0x0 R_AMDGPU_ABS32_LO .data
# CHECK-NEXT: 0x0 R_AMDGPU_ABS32_HI .data
# CHECK-NEXT: 0x0 R_AMDGPU_ABS64 .data
# CHECK-NEXT: 0x0 R_AMDGPU_REL32 .data
# CHECK-NEXT: 0x0 R_AMDGPU_REL64 .data
# CHECK-NEXT: 0x0 R_AMDGPU_ABS32 .data
# CHECK-NEXT: 0x0 R_AMDGPU_GOTPCREL .data
# CHECK-NEXT: 0x0 R_AMDGPU_GOTPCREL32_LO .data
# CHECK-NEXT: 0x0 R_AMDGPU_GOTPCREL32_HI .data
# CHECK-NEXT: 0x0 R_AMDGPU_REL32_LO .data
# CHECK-NEXT: 0x0 R_AMDGPU_REL32_HI .data
# CHECK-NEXT: 0x0 R_AMDGPU_RELATIVE64 .data
# CHECK-NEXT: 0x0 R_AMDGPU_REL16 .data

.text
  s_nop 0
  s_nop 0
  .reloc 2, R_AMDGPU_NONE, .data
  .reloc 1, R_AMDGPU_NONE, foo+4
  .reloc 0, R_AMDGPU_NONE, 8
  .reloc 0, R_AMDGPU_ABS32_LO, .data
  .reloc 0, R_AMDGPU_ABS32_HI, .data
  .reloc 0, R_AMDGPU_ABS64, .data
  .reloc 0, R_AMDGPU_REL32, .data
  .reloc 0, R_AMDGPU_REL64, .data
  .reloc 0, R_AMDGPU_ABS32, .data
  .reloc 0, R_AMDGPU_GOTPCREL, .data
  .reloc 0, R_AMDGPU_GOTPCREL32_LO, .data
  .reloc 0, R_AMDGPU_GOTPCREL32_HI, .data
  .reloc 0, R_AMDGPU_REL32_LO, .data
  .reloc 0, R_AMDGPU_REL32_HI, .data
  .reloc 0, R_AMDGPU_RELATIVE64, .data
  .reloc 0, R_AMDGPU_REL16, .data

.data
.globl foo
foo:
  .long 0
  .long 0
