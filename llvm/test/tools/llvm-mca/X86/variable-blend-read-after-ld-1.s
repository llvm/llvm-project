# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=sandybridge -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=SANDY

# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=ivybridge -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=IVY

# RUN: llvm-mca -mtriple=x86_64-unknown-unknown  -mcpu=haswell -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=HASWELL

# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=broadwell -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=BDWELL

# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=skylake -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=SKYLAKE

# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=BTVER2

# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=znver1 -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=ZNVER1

vaddps %xmm0, %xmm0, %xmm1
vblendvps %xmm1, (%rdi), %xmm2, %xmm3


# SANDY:         [0,0]	DeeeER    .  .	vaddps	  %xmm0, %xmm0, %xmm1
# SANDY-NEXT:    [0,1]	D===eeeeeeeeER	vblendvps %xmm1, (%rdi), %xmm2, %xmm3

# IVY:           [0,0]	DeeeER    .  .	vaddps	  %xmm0, %xmm0, %xmm1
# IVY-NEXT:      [0,1]	D===eeeeeeeeER	vblendvps %xmm1, (%rdi), %xmm2, %xmm3

# HASWELL:       [0,0]	DeeeER    .  .	vaddps	  %xmm0, %xmm0, %xmm1
# HASWELL-NEXT:  [0,1]	D===eeeeeeeeER	vblendvps %xmm1, (%rdi), %xmm2, %xmm3

# BDWELL:        [0,0]	DeeeER    . .	vaddps	  %xmm0, %xmm0, %xmm1
# BDWELL-NEXT:   [0,1]	D===eeeeeeeER	vblendvps %xmm1, (%rdi), %xmm2, %xmm3

# SKYLAKE:       [0,0]	DeeeeER   .   .	vaddps	  %xmm0, %xmm0, %xmm1
# SKYLAKE-NEXT:  [0,1]	D====eeeeeeeeER	vblendvps %xmm1, (%rdi), %xmm2, %xmm3

# BTVER2:        [0,0]	DeeeER    .	vaddps	  %xmm0, %xmm0, %xmm1
# BTVER2-NEXT:   [0,1]	.DeeeeeeeER	vblendvps %xmm1, (%rdi), %xmm2, %xmm3

# ZNVER1:        [0,0]	DeeeER    .	vaddps	  %xmm0, %xmm0, %xmm1
# ZNVER1-NEXT:   [0,1]	DeeeeeeeeER	vblendvps %xmm1, (%rdi), %xmm2, %xmm3
