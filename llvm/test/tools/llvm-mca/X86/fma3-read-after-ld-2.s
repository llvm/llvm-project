# RUN: llvm-mca -mtriple=x86_64-unknown-unknown  -mcpu=haswell -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=HASWELL

# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=broadwell -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=BDWELL

# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=skylake -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=SKYLAKE

# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=znver1 -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=ZNVER1

vaddps %xmm0, %xmm0, %xmm2
vfmadd213ps (%rdi), %xmm1, %xmm2

# HASWELL:      [0,0]	DeeeER    . .	vaddps	%xmm0, %xmm0, %xmm2
# HASWELL-NEXT: [0,1]	DeeeeeeeeeeeER	vfmadd213ps	(%rdi), %xmm1, %xmm2

# BDWELL:       [0,0]	DeeeER    . .	vaddps	%xmm0, %xmm0, %xmm2
# BDWELL-NEXT:  [0,1]	DeeeeeeeeeeER	vfmadd213ps	(%rdi), %xmm1, %xmm2

# SKYLAKE:      [0,0]	DeeeeER   . .	vaddps	%xmm0, %xmm0, %xmm2
# SKYLAKE-NEXT: [0,1]	DeeeeeeeeeeER	vfmadd213ps	(%rdi), %xmm1, %xmm2

# ZNVER1:       [0,0]	DeeeER    .   .	  vaddps	%xmm0, %xmm0, %xmm2
# ZNVER1-NEXT:  [0,1]	DeeeeeeeeeeeeER	  vfmadd213ps	(%rdi), %xmm1, %xmm2
