# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=haswell -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=HASWELL
# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=broadwell -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=BDWELL
# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=skylake -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=SKYLAKE
# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=znver1 -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=ZNVER1

add     %edi, %esi
bzhil	%esi, (%rdi), %eax


# HASWELL:      Index	012345678
# HASWELL:      [0,0]	DeER .  .	addl	%edi, %esi
# HASWELL-NEXT: [0,1]	DeeeeeeER	bzhil	%esi, (%rdi), %eax

# BDWELL:       Index	012345678
# BDWELL:       [0,0]	DeER .  .	addl	%edi, %esi
# BDWELL-NEXT:  [0,1]	DeeeeeeER	bzhil	%esi, (%rdi), %eax

# SKYLAKE:      Index	012345678
# SKYLAKE:      [0,0]	DeER .  .	addl	%edi, %esi
# SKYLAKE-NEXT: [0,1]	DeeeeeeER	bzhil	%esi, (%rdi), %eax

# ZNVER1:       Index	01234567
# ZNVER1:       [0,0]	DeER . .	addl	%edi, %esi
# ZNVER1-NEXT:  [0,1]	DeeeeeER	bzhil	%esi, (%rdi), %eax
