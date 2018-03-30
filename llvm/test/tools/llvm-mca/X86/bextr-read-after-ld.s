# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=haswell -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=HASWELL
# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=broadwell -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=BDWELL
# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=skylake -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=SKYLAKE
# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=BTVER2
# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=znver1 -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s -check-prefix=ZNVER1

add     %edi, %esi
bextrl	%esi, (%rdi), %eax


# HASWELL:      Index	0123456789
# HASWELL:      [0,0]	DeER .   .	addl	%edi, %esi
# HASWELL-NEXT: [0,1]	DeeeeeeeER	bextrl	%esi, (%rdi), %eax

# BDWELL:       Index	0123456789
# BDWELL:       [0,0]	DeER .   .	addl	%edi, %esi
# BDWELL-NEXT:  [0,1]	DeeeeeeeER	bextrl	%esi, (%rdi), %eax

# SKYLAKE:      Index	0123456789
# SKYLAKE:      [0,0]	DeER .   .	addl	%edi, %esi
# SKYLAKE-NEXT: [0,1]	DeeeeeeeER	bextrl	%esi, (%rdi), %eax

# BTVER2:       Index	0123456
# BTVER2:       [0,0]	DeER ..	        addl	%edi, %esi
# BTVER2-NEXT:  [0,1]	DeeeeER	        bextrl	%esi, (%rdi), %eax

# ZNVER1:       Index	01234567
# ZNVER1:       [0,0]	DeER . .	addl	%edi, %esi
# ZNVER1-NEXT:  [0,1]	DeeeeeER	bextrl	%esi, (%rdi), %eax
