# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=btver2  < %s | FileCheck --check-prefix=ALL --check-prefix=BTVER2 %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=znver1  < %s | FileCheck --check-prefix=ALL --check-prefix=ZNVER1 %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=sandybridge  < %s | FileCheck --check-prefix=ALL --check-prefix=SANDYBRIDGE %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=ivybridge  < %s | FileCheck --check-prefix=ALL --check-prefix=IVYBRIDGE %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=haswell  < %s | FileCheck --check-prefix=ALL --check-prefix=HASWELL %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=broadwell  < %s | FileCheck --check-prefix=ALL --check-prefix=BROADWELL %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=knl  < %s | FileCheck --check-prefix=ALL --check-prefix=KNL %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=skylake  < %s | FileCheck --check-prefix=ALL --check-prefix=SKX %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=skylake-avx512  < %s | FileCheck --check-prefix=ALL --check-prefix=SKX-AVX512 %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=slm  < %s | FileCheck --check-prefix=ALL --check-prefix=SLM %s

add %edi, %eax

# ALL: Iterations:     70
# ALL-NEXT: Instructions:   70

# BTVER2: Dispatch Width: 2
# ZNVER1: Dispatch Width: 4
# SANDYBRIDGE: Dispatch Width: 4
# IVYBRIDGE: Dispatch Width: 4
# HASWELL: Dispatch Width: 4
# BROADWELL: Dispatch Width: 4
# KNL: Dispatch Width: 4
# SKX: Dispatch Width: 6
# SKX-AVX512: Dispatch Width: 6
# SLM: Dispatch Width: 2

