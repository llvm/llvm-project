# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=btver2 -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=BTVER2 %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=znver1 -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=ZNVER1 %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=sandybridge -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=SANDYBRIDGE %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=ivybridge -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=IVYBRIDGE %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=haswell -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=HASWELL %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=broadwell -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=BROADWELL %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=knl -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=KNL %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=skylake -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=SKX %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=skylake-avx512 -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=SKX-AVX512 %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=slm -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=SLM %s

add %edi, %eax

# BTVER2:      Iterations:     100
# BTVER2-NEXT: Instructions:   100
# BTVER2-NEXT: Total Cycles:   103
# BTVER2-NEXT: Dispatch Width: 2
# BTVER2-NEXT: IPC:            0.97

# SLM:      Iterations:     100
# SLM-NEXT: Instructions:   100
# SLM-NEXT: Total Cycles:   103
# SLM-NEXT: Dispatch Width: 2
# SLM-NEXT: IPC:            0.97

# BROADWELL:      Iterations:     100
# BROADWELL-NEXT: Instructions:   100
# BROADWELL-NEXT: Total Cycles:   103
# BROADWELL-NEXT: Dispatch Width: 4
# BROADWELL-NEXT: IPC:            0.97

# HASWELL:      Iterations:     100
# HASWELL-NEXT: Instructions:   100
# HASWELL-NEXT: Total Cycles:   103
# HASWELL-NEXT: Dispatch Width: 4
# HASWELL-NEXT: IPC:            0.97

# IVYBRIDGE:      Iterations:     100
# IVYBRIDGE-NEXT: Instructions:   100
# IVYBRIDGE-NEXT: Total Cycles:   103
# IVYBRIDGE-NEXT: Dispatch Width: 4
# IVYBRIDGE-NEXT: IPC:            0.97

# KNL:      Iterations:     100
# KNL-NEXT: Instructions:   100
# KNL-NEXT: Total Cycles:   103
# KNL-NEXT: Dispatch Width: 4
# KNL-NEXT: IPC:            0.97

# SANDYBRIDGE:      Iterations:     100
# SANDYBRIDGE-NEXT: Instructions:   100
# SANDYBRIDGE-NEXT: Total Cycles:   103
# SANDYBRIDGE-NEXT: Dispatch Width: 4
# SANDYBRIDGE-NEXT: IPC:            0.97

# ZNVER1:      Iterations:     100
# ZNVER1-NEXT: Instructions:   100
# ZNVER1-NEXT: Total Cycles:   103
# ZNVER1-NEXT: Dispatch Width: 4
# ZNVER1-NEXT: IPC:            0.97

# SKX:      Iterations:     100
# SKX-NEXT: Instructions:   100
# SKX-NEXT: Total Cycles:   103
# SKX-NEXT: Dispatch Width: 6
# SKX-NEXT: IPC:            0.97

# SKX-AVX512:      Iterations:     100
# SKX-AVX512-NEXT: Instructions:   100
# SKX-AVX512-NEXT: Total Cycles:   103
# SKX-AVX512-NEXT: Dispatch Width: 6
# SKX-AVX512-NEXT: IPC:            0.97

