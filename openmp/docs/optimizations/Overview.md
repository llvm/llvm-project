(llvm_openmp_optimizations)=

# OpenMP Optimizations in LLVM

LLVM, since [version 11](https://releases.llvm.org/download.html#11.0.0) (12 Oct
2020), has an [OpenMP-Aware optimization pass](OpenMPOpt.md)
as well as the ability to [perform "scalar optimizations" across OpenMP region
boundaries](OpenMPUnawareOptimizations.md).

:::{toctree}
:glob: true
:hidden: true
:maxdepth: 1
:titlesonly: true

OpenMPOpt
OpenMPUnawareOptimizations
:::
