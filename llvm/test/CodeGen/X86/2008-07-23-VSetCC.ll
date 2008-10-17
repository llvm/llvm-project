; RUN: llvm-as < %s | llc -march=x86 -mcpu=pentium

declare float @fmaxf(float, float)
