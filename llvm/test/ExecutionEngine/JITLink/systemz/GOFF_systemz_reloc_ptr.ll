; llvm-jitlink -check is not available as it requries implementation of registerXCOFFGraphInfo. 
; Will revisit this testcase once support is more complete.

; RUN: llc --filetype=obj -mtriple=s390x-ibm-zos -o GOFF_systemz_reloc_ptr.o < %s
; RUN: llvm-jitlink -noexec -num-threads=0 --triple=s390x-ibm-zos GOFF_systemz_reloc_ptr.o \
; RUN: -abs CELQSTRT=0x00400000 -abs CELQBST=0x0041000 -abs CELQINPL=0x0042000

define i32 @main() {
entry:
  ret i32 0
}

