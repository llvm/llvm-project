; llvm-jitlink -check is not available as it requries implementation of registerXCOFFGraphInfo. 
; Will revisit this testcase once support is more complete.

; RUN: llc --filetype=obj -mtriple=s390x-ibm-zos -o GOFF_systemz_reloc_ptr.o < %s
; RUN: llvm-jitlink -noexec -num-threads=0 --triple=s390x-ibm-zos GOFF_systemz_reloc_ptr.o \
; RUN: -abs CELQSTRT=0x00400000

;target datalayout = "E-m:z-p:64:64-i1:8:16-i8:8:16-i16:16-i32:32-i64:64-f32:32-f64:64-f128:64-a:8:16-n32:64"
;target triple = "s390x-ibm-zos"

define i32 @main() {
entry:
  ret i32 0
}

