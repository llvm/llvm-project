; AIX's support for llvm-mc does not have enough support for directives like .csect
; so we can't use the tool. llvm-jitlink -check is not available as it requries
; implementation of registerXCOFFGraphInfo. Will revisit this testcase once support
; is more complete.

; RUN: mkdir -p %t
; RUN: llc --filetype=obj -o %t/xcoff_ppc64.o %s 
; RUN: llvm-jitlink -noexec -num-threads=0 -triple=powerpc64-ibm-aix %t/xcoff_ppc64.o

target datalayout = "E-m:a-Fi64-i64:64-i128:128-n32:64-S128-v256:256:256-v512:512:512"
target triple = "powerpc64-ibm-aix"

define i32 @main() #0 {
entry:
  ret i32 0
}

attributes #0 = { "target-cpu"="pwr7" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}

