; RUN: not --crash llc -mtriple powerpc-ibm-aix-xcoff < %s 2>&1 | FileCheck %s --check-prefix CHECK-ERROR
; RUN: not --crash llc -mtriple powerpc64-ibm-aix-xcoff < %s 2>&1 | FileCheck %s --check-prefix CHECK-ERROR

@a = global [5 x i16] zeroinitializer, align 2 #0

; Function Attrs: noinline
define i16 @foo() #1 {
entry:
  %0 = load i16, ptr @a, align 2
  ret i16 %0
}

attributes #0 = { "toc-data" }
attributes #1 = { noinline }

; CHECK-ERROR: LLVM ERROR: A GlobalVariable with size larger than a TOC entry is not currently supported by the toc data transformation.
