; RUN: llc -filetype=obj -mtriple powerpc-ibm-aix-xcoff -o %t.o < %s
; RUN: llvm-objdump --syms --symbol-description %t.o | FileCheck --check-prefix=CHECK32 %s
; RUN: llc -filetype=obj -mtriple powerpc64-ibm-aix-xcoff -o %t.o < %s
; RUN: llvm-objdump --syms --symbol-description %t.o | FileCheck --check-prefix=CHECK64 %s

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  %call = call i32 @text()
  ret i32 %call
}

declare i32 @text(...)

; CHECK32: 00000000         *UND*  00000000 (idx: {{[[:digit:]]*}}) .text[PR]

; CHECK64: 0000000000000000         *UND*  0000000000000000 (idx: {{[[:digit:]]*}}) .text[PR]
