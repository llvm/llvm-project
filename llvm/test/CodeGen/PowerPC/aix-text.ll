; RUN: llc -filetype=obj --function-sections -mtriple powerpc-ibm-aix-xcoff -o %t.o < %s
; RUN: llvm-objdump --syms --symbol-description %t.o | FileCheck --check-prefix=CHECKFS32 %s
; RUN: llc -filetype=obj --function-sections -mtriple powerpc64-ibm-aix-xcoff -o %t.o < %s
; RUN: llvm-objdump --syms --symbol-description %t.o | FileCheck --check-prefix=CHECKFS64 %s
; RUN: llc -filetype=obj -mtriple powerpc-ibm-aix-xcoff -o %t.o < %s
; RUN: llvm-objdump --syms --symbol-description %t.o | FileCheck --check-prefix=CHECK32 %s
; RUN: llc -filetype=obj -mtriple powerpc64-ibm-aix-xcoff -o %t.o < %s
; RUN: llvm-objdump --syms --symbol-description %t.o | FileCheck --check-prefix=CHECK64 %s

define i32 @text() {
entry:
  ret i32 1
}

define i32 @text2() {
entry:
  ret i32 2
}

; CHECKFS32: 00000000 l        .text  00000000 (idx: {{[[:digit:]]*}}) [PR]
; CHECKFS32-NEXT: 00000000 g      F .text  {{([[:xdigit:]]{8})}} (idx: {{[[:digit:]]*}}) .text[PR]
; CHECKFS32-NEXT: {{([[:xdigit:]]{8})}} g      F .text  {{([[:xdigit:]]{8})}} (idx: {{[[:digit:]]*}}) .text2[PR]

; CHECKFS64: 0000000000000000 l      .text  0000000000000000
; CHECKFS64-NEXT: 0000000000000000 g      F .text  {{([[:xdigit:]]{16})}} (idx: {{[[:digit:]]*}}) .text[PR]
; CHECKFS64-NEXT: {{([[:xdigit:]]{16})}} g      F .text  {{([[:xdigit:]]{16})}} (idx: {{[[:digit:]]*}}) .text2[PR]

; CHECK32: 00000000 l       .text  {{([[:xdigit:]]{8})}} (idx: {{[[:digit:]]*}}) [PR]
; CHECK32-NEXT: {{([[:xdigit:]]{8})}} g     F .text (csect: (idx: {{[[:digit:]]*}}) [PR])   00000000 (idx: {{[[:digit:]]*}}) .text
; CHECK32-NEXT: {{([[:xdigit:]]{8})}} g     F .text (csect: (idx: {{[[:digit:]]*}}) [PR])   00000000 (idx: {{[[:digit:]]*}}) .text2

; CHECK64: 0000000000000000 l       .text  {{([[:xdigit:]]{16})}} (idx: {{[[:digit:]]*}}) [PR]
; CHECK64-NEXT: {{([[:xdigit:]]{16})}} g     F .text (csect: (idx: {{[[:digit:]]*}}) [PR])   0000000000000000 (idx: {{[[:digit:]]*}}) .text
; CHECK64-NEXT: {{([[:xdigit:]]{8})}} g     F .text (csect: (idx: {{[[:digit:]]*}}) [PR])   0000000000000000 (idx: {{[[:digit:]]*}}) .text2
