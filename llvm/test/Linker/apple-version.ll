; RUN: llvm-link %s %S/Inputs/apple-version/1.ll -S -o - 2>%t.err | FileCheck %s -check-prefix=CHECK1
; RUN: cat %t.err | FileCheck --check-prefix=WARN1 --allow-empty %s
; RUN: llvm-link %s %S/Inputs/apple-version/2.ll -S -o - 2>%t.err | FileCheck %s -check-prefix=CHECK2
; RUN: cat %t.err | FileCheck --check-prefix=WARN2 --allow-empty %s
; RUN: llvm-link %s %S/Inputs/apple-version/3.ll -S -o /dev/null 2>%t.err
; RUN: cat %t.err | FileCheck --check-prefix=WARN3 %s
; RUN: llvm-link %s %S/Inputs/apple-version/4.ll -S -o /dev/null 2>%t.err
; RUN: cat %t.err | FileCheck --check-prefix=WARN4 --allow-empty %s

; Check that the triple that has the larger version number is chosen and no
; warnings are issued when the Triples differ only in version numbers.

; CHECK1: target triple = "x86_64-apple-macosx10.10.0"
; WARN1-NOT: warning
; CHECK2: target triple = "x86_64-apple-macosx10.9.0"
; WARN2-NOT: warning

; i386 and x86_64 map to different ArchType enums.
; WARN3: warning: Linking two modules of different target triples

; x86_64h and x86_64 map to the same ArchType enum.
; WARN4-NOT: warning

target triple = "x86_64-apple-macosx10.9.0"
