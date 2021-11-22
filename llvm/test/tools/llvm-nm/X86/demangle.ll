; RUN: llc -filetype=obj -mtriple=x86_64-pc-linux -o %t.o %s
; RUN: llvm-nm %t.o | FileCheck --check-prefix="MANGLED" %s
; RUN: llvm-nm %t.o --no-demangle | FileCheck --check-prefix="MANGLED" %s
; RUN: llvm-nm -C %t.o | FileCheck --check-prefix="DEMANGLED" %s
; RUN: llvm-nm --demangle %t.o | FileCheck --check-prefix="DEMANGLED" %s

; RUN: llc -filetype=obj -mtriple=x86_64-apple-darwin9 -o %t.macho %s
; RUN: llvm-nm %t.macho | FileCheck --check-prefix="MACHO-MANGLED" %s
; RUN: llvm-nm -C %t.macho | FileCheck --check-prefix="DEMANGLED" %s

; RUN: llc -filetype=obj -mtriple=x86_64-pc-win32 -o %t.coff %s
; RUN: llvm-nm %t.coff | FileCheck --check-prefix="COFF-MANGLED" %s
; RUN: llvm-nm -C %t.coff | FileCheck --check-prefix="COFF-DEMANGLED" %s

; Show that the last of --no-demangle/--demangle wins:
; RUN: llvm-nm --demangle --no-demangle %t.o | FileCheck --check-prefix="MANGLED" %s
; RUN: llvm-nm --no-demangle --demangle %t.o | FileCheck --check-prefix="DEMANGLED" %s
; RUN: llvm-nm --no-demangle --demangle --no-demangle %t.o | FileCheck --check-prefix="MANGLED" %s
; RUN: llvm-nm --demangle --no-demangle --demangle %t.o | FileCheck --check-prefix="DEMANGLED" %s

define i32 @_Z3fooi(i32) #0 {
entry:
  ret i32 1
}

define float @_Z3barf(float) #0 {
entry:
  ret float 0.000000e+00
}

define i32 @_RNvC1a3baz(i32) #0 {
entry:
  ret i32 1
}

; MANGLED:       0000000000000020 T _RNvC1a3baz
; MANGLED:       0000000000000010 T _Z3barf
; MANGLED:       0000000000000000 T _Z3fooi

; MACHO-MANGLED: 0000000000000020 T __RNvC1a3baz
; MACHO-MANGLED: 0000000000000010 T __Z3barf
; MACHO-MANGLED: 0000000000000000 T __Z3fooi

; COFF-MANGLED:          00000020 T _RNvC1a3baz
; COFF-MANGLED:          00000010 T _Z3barf
; COFF-MANGLED:          00000000 T _Z3fooi

; DEMANGLED:     0000000000000020 T a::baz
; DEMANGLED:     0000000000000010 T bar(float)
; DEMANGLED:     0000000000000000 T foo(int)

; COFF-DEMANGLED:        00000020 T a::baz
; COFF-DEMANGLED:        00000010 T bar(float)
; COFF-DEMANGLED:        00000000 T foo(int)
