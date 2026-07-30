; runtime library implementations should be added to llvm.compiler.used
; RUN: llvm-as <%s >%t1
; RUN: llvm-lto -o %t2 %t1
; RUN: llvm-nm --no-sort %t2 | FileCheck %s -check-prefix=KEEP -check-prefix=LOSE

target triple = "x86_64-apple-darwin9"

; KEEP-LABEL: _puts
define void @puts() {
  ret void
}

; KEEP-LABEL: ___divti3
define void @__divti3() {
  ret void
}

; KEEP-LABEL: _memset
define void @memset() {
  ret void
}

; LOSE-NOT: _myprintf
define void @myprintf() {
  ret void
}

