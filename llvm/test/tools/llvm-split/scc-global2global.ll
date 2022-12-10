; All of the functions and globals in this module must end up
; in the same partition.

; RUN: llvm-split  -j=2 -preserve-locals -o %t %s
; RUN: llvm-dis  -o - %t0 | FileCheck --check-prefix=CHECK1 %s
; RUN: llvm-dis  -o - %t1 | FileCheck --check-prefix=CHECK0 %s

; CHECK0: declare dso_local ptr @local0
; CHECK0: declare dso_local ptr @local1

; CHECK1: @bla
; CHECK1: @ptr
; CHECK1: define internal ptr @local0
; CHECK1: define internal ptr @local1

%struct.anon = type { i64, i64 }

@bla = internal global %struct.anon { i64 1, i64 2 }, align 8
@ptr = internal global ptr @bla, align 4

define internal ptr @local0() {
  ret ptr @bla
}

define internal ptr @local1() {
  ret ptr @ptr
}

