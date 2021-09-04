; REQUIRES: asserts
; RUN: not --crash llc -verify-machineinstrs -mtriple=powerpc64le-linux-gnu < %s \
; RUN:   2>&1 | FileCheck --check-prefix=CHECK-LE %s
; RUN: not --crash llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix-xcoff < %s \
; RUN:   2>&1 | FileCheck --check-prefix=CHECK-BE %s

declare void @bar(i8*)

define void @foo(i8 %x) {
; CHECK-LE: Unhandled stack size
; CHECK-BE: Unhandled stack size
entry:
  %a = alloca i8, i64 4294967296, align 16
  %b = getelementptr i8, i8* %a, i64 0
  store volatile i8 %x, i8* %b
  ret void
}
