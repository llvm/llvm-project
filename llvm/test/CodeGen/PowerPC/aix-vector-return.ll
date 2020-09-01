; RUN: not --crash llc --verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff \
; RUN:   -mattr=+altivec 2>&1 < %s | FileCheck %s

; RUN: not --crash llc --verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff \
; RUN:   -mattr=+altivec 2>&1 < %s | FileCheck %s

; CHECK: LLVM ERROR: UPDATE_VRSAVE is unexpected on AIX.

define dso_local <4 x i32> @test() local_unnamed_addr #0 {
  entry:
    ret <4 x i32> zeroinitializer
}
