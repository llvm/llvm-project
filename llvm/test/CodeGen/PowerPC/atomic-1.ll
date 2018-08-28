; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu |  FileCheck %s

define i32 @exchange_and_add(i32* %mem, i32 %val) nounwind {
; CHECK-LABEL: exchange_and_add:
; CHECK: lwarx {{[0-9]+}}, 0, {{[0-9]+}}
  %tmp = atomicrmw add i32* %mem, i32 %val monotonic
; CHECK: stwcx. {{[0-9]+}}, 0, {{[0-9]+}}
  ret i32 %tmp
}

define i32 @exchange_and_cmp(i32* %mem) nounwind {
; CHECK-LABEL: exchange_and_cmp:
; CHECK: lwarx
  %tmppair = cmpxchg i32* %mem, i32 0, i32 1 monotonic monotonic
  %tmp = extractvalue { i32, i1 } %tmppair, 0
; CHECK: stwcx.
; CHECK: stwcx.
  ret i32 %tmp
}

define i32 @exchange(i32* %mem, i32 %val) nounwind {
; CHECK-LABEL: exchange:
; CHECK: lwarx
  %tmp = atomicrmw xchg i32* %mem, i32 1 monotonic
; CHECK: stwcx.
  ret i32 %tmp
}
