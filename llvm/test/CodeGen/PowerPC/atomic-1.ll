; RUN: llc < %s -march=ppc32 |  FileCheck %s

define i32 @exchange_and_add(i32* %mem, i32 %val) nounwind {
; CHECK: exchange_and_add:
; CHECK: lwarx
  %tmp = atomicrmw add i32* %mem, i32 %val monotonic
; CHECK: stwcx.
  ret i32 %tmp
}

define i32 @exchange_and_cmp(i32* %mem) nounwind {
; CHECK: exchange_and_cmp:
; CHECK: lwarx
  %tmp = cmpxchg i32* %mem, i32 0, i32 1 monotonic
; CHECK: stwcx.
; CHECK: stwcx.
  ret i32 %tmp
}

define i32 @exchange(i32* %mem, i32 %val) nounwind {
; CHECK: exchange:
; CHECK: lwarx
  %tmp = atomicrmw xchg i32* %mem, i32 1 monotonic
; CHECK: stwcx.
  ret i32 %tmp
}
