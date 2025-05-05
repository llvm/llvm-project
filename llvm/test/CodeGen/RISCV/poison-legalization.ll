; RUN: llc < %s -mtriple=riscv32  | FileCheck %s

define void @SoftenFloat(ptr %p1)  {
; CHECK-LABEL: SoftenFloat:
; CHECK:       # %bb.0:                                # %entry
; CHECK-NEXT:     sw      a0, 4(a0)
; CHECK-NEXT:     sw      a0, 0(a0)
; CHECK-NEXT:     ret

entry:
  store volatile double poison, ptr %p1
  ret void
}

define void @PromoteHalf(ptr %p1 )  {
; CHECK-LABEL: PromoteHalf:
; CHECK:       # %bb.0:                                # %entry
; CHECK-NEXT:     sh      a0, 0(a0)
; CHECK-NEXT:     ret
entry:
   store volatile half poison, ptr %p1
   ret void
}

