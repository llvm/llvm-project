; RUN: llc -mtriple=hexagon < %s | FileCheck %s

target triple = "hexagon"

; CHECK-LABEL: ua
; CHECK: extractu(r0,#26,#0)
define i32 @ua(i32 %x) local_unnamed_addr #0 {
entry:
  %shl = and i32 %x, 67108863
  ret i32 %shl
}

; CHECK-LABEL: ub
; CHECK: extractu(r0,#16,#4)
define i32 @ub(i32 %x) local_unnamed_addr #0 {
entry:
  %0 = lshr i32 %x, 4
  %shr = and i32 %0, 65535
  ret i32 %shr
}

; CHECK-LABEL: uc
; CHECK: extractu(r0,#24,#0)
define i32 @uc(i32 %x) local_unnamed_addr #0 {
entry:
  %shl = and i32 %x, 16777215
  ret i32 %shl
}

; CHECK-LABEL: ud
; CHECK: extractu(r0,#16,#8)
define i32 @ud(i32 %x) local_unnamed_addr #0 {
entry:
  %bf.lshr = lshr i32 %x, 8
  %bf.clear = and i32 %bf.lshr, 65535
  ret i32 %bf.clear
}

; CHECK-LABEL: sa
; CHECK: extract(r0,#26,#0)
define i32 @sa(i32 %x) local_unnamed_addr #0 {
entry:
  %shl = shl i32 %x, 6
  %shr = ashr exact i32 %shl, 6
  ret i32 %shr
}

; CHECK-LABEL: sb
; CHECK: extract(r0,#16,#4)
define i32 @sb(i32 %x) local_unnamed_addr #0 {
entry:
  %shl = shl i32 %x, 12
  %shr = ashr i32 %shl, 16
  ret i32 %shr
}

; CHECK-LABEL: sc
; CHECK: extract(r0,#24,#0)
define i32 @sc(i32 %x) local_unnamed_addr #0 {
entry:
  %shl = shl i32 %x, 8
  %shr = ashr exact i32 %shl, 8
  ret i32 %shr
}

; CHECK-LABEL: sd
; CHECK: extract(r0,#16,#8)
define i32 @sd(i32 %x) local_unnamed_addr #0 {
entry:
  %bf.shl = shl i32 %x, 8
  %bf.ashr = ashr i32 %bf.shl, 16
  ret i32 %bf.ashr
}

attributes #0 = { noinline norecurse nounwind readnone "target-cpu"="hexagonv60" "target-features"="-hvx,-long-calls" }
