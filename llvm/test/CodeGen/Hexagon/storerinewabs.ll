; RUN: llc -mtriple=hexagon -hexagon-small-data-threshold=0 < %s | FileCheck %s

@global = external global i32, align 4

; There was a bug causing ### to be printed. Make sure we print ## instead.
; CHECK-LABEL: foo
; CHECK: memw(##global) =

define void @foo(i32 %x) #0 {
entry:
  %add = add nsw i32 %x, 1
  store i32 %add, ptr @global, align 4
  ret void
}

attributes #0 = { norecurse nounwind }

