; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN: FileCheck %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN: FileCheck %s

@x = local_unnamed_addr global i32 218114560, align 4 #0

define i32 @main() local_unnamed_addr {
entry:
  %0 = load i32, ptr @x, align 4
  %shr = lshr i32 %0, 8
  %and = and i32 %shr, 255
  ret i32 %and
}

attributes #0 = { "toc-data" }

; CHECK: la [[ADDR:[0-9]+]], x[TD](2)
; CHECK: lbz {{.*}}, 2([[ADDR]])
