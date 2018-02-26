; RUN: llc -O0 -mtriple=aarch64-apple-ios -o - %s | FileCheck %s

@g = global i16 0, align 2
declare void @bar(i32)

; Check that only one load is generated. We fall back to
define hidden void @foo() {
; CHECK-NOT: ldrh
; CHECK: ldrsh
  %1 = load volatile i16, i16* @g, align 2
  %2 = sext i16 %1 to i32
  call void @bar(i32 %2)
  ret void
}
