; RUN: llc -mtriple=arm64-apple-ios %s -o - | FileCheck %s


declare swifttailcc void @swifttail_callee()
define swifttailcc void @swifttail() {
; CHECK-LABEL: swifttail:
; CHECK-NOT: ld{{.*}}x22
  call void asm "","~{x22}"()
  tail call swifttailcc void @swifttail_callee()
  ret void
}

define swifttailcc void @no_preserve_swiftself() {
; CHECK-LABEL: no_preserve_swiftself:
; CHECK-NOT: ld{{.*}}x20
  call void asm "","~{x20}"()
  ret void
}
