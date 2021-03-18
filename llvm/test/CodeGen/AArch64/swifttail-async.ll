; RUN: llc -mtriple=arm64-apple-ios %s -o - | FileCheck %s


declare swifttailcc void @swifttail_callee()
define swifttailcc void @swifttail() {
; CHECK-LABEL: swifttail:
; CHECK-NOT: ld{{.*}}x22
  call void asm "","~{x22}"()
  tail call swifttailcc void @swifttail_callee()
  ret void
}

define void @has_swiftasync(i8* swiftasync %in) {
; CHECK-LABEL: has_swiftasync:
; CHECK: ld{{.*}}x22
  call void asm "","~{x22}"()
  ret void
}

; It's impossible to get a tail call from a function without a swiftasync
; parameter to one with unless the CC is swifttailcc. So it doesn't matter
; whether x22 is callee-saved in this case.
define void @calls_swiftasync() {
; CHECK-LABEL: calls_swiftasync:
; CHECK-NOT: b _has_swiftasync
  call void asm "","~{x22}"()
  tail call void @has_swiftasync(i8* swiftasync null)
  ret void
}

define swifttailcc void @no_preserve_swiftself() {
; CHECK-LABEL: no_preserve_swiftself:
; CHECK-NOT: ld{{.*}}x20
  call void asm "","~{x20}"()
  ret void
}
