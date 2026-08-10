; RUN: llc -mtriple=thumbv7-apple-ios -O0 -fast-isel %s -o - | FileCheck %s

declare swifttailcc ptr @SwiftSelf(ptr swiftasync %context, ptr swiftself %closure)

define swifttailcc ptr @CallSwiftSelf(ptr swiftself %closure, ptr %context) {
; CHECK-LABEL: CallSwiftSelf:
; CHECK: bl _SwiftSelf
; CHECK: pop {r7, pc}
  %res = call swifttailcc ptr @SwiftSelf(ptr swiftasync %context, ptr swiftself null)
  ret ptr %res
}
