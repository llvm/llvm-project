; RUN: llc -mtriple=thumbv7-apple-ios -O0 -fast-isel %s -o - | FileCheck %s

declare swifttailcc i8* @SwiftSelf(i8 * swiftasync %context, i8* swiftself %closure)

define swifttailcc i8* @CallSwiftSelf(i8* swiftself %closure, i8* %context) {
; CHECK-LABEL: CallSwiftSelf:
; CHECK: bl _SwiftSelf
; CHECK: pop {r7, pc}
  %res = call swifttailcc i8* @SwiftSelf(i8 * swiftasync %context, i8* swiftself null)
  ret i8* %res
}
