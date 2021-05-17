; RUN: llc -mtriple=i386-apple-darwin %s -o - | FileCheck %s

declare void @clobber()

declare swifttailcc void @swifttail_callee()
define swifttailcc void @swifttail() {
; CHECK-LABEL: swifttail:
; CHECK-NOT: %rbx
  call void @clobber()
  tail call swifttailcc void @swifttail_callee()
  ret void
}

declare swifttailcc void @swiftself(i8* swiftself)

define swifttailcc void @swifttail2(i8* %arg) {
; CHECK-LABEL: swifttail2:
; CHECK: movl {{.*}}, %ecx
; CHECK: jmp _swiftself
  tail call swifttailcc void @swiftself(i8* swiftself %arg)
  ret void
}
