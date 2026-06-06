; RUN: not llc -mtriple=x86_64 %s -o - 2>&1 | FileCheck %s

; Swift attributes should not be used with preserve_none.

declare preserve_nonecc void @foo(ptr swiftself)

; CHECK: error: <unknown>:0:0: in function bar void (ptr): Swift attributes can't be used with preserve_none
define preserve_nonecc void @bar(ptr swifterror) {
  ret void
}

; CHECK: error: <unknown>:0:0: in function qux void (ptr): Swift attributes can't be used with preserve_none
define void @qux(ptr %addr) {
  call preserve_nonecc void @foo(ptr swiftself %addr)
  ret void
}
