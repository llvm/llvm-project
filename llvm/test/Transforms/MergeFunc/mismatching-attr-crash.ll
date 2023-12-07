; RUN: opt -S -passes=mergefunc %s | FileCheck %s

; CHECK-LABEL: define void @foo
; CHECK: call void %a0
define void @foo(ptr byval(i8) %a0, ptr swiftself %a4) {
entry:
  call void %a0(ptr byval(i8) %a0, ptr swiftself %a4)
  ret void
}

; CHECK-LABEL: define void @bar
; CHECK: call void %a0
define void @bar(ptr byval(i8) %a0, ptr swifterror %a4) {
entry:
  call void %a0(ptr byval(i8) %a0, ptr swifterror %a4)
  ret void
}


