; RUN: opt -S -mergefunc %s | FileCheck %s

; CHECK-LABEL: define void @foo
; CHECK: call void %bc
define void @foo(i8* byval %a0, i8* swiftself %a4) {
entry:
  %bc = bitcast i8* %a0 to void (i8*, i8*)*
  call void %bc(i8* byval %a0, i8* swiftself %a4)
  ret void
}

; CHECK-LABEL: define void @bar
; CHECK: call void %bc
define void @bar(i8* byval(i8) %a0, i8** swifterror %a4) {
entry:
  %bc = bitcast i8* %a0 to void (i8*, i8**)*
  call void %bc(i8* byval(i8) %a0, i8** swifterror %a4)
  ret void
}


