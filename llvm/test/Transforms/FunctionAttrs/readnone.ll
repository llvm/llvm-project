; RUN: opt < %s -function-attrs -S | FileCheck %s
; RUN: opt < %s -passes=function-attrs -S | FileCheck %s

; CHECK: define void @bar(i8* nocapture readnone %0)
define void @bar(i8* readonly %0) {
  call void @foo(i8* %0)
    ret void
}

; CHECK: define void @foo(i8* nocapture readnone %0)
define void @foo(i8* readonly %0) {
  call void @bar(i8* %0)
  ret void
}
