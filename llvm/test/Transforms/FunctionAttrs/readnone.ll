; RUN: opt < %s -functionattrs -S | FileCheck %s
; RUN: opt < %s -passes=function-attrs -S | FileCheck %s

; CHECK: define void @bar(i8* nocapture readnone)
define void @bar(i8* readonly) {
  call void @foo(i8* %0)
    ret void
}

; CHECK: define void @foo(i8* nocapture readnone)
define void @foo(i8* readonly) {
  call void @bar(i8* %0)
  ret void
}
