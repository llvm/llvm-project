; RUN: opt < %s -passes=function-attrs -S | FileCheck %s

; CHECK: define void @bar(ptr nocapture readnone %0)
define void @bar(ptr readonly %0) {
  call void @foo(ptr %0)
    ret void
}

; CHECK: define void @foo(ptr nocapture readnone %0)
define void @foo(ptr readonly %0) {
  call void @bar(ptr %0)
  ret void
}
