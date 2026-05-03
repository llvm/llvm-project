; RUN: opt -passes='module-inline' -inline-priority-mode=cost-benefit -S < %s

define i1 @foo() {
  call ptr @bar(ptr null)
  ret i1 true
}

define ptr @bar(ptr %0) {
  call ptr @bar(ptr null)
  ret ptr null
}
