; RUN: opt -passes=inline,argpromotion < %s -S | FileCheck %s

; Make sure we update the list of libcalls when we replace a libcall.

; CHECK: define {{.*}}@a

define void @a() {
entry:
  %call = call float @strtof(ptr noundef null, ptr noundef null)
  ret void
}

define internal float @strtof(ptr noundef %0, ptr noundef %1) nounwind {
entry:
  ret float 0.0
}

