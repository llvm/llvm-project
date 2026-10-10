; RUN: not --crash opt < %s -passes=globalopt -S

; The default globals address space (G1) is different from the allocation's
; address space, so the allocation can't be replaced with a new global.

target datalayout = "G1"

@g = internal global ptr null

define void @init() {
  %m = call noalias ptr @malloc(i64 1)
  store ptr %m, ptr @g
  ret void
}

define i8 @use() {
  %p = load ptr, ptr @g
  %v = load i8, ptr %p
  ret i8 %v
}

declare noalias ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0)
