; RUN: opt -passes=attributor -attributor-assume-closed-world -S < %s | FileCheck %s

; The call to @dispatcher goes through an addrspacecast, so the call site is a
; user of the constant expression rather than of @dispatcher, and matching on
; getCalledFunction() would leave its norecurse in place after specialization
; drops the attribute from @dispatcher itself.

@g = external global i32

define internal void @inner() norecurse {
  store i32 1, ptr @g
  ret void
}

define internal void @outer() norecurse {
  call void @dispatcher(ptr @inner)
  ret void
}

define internal void @dispatcher(ptr %fn) norecurse {
  call void %fn()
  ret void
}

define void @entry() norecurse {
  call addrspace(1) void addrspacecast (ptr @dispatcher to ptr addrspace(1))(ptr @outer) norecurse
  ret void
}

; CHECK: define internal void @dispatcher(ptr {{.*}}) #[[ATTR:[0-9]+]] {
; CHECK: call addrspace(1) void addrspacecast (ptr @dispatcher to ptr addrspace(1))(ptr @outer){{$}}
; CHECK: attributes #[[ATTR]] = { mustprogress nofree nosync nounwind willreturn memory(write) }
