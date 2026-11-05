; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=inline -disable-output %s

; The inliner analyzes the indirect call with a known target.
; getCallerAllocaCost must not dereference the null callee when
; the alloca args exceed the cutoff.

define void @take(ptr addrspace(5) %p) {
  ret void
}

define void @caller() {
  call void @wrapper(ptr @take)
  ret void
}

define void @wrapper(ptr %fn) {
  %buf = alloca [512 x i8], addrspace(5)
  call void %fn(ptr addrspace(5) %buf)
  ret void
}
