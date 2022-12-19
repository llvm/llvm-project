; RUN: llvm-as  < %s | llvm-dis  | llvm-as  | llvm-dis  | FileCheck %s

; CHECK: define void @foo(ptr sret(i32) align 4 %0)
define void @foo(ptr sret(i32) align 4 %0) {
  ret void
}

; CHECK: define void @bar(ptr sret({ ptr, i8 }) align 4 %0)
define void @bar(ptr sret({ptr, i8}) align 4 %0) {
  ret void
}

define void @caller(ptr %ptr) personality ptr @__gxx_personality_v0 {
; CHECK: call void @bar(ptr sret({ ptr, i8 }) %ptr)
; CHECK: invoke void @bar(ptr sret({ ptr, i8 }) %ptr)
  call void @bar(ptr sret({ptr, i8}) %ptr)
  invoke void @bar(ptr sret({ptr, i8}) %ptr) to label %success unwind label %fail

success:
  ret void

fail:
  landingpad { ptr, i32 } cleanup
  ret void
}

; CHECK: declare void @baz(ptr sret([8 x i8]))
%named_type = type [8 x i8]
declare void @baz(ptr sret(%named_type))

declare i32 @__gxx_personality_v0(...)

%0 = type opaque

; CHECK: define void @anon(ptr sret({ ptr }) %arg)
; CHECK:   call void @anon_callee(ptr sret({ ptr }) %arg)
define void @anon(ptr sret({ ptr }) %arg) {
  call void @anon_callee(ptr sret({ ptr }) %arg)
  ret void
}

; CHECK: declare void @anon_callee(ptr sret({ ptr }))
declare void @anon_callee(ptr sret({ ptr }))
