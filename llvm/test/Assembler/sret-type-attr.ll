; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

; CHECK: define void @foo(i32* sret(i32) align 4 %0)
define void @foo(i32* sret(i32) align 4 %0) {
  ret void
}

; CHECK: define void @bar({ i32*, i8 }* sret({ i32*, i8 }) align 4 %0)
define void @bar({i32*, i8}* sret({i32*, i8}) align 4 %0) {
  ret void
}

define void @caller({ i32*, i8 }* %ptr) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK: call void @bar({ i32*, i8 }* sret({ i32*, i8 }) %ptr)
; CHECK: invoke void @bar({ i32*, i8 }* sret({ i32*, i8 }) %ptr)
  call void @bar({i32*, i8}* sret({i32*, i8}) %ptr)
  invoke void @bar({i32*, i8}* sret({i32*, i8}) %ptr) to label %success unwind label %fail

success:
  ret void

fail:
  landingpad { i8*, i32 } cleanup
  ret void
}

; CHECK: declare void @baz([8 x i8]* sret([8 x i8]))
%named_type = type [8 x i8]
declare void @baz(%named_type* sret(%named_type))

declare i32 @__gxx_personality_v0(...)

%0 = type opaque

; CHECK: define void @anon({ %0* }* sret({ %0* }) %arg)
; CHECK:   call void @anon_callee({ %0* }* sret({ %0* }) %arg)
define void @anon({ %0* }* sret({ %0* }) %arg) {
  call void @anon_callee({ %0* }* sret({ %0* }) %arg)
  ret void
}

; CHECK: declare void @anon_callee({ %0* }* sret({ %0* }))
declare void @anon_callee({ %0* }* sret({ %0* }))
