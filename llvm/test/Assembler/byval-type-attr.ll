; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

; CHECK: define void @foo(i32* byval(i32) align 4)
define void @foo(i32* byval(i32) align 4) {
  ret void
}

; CHECK: define void @bar({ i32*, i8 }* byval({ i32*, i8 }) align 4)
define void @bar({i32*, i8}* byval({i32*, i8}) align 4) {
  ret void
}

define void @caller({ i32*, i8 }* %ptr) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK: call void @bar({ i32*, i8 }* byval({ i32*, i8 }) %ptr)
; CHECK: invoke void @bar({ i32*, i8 }* byval({ i32*, i8 }) %ptr)
  call void @bar({i32*, i8}* byval %ptr)
  invoke void @bar({i32*, i8}* byval %ptr) to label %success unwind label %fail

success:
  ret void

fail:
  landingpad { i8*, i32 } cleanup
  ret void
}

; CHECK: declare void @baz([8 x i8]* byval([8 x i8]))
%named_type = type [8 x i8]
declare void @baz(%named_type* byval(%named_type))

declare i32 @__gxx_personality_v0(...)
