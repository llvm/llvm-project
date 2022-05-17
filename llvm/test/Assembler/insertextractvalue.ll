; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK:      @foo
; CHECK-NEXT: load
; CHECK-NEXT: extractvalue
; CHECK-NEXT: insertvalue
; CHECK-NEXT: store
; CHECK-NEXT: ret
define float @foo({{i32},{float, double}}* %p) nounwind {
  %t = load {{i32},{float, double}}, {{i32},{float, double}}* %p
  %s = extractvalue {{i32},{float, double}} %t, 1, 0
  %r = insertvalue {{i32},{float, double}} %t, double 2.0, 1, 1
  store {{i32},{float, double}} %r, {{i32},{float, double}}* %p
  ret float %s
}

; CHECK-LABEL: @bar
; CHECK-NEXT: store { { i32 }, { float, double } } { { i32 } { i32 4 }, { float, double } { float 4.000000e+00, double 2.000000e+01 } }, { { i32 }, { float, double } }* %p
define void @bar({{i32},{float, double}}* %p) nounwind {
  store {{i32},{float, double}} insertvalue ({{i32},{float, double}}{{i32}{i32 4},{float, double}{float 4.0, double 5.0}}, double 20.0, 1, 1), {{i32},{float, double}}* %p
  ret void
}

; CHECK-LABEL: @car
; CHECK-NEXT: store { { i32 }, { float, double } } { { i32 } undef, { float, double } { float undef, double 2.000000e+01 } }, { { i32 }, { float, double } }* %p
define void @car({{i32},{float, double}}* %p) nounwind {
  store {{i32},{float, double}} insertvalue ({{i32},{float, double}} undef, double 20.0, 1, 1), {{i32},{float, double}}* %p
  ret void
}

; CHECK-LABEL: @dar
; CHECK-NEXT: store { { i32 }, { float, double } } { { i32 } zeroinitializer, { float, double } { float 0.000000e+00, double 2.000000e+01 } }, { { i32 }, { float, double } }* %p
define void @dar({{i32},{float, double}}* %p) nounwind {
  store {{i32},{float, double}} insertvalue ({{i32},{float, double}} zeroinitializer, double 20.0, 1, 1), {{i32},{float, double}}* %p
  ret void
}

; PR4963
; CHECK:      @test57
; CHECK-NEXT: ret <{ i32, i32 }> <{ i32 0, i32 4 }>
define <{ i32, i32 }> @test57() {
  ret <{ i32, i32 }> insertvalue (<{ i32, i32 }> zeroinitializer, i32 4, 1)
}
