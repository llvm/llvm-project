; RUN: llc < %s -mtriple=i686-- -mattr=+sse2 | FileCheck %s

; CHECK: foo
; CHECK: xorps
define void @foo(<4 x float>* %P) {
        %T = load <4 x float>, <4 x float>* %P               ; <<4 x float>> [#uses=1]
        %S = fadd <4 x float> zeroinitializer, %T                ; <<4 x float>> [#uses=1]
        store <4 x float> %S, <4 x float>* %P
        ret void
}

; CHECK: bar
; CHECK: pxor
define void @bar(<4 x i32>* %P) {
        %T = load <4 x i32>, <4 x i32>* %P         ; <<4 x i32>> [#uses=1]
        %S = sub <4 x i32> zeroinitializer, %T          ; <<4 x i32>> [#uses=1]
        store <4 x i32> %S, <4 x i32>* %P
        ret void
}

; Without any type hints from operations, we fall back to the smaller xorps.
; The IR type <4 x i32> is ignored.
; CHECK: untyped_zero
; CHECK: xorps
; CHECK: movaps
define void @untyped_zero(<4 x i32>* %p) {
entry:
  store <4 x i32> zeroinitializer, <4 x i32>* %p, align 16
  ret void
}
