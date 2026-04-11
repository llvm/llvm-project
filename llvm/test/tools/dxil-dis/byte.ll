; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s
target triple = "dxil-unknown-shadermodel6.7-library"

; CHECK: common global [32 x i8] zeroinitializer, align 1
@a = common global [32 x b8] zeroinitializer, align 1

define void @bytes(b1 %a, b3 %b, b5 %c, b8 %d, b16 %e, b32 %f, b64 %g, b128 %h, <8 x b5> %i, <2 x b64> %j) {
  ; Check that we generally convert byte types into int types
  ; CHECK-LABEL: define void @bytes(i1 %a, i3 %b, i5 %c, i8 %d, i16 %e, i32 %f, i64 %g, i128 %h, <8 x i5> %i, <2 x i64> %j
  ; CHECK-NEXT:    ret void
  ret void
}

define b32 @constant32() {
  ; Check that we write constant byte types as constant ints
  ; CHECK-LABEL: define i32 @constant32()
  ; CHECK:         ret i32 255
  ret b32 255
}

define b128 @constant128() {
  ; Check that we write large constant byte types as constant ints
  ; CHECK-LABEL: define i128 @constant128()
  ; CHECK:         ret i128 18446744073709551615
  ret b128 18446744073709551615
}
