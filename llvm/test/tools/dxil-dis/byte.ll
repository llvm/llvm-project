; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s
target triple = "dxil-unknown-shadermodel6.7-library"

; CHECK: common global [32 x i8] zeroinitializer, align 1
@a = common global [32 x b8] zeroinitializer, align 1

define void @bytes(b1 %a, b3 %b, b5 %c, b8 %d, b16 %e, b32 %f, b64 %g, b128 %h, <8 x b5> %i, <2 x b64> %j) {
; CHECK-LABEL: define void @bytes(
; CHECK-SAME: i1 [[A:%.*]], i3 [[B:%.*]], i5 [[C:%.*]], i8 [[D:%.*]], i16 [[E:%.*]], i32 [[F:%.*]], i64 [[G:%.*]], i128 [[H:%.*]], <8 x i5> [[I:%.*]], <2 x i64> [[J:%.*]]) {
; CHECK-NEXT:    ret void
;
  ret void
}
