; RUN: not opt -verify < %s 2>&1 | FileCheck %s

target datalayout = "e-ni:4:6"

define i64 @f_0(i8 addrspace(4)* %ptr) {
; CHECK: ptrtoint not supported for non-integral pointers
  %val = ptrtoint i8 addrspace(4)* %ptr to i64
  ret i64 %val
}

define <4 x i64> @f_1(<4 x i8 addrspace(4)*> %ptr) {
; CHECK: ptrtoint not supported for non-integral pointers
  %val = ptrtoint <4 x i8 addrspace(4)*> %ptr to <4 x i64>
  ret <4 x i64> %val
}

define i64 @f_2(i8 addrspace(3)* %ptr) {
; Negative test
  %val = ptrtoint i8 addrspace(3)* %ptr to i64
  ret i64 %val
}

define i8 addrspace(4)* @f_3(i64 %integer) {
; CHECK: inttoptr not supported for non-integral pointers
  %val = inttoptr i64 %integer to i8 addrspace(4)*
  ret i8 addrspace(4)* %val
}

define <4 x i8 addrspace(4)*> @f_4(<4 x i64> %integer) {
; CHECK: inttoptr not supported for non-integral pointers
  %val = inttoptr <4 x i64> %integer to <4 x i8 addrspace(4)*>
  ret <4 x i8 addrspace(4)*> %val
}

define i8 addrspace(3)* @f_5(i64 %integer) {
; Negative test
  %val = inttoptr i64 %integer to i8 addrspace(3)*
  ret i8 addrspace(3)* %val
}

define i64 @f_6(i8 addrspace(6)* %ptr) {
; CHECK: ptrtoint not supported for non-integral pointers
  %val = ptrtoint i8 addrspace(6)* %ptr to i64
  ret i64 %val
}

define i8 addrspace(4)* @f_7() {
; CHECK: inttoptr not supported for non-integral pointers
  ret i8 addrspace(4)* inttoptr (i64 50 to i8 addrspace(4)*)
}

@global0 = addrspace(4) constant i8 42

define i64 @f_8() {
; CHECK: ptrtoint not supported for non-integral pointers
  ret i64 ptrtoint (i8 addrspace(4)* @global0 to i64)
}

define i8 addrspace(4)* @f_9() {
; CHECK: inttoptr not supported for non-integral pointers
  ret i8 addrspace(4)* getelementptr (i8, i8 addrspace(4)* inttoptr (i64 55 to i8 addrspace(4)*), i32 100)
}

@global1 = addrspace(4) constant i8 42

define i8 addrspace(4)* @f_10() {
; CHECK: ptrtoint not supported for non-integral pointers
  ret i8 addrspace(4)* getelementptr (i8, i8 addrspace(4)* @global0, i64 ptrtoint (i8 addrspace(4)* @global1 to i64))
}

@cycle_0 = addrspace(4) constant i64 ptrtoint (i64 addrspace(4)* addrspace(4)* @cycle_1 to i64)
@cycle_1 = addrspace(4) constant i64 addrspace(4) * @cycle_0

define i64 addrspace(4)* addrspace(4)* @f_11() {
; CHECK: ptrtoint not supported for non-integral pointers
  ret i64 addrspace(4)* addrspace(4)* @cycle_1
}

@cycle_self = addrspace(4) constant i64 ptrtoint (i64 addrspace(4)* @cycle_self to i64)

define i64 addrspace(4)* @f_12() {
; CHECK: ptrtoint not supported for non-integral pointers
  ret i64 addrspace(4)* @cycle_self
}
