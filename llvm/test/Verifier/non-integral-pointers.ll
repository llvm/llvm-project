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
