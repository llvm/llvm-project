; RUN: opt -codegenprepare -S %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: shuffle_one_source

define <2 x i8> @shuffle_one_source(i32 %x) {
  %Shuf = shufflevector <2 x i8> zeroinitializer, <2 x i8> zeroinitializer, <2 x i32> poison
  %Cmp = icmp slt i32 480483, %x
  %B = mul <2 x i8> %Shuf, %Shuf
  %S = select i1 %Cmp, <2 x i8> %B, <2 x i8> zeroinitializer
  ret <2 x i8> %Shuf
}

