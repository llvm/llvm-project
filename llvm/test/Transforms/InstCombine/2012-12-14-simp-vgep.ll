; RUN: opt < %s -passes=instcombine -S

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

define <4 x i32> @foo(ptr %in) {
  %t17 = load <4 x ptr>, ptr %in, align 8
  %t18 = icmp eq <4 x ptr> %t17, zeroinitializer
  %t19 = zext <4 x i1> %t18 to <4 x i32>
  ret <4 x i32> %t19
}
