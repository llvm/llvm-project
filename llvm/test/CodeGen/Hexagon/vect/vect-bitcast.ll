; RUN: llc -mtriple=hexagon < %s
; REQUIRES: asserts
; Used to fail with "Cannot BITCAST between types of different sizes!"

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define void @foo() nounwind {
entry:
  br label %while.body

while.body:                                       ; preds = %if.then155, %if.then12, %if.then, %entry
  br i1 undef, label %if.then, label %if.else

if.then:                                          ; preds = %while.body
  br label %while.body

if.else:                                          ; preds = %while.body
  br i1 undef, label %if.then12, label %if.else17

if.then12:                                        ; preds = %if.else
  br label %while.body

if.else17:                                        ; preds = %if.else
  br i1 false, label %if.then22, label %if.else35

if.then22:                                        ; preds = %if.else17
  unreachable

if.else35:                                        ; preds = %if.else17
  br i1 false, label %if.then40, label %if.else83

if.then40:                                        ; preds = %if.else35
  unreachable

if.else83:                                        ; preds = %if.else35
  br i1 false, label %if.then88, label %if.else150

if.then88:                                        ; preds = %if.else83
  unreachable

if.else150:                                       ; preds = %if.else83
  %cmp154 = icmp eq i32 undef, 0
  br i1 %cmp154, label %if.then155, label %if.else208

if.then155:                                       ; preds = %if.else150
  %_p_splat.1 = shufflevector <1 x i16> zeroinitializer, <1 x i16> undef, <2 x i32> zeroinitializer
  %0 = sext <2 x i16> %_p_splat.1 to <2 x i32>
  %mul198p_vec.1 = mul <2 x i32> %0, <i32 4, i32 4>
  %1 = extractelement <2 x i32> %mul198p_vec.1, i32 0
  store i32 %1, ptr undef, align 4
  br label %while.body

if.else208:                                       ; preds = %if.else150
  unreachable
}
