; RUN: llc -march=hexagon < %s
; REQUIRES: asserts
; Used to fail with: Assertion `VT.getSizeInBits() == Operand.getValueType().getSizeInBits() && "Cannot BITCAST between types of different sizes!"' failed.

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-a0:0-n32"
target triple = "hexagon-unknown-linux-gnu"

define void @foo() nounwind {
entry:
  br label %while.body

while.body:                                       ; preds = %if.then155, %if.then12, %entry
  %cmp.i = icmp eq ptr undef, null
  br i1 %cmp.i, label %lab_ci.exit, label %if.end.i

if.end.i:                                         ; preds = %while.body
  unreachable

lab_ci.exit:      ; preds = %while.body
  br i1 false, label %if.then, label %if.else

if.then:                                          ; preds = %lab_ci.exit
  unreachable

if.else:                                          ; preds = %lab_ci.exit
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
  %call191 = call i32 @strtol() nounwind
  %conv192 = trunc i32 %call191 to i16
  %_p_splat_one = insertelement <1 x i16> undef, i16 %conv192, i32 0
  %_p_splat = shufflevector <1 x i16> %_p_splat_one, <1 x i16> undef, <2 x i32> zeroinitializer
  %0 = sext <2 x i16> %_p_splat to <2 x i32>
  %mul198p_vec = shl <2 x i32> %0, <i32 2, i32 2>
  %1 = extractelement <2 x i32> %mul198p_vec, i32 0
  store i32 %1, ptr null, align 4
  br label %while.body

if.else208:                                       ; preds = %if.else150
  unreachable
}

declare i32 @strtol() nounwind
