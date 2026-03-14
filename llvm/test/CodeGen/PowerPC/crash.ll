; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7

define void @test1(i1 %x, i8 %x2, ptr %x3, i64 %x4) {
entry:
  %tmp3 = and i64 %x4, 16
  %bf.shl = trunc i64 %tmp3 to i8
  %bf.clear = and i8 %x2, -17
  %bf.set = or i8 %bf.shl, %bf.clear
  br i1 %x, label %if.then, label %if.end

if.then:
  ret void

if.end:
  store i8 %bf.set, ptr %x3, align 4
  ret void
}

; A BUILD_VECTOR of 1 element caused a crash in combineBVOfConsecutiveLoads()
; Test that this is no longer the case
define signext i32 @test2() {
entry:
  %retval = alloca i32, align 4
  %__a = alloca i128, align 16
  %b = alloca i64, align 8
  store i32 0, ptr %retval, align 4
  %0 = load i128, ptr %__a, align 16
  %splat.splatinsert = insertelement <1 x i128> undef, i128 %0, i32 0
  %splat.splat = shufflevector <1 x i128> %splat.splatinsert, <1 x i128> undef, <1 x i32> zeroinitializer
  %1 = bitcast <1 x i128> %splat.splat to <2 x i64>
  %2 = extractelement <2 x i64> %1, i32 0
  store i64 %2, ptr %b, align 8
  ret i32 0
}
