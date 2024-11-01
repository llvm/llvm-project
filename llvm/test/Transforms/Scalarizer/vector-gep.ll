; RUN: opt -S -passes='function(scalarizer)' %s | FileCheck %s

; Check that the scalarizer can handle vector GEPs with scalar indices

@vec = global <4 x ptr> <ptr null, ptr null, ptr null, ptr null>
@index = global i16 1
@ptr = global [4 x i16] [i16 1, i16 2, i16 3, i16 4]
@ptrptr = global ptr null

; constant index
define void @test1() {
bb:
  %0 = load <4 x ptr>, ptr @vec
  %1 = getelementptr i16, <4 x ptr> %0, i16 1

  ret void
}

;CHECK-LABEL: @test1
;CHECK: %[[I0:.i[0-9]*]] = extractelement <4 x ptr> %0, i32 0
;CHECK: getelementptr i16, ptr %[[I0]], i16 1
;CHECK: %[[I1:.i[0-9]*]] = extractelement <4 x ptr> %0, i32 1
;CHECK: getelementptr i16, ptr %[[I1]], i16 1
;CHECK: %[[I2:.i[0-9]*]] = extractelement <4 x ptr> %0, i32 2
;CHECK: getelementptr i16, ptr %[[I2]], i16 1
;CHECK: %[[I3:.i[0-9]*]] = extractelement <4 x ptr> %0, i32 3
;CHECK: getelementptr i16, ptr %[[I3]], i16 1

; non-constant index
define void @test2() {
bb:
  %0 = load <4 x ptr>, ptr @vec
  %index = load i16, ptr @index
  %1 = getelementptr i16, <4 x ptr> %0, i16 %index

  ret void
}

;CHECK-LABEL: @test2
;CHECK: %0 = load <4 x ptr>, ptr @vec
;CHECK: %[[I0:.i[0-9]*]] = extractelement <4 x ptr> %0, i32 0
;CHECK: %[[I1:.i[0-9]*]] = extractelement <4 x ptr> %0, i32 1
;CHECK: %[[I2:.i[0-9]*]] = extractelement <4 x ptr> %0, i32 2
;CHECK: %[[I3:.i[0-9]*]] = extractelement <4 x ptr> %0, i32 3
;CHECK: %index = load i16, ptr @index
;CHECK: %.splatinsert = insertelement <4 x i16> poison, i16 %index, i32 0
;CHECK: %.splat = shufflevector <4 x i16> %.splatinsert, <4 x i16> poison, <4 x i32> zeroinitializer
;CHECK: %.splat[[I0]] = extractelement <4 x i16> %.splat, i32 0
;CHECK: getelementptr i16, ptr %[[I0]], i16 %.splat[[I0]]
;CHECK: %.splat[[I1]] = extractelement <4 x i16> %.splat, i32 1
;CHECK: getelementptr i16, ptr %[[I1]], i16 %.splat[[I1]]
;CHECK: %.splat[[I2]] = extractelement <4 x i16> %.splat, i32 2
;CHECK: getelementptr i16, ptr %[[I2]], i16 %.splat[[I2]]
;CHECK: %.splat[[I3]] = extractelement <4 x i16> %.splat, i32 3
;CHECK: getelementptr i16, ptr %[[I3]], i16 %.splat[[I3]]


; Check that the scalarizer can handle vector GEPs with scalar pointer

; constant pointer
define <4 x ptr> @test3_constexpr() {
bb:
  ret <4 x ptr> getelementptr (i16, ptr @ptr, <4 x i64> <i64 0, i64 1, i64 2, i64 3>)
}

; CHECK-LABEL: @test3_constexpr
; CHECK: ret <4 x ptr> getelementptr (i16, ptr @ptr, <4 x i64> <i64 0, i64 1, i64 2, i64 3>)


define <4 x ptr> @test3_constbase(i16 %idx) {
bb:
  %offset = getelementptr [4 x i16], ptr @ptr, i16 0, i16 %idx
  %gep = getelementptr i16, ptr %offset, <4 x i16> <i16 0, i16 1, i16 2, i16 3>
  ret <4 x ptr> %gep
}

; CHECK-LABEL: @test3_constbase(
; CHECK: %offset = getelementptr [4 x i16], ptr @ptr, i16 0, i16 %idx
; CHECK: %.splatinsert = insertelement <4 x ptr> poison, ptr %offset, i32 0
; CHECK: %.splat = shufflevector <4 x ptr> %.splatinsert, <4 x ptr> poison, <4 x i32> zeroinitializer
; CHECK: %.splat[[I0:.i[0-9]*]] = extractelement <4 x ptr> %.splat, i32 0
; CHECK: getelementptr i16, ptr %.splat[[I0]], i16 0
; CHECK: %.splat[[I1:.i[0-9]*]] = extractelement <4 x ptr> %.splat, i32 1
; CHECK: getelementptr i16, ptr %.splat[[I1]], i16 1
; CHECK: %.splat[[I2:.i[0-9]*]] = extractelement <4 x ptr> %.splat, i32 2
; CHECK: getelementptr i16, ptr %.splat[[I2]], i16 2
; CHECK: %.splat[[I3:.i[0-9]*]] = extractelement <4 x ptr> %.splat, i32 3
; CHECK: getelementptr i16, ptr %.splat[[I3]], i16 3

; Constant pointer with a variable vector offset
define <4 x ptr> @test3_varoffset(<4 x i16> %offset) {
bb:
  %gep = getelementptr i16, ptr @ptr, <4 x i16> %offset
  ret <4 x ptr> %gep
}

; CHECK-LABEL: @test3_varoffset
; CHECK: %offset.i0 = extractelement <4 x i16> %offset, i32 0
; CHECK: %gep.i0 = getelementptr i16, ptr @ptr, i16 %offset.i0
; CHECK: %offset.i1 = extractelement <4 x i16> %offset, i32 1
; CHECK: %gep.i1 = getelementptr i16, ptr @ptr, i16 %offset.i1
; CHECK: %offset.i2 = extractelement <4 x i16> %offset, i32 2
; CHECK: %gep.i2 = getelementptr i16, ptr @ptr, i16 %offset.i2
; CHECK: %offset.i3 = extractelement <4 x i16> %offset, i32 3
; CHECK: %gep.i3 = getelementptr i16, ptr @ptr, i16 %offset.i3

; non-constant pointer
define void @test4() {
bb:
  %0 = load ptr, ptr @ptrptr
  %1 = getelementptr i16, ptr %0, <4 x i16> <i16 0, i16 1, i16 2, i16 3>

  ret void
}

;CHECK-LABEL: @test4
;CHECK: %0 = load ptr, ptr @ptrptr
;CHECK: %.splatinsert = insertelement <4 x ptr> poison, ptr %0, i32 0
;CHECK: %.splat = shufflevector <4 x ptr> %.splatinsert, <4 x ptr> poison, <4 x i32> zeroinitializer
;CHECK: %.splat[[I0:.i[0-9]*]] = extractelement <4 x ptr> %.splat, i32 0
;CHECK: getelementptr i16, ptr %.splat[[I0]], i16 0
;CHECK: %.splat[[I1:.i[0-9]*]] = extractelement <4 x ptr> %.splat, i32 1
;CHECK: getelementptr i16, ptr %.splat[[I1]], i16 1
;CHECK: %.splat[[I2:.i[0-9]*]] = extractelement <4 x ptr> %.splat, i32 2
;CHECK: getelementptr i16, ptr %.splat[[I2]], i16 2
;CHECK: %.splat[[I3:.i[0-9]*]] = extractelement <4 x ptr> %.splat, i32 3
;CHECK: getelementptr i16, ptr %.splat[[I3]], i16 3

; constant index, inbounds
define void @test5() {
bb:
  %0 = load <4 x ptr>, ptr @vec
  %1 = getelementptr inbounds i16, <4 x ptr> %0, i16 1

  ret void
}

;CHECK-LABEL: @test5
;CHECK: %[[I0:.i[0-9]*]] = extractelement <4 x ptr> %0, i32 0
;CHECK: getelementptr inbounds i16, ptr %[[I0]], i16 1
;CHECK: %[[I1:.i[0-9]*]] = extractelement <4 x ptr> %0, i32 1
;CHECK: getelementptr inbounds i16, ptr %[[I1]], i16 1
;CHECK: %[[I2:.i[0-9]*]] = extractelement <4 x ptr> %0, i32 2
;CHECK: getelementptr inbounds i16, ptr %[[I2]], i16 1
;CHECK: %[[I3:.i[0-9]*]] = extractelement <4 x ptr> %0, i32 3
;CHECK: getelementptr inbounds i16, ptr %[[I3]], i16 1

