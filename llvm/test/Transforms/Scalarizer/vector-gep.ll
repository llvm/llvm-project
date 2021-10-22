; RUN: opt -S -passes='function(scalarizer)' %s | FileCheck %s

; Check that the scalarizer can handle vector GEPs with scalar indices

@vec = global <4 x i16*> <i16* null, i16* null, i16* null, i16* null>
@index = global i16 1
@ptr = global [4 x i16] [i16 1, i16 2, i16 3, i16 4]
@ptrptr = global i16* null

; constant index
define void @test1() {
bb:
  %0 = load <4 x i16*>, <4 x i16*>* @vec
  %1 = getelementptr i16, <4 x i16*> %0, i16 1

  ret void
}

;CHECK-LABEL: @test1
;CHECK: %[[I0:.i[0-9]*]] = extractelement <4 x i16*> %0, i32 0
;CHECK: getelementptr i16, i16* %[[I0]], i16 1
;CHECK: %[[I1:.i[0-9]*]] = extractelement <4 x i16*> %0, i32 1
;CHECK: getelementptr i16, i16* %[[I1]], i16 1
;CHECK: %[[I2:.i[0-9]*]] = extractelement <4 x i16*> %0, i32 2
;CHECK: getelementptr i16, i16* %[[I2]], i16 1
;CHECK: %[[I3:.i[0-9]*]] = extractelement <4 x i16*> %0, i32 3
;CHECK: getelementptr i16, i16* %[[I3]], i16 1

; non-constant index
define void @test2() {
bb:
  %0 = load <4 x i16*>, <4 x i16*>* @vec
  %index = load i16, i16* @index
  %1 = getelementptr i16, <4 x i16*> %0, i16 %index

  ret void
}

;CHECK-LABEL: @test2
;CHECK: %0 = load <4 x i16*>, <4 x i16*>* @vec
;CHECK: %[[I0:.i[0-9]*]] = extractelement <4 x i16*> %0, i32 0
;CHECK: %[[I1:.i[0-9]*]] = extractelement <4 x i16*> %0, i32 1
;CHECK: %[[I2:.i[0-9]*]] = extractelement <4 x i16*> %0, i32 2
;CHECK: %[[I3:.i[0-9]*]] = extractelement <4 x i16*> %0, i32 3
;CHECK: %index = load i16, i16* @index
;CHECK: %.splatinsert = insertelement <4 x i16> poison, i16 %index, i32 0
;CHECK: %.splat = shufflevector <4 x i16> %.splatinsert, <4 x i16> poison, <4 x i32> zeroinitializer
;CHECK: %.splat[[I0]] = extractelement <4 x i16> %.splat, i32 0
;CHECK: getelementptr i16, i16* %[[I0]], i16 %.splat[[I0]]
;CHECK: %.splat[[I1]] = extractelement <4 x i16> %.splat, i32 1
;CHECK: getelementptr i16, i16* %[[I1]], i16 %.splat[[I1]]
;CHECK: %.splat[[I2]] = extractelement <4 x i16> %.splat, i32 2
;CHECK: getelementptr i16, i16* %[[I2]], i16 %.splat[[I2]]
;CHECK: %.splat[[I3]] = extractelement <4 x i16> %.splat, i32 3
;CHECK: getelementptr i16, i16* %[[I3]], i16 %.splat[[I3]]


; Check that the scalarizer can handle vector GEPs with scalar pointer

; constant pointer
define void @test3() {
bb:
  %0 = bitcast [4 x i16]* @ptr to i16*
  %1 = getelementptr i16, i16* %0, <4 x i16> <i16 0, i16 1, i16 2, i16 3>

  ret void
}

;CHECK-LABEL: @test3
;CHECK: %0 = bitcast [4 x i16]* @ptr to i16*
;CHECK: %.splatinsert = insertelement <4 x i16*> poison, i16* %0, i32 0
;CHECK: %.splat = shufflevector <4 x i16*> %.splatinsert, <4 x i16*> poison, <4 x i32> zeroinitializer
;CHECK: %.splat[[I0:.i[0-9]*]] = extractelement <4 x i16*> %.splat, i32 0
;CHECK: getelementptr i16, i16* %.splat[[I0]], i16 0
;CHECK: %.splat[[I1:.i[0-9]*]] = extractelement <4 x i16*> %.splat, i32 1
;CHECK: getelementptr i16, i16* %.splat[[I1]], i16 1
;CHECK: %.splat[[I2:.i[0-9]*]] = extractelement <4 x i16*> %.splat, i32 2
;CHECK: getelementptr i16, i16* %.splat[[I2]], i16 2
;CHECK: %.splat[[I3:.i[0-9]*]] = extractelement <4 x i16*> %.splat, i32 3
;CHECK: getelementptr i16, i16* %.splat[[I3]], i16 3

; non-constant pointer
define void @test4() {
bb:
  %0 = load i16*, i16** @ptrptr
  %1 = getelementptr i16, i16* %0, <4 x i16> <i16 0, i16 1, i16 2, i16 3>

  ret void
}

;CHECK-LABEL: @test4
;CHECK: %0 = load i16*, i16** @ptrptr
;CHECK: %.splatinsert = insertelement <4 x i16*> poison, i16* %0, i32 0
;CHECK: %.splat = shufflevector <4 x i16*> %.splatinsert, <4 x i16*> poison, <4 x i32> zeroinitializer
;CHECK: %.splat[[I0:.i[0-9]*]] = extractelement <4 x i16*> %.splat, i32 0
;CHECK: getelementptr i16, i16* %.splat[[I0]], i16 0
;CHECK: %.splat[[I1:.i[0-9]*]] = extractelement <4 x i16*> %.splat, i32 1
;CHECK: getelementptr i16, i16* %.splat[[I1]], i16 1
;CHECK: %.splat[[I2:.i[0-9]*]] = extractelement <4 x i16*> %.splat, i32 2
;CHECK: getelementptr i16, i16* %.splat[[I2]], i16 2
;CHECK: %.splat[[I3:.i[0-9]*]] = extractelement <4 x i16*> %.splat, i32 3
;CHECK: getelementptr i16, i16* %.splat[[I3]], i16 3

; constant index, inbounds
define void @test5() {
bb:
  %0 = load <4 x i16*>, <4 x i16*>* @vec
  %1 = getelementptr inbounds i16, <4 x i16*> %0, i16 1

  ret void
}

;CHECK-LABEL: @test5
;CHECK: %[[I0:.i[0-9]*]] = extractelement <4 x i16*> %0, i32 0
;CHECK: getelementptr inbounds i16, i16* %[[I0]], i16 1
;CHECK: %[[I1:.i[0-9]*]] = extractelement <4 x i16*> %0, i32 1
;CHECK: getelementptr inbounds i16, i16* %[[I1]], i16 1
;CHECK: %[[I2:.i[0-9]*]] = extractelement <4 x i16*> %0, i32 2
;CHECK: getelementptr inbounds i16, i16* %[[I2]], i16 1
;CHECK: %[[I3:.i[0-9]*]] = extractelement <4 x i16*> %0, i32 3
;CHECK: getelementptr inbounds i16, i16* %[[I3]], i16 1

