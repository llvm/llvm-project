; RUN: opt < %s -sroa -S | FileCheck %s
; RUN: opt < %s -passes=sroa -S | FileCheck %s

target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

declare void @llvm.lifetime.start(i64, i8* nocapture)
declare void @llvm.lifetime.end(i64, i8* nocapture)

define i32 @test0() {
; CHECK-LABEL: @test0(
; CHECK-NOT: alloca
; CHECK: ret i32

entry:
  %a1 = alloca i32
  %a2 = alloca float

  %a1.i8 = bitcast i32* %a1 to i8*
  call void @llvm.lifetime.start(i64 4, i8* %a1.i8)

  store i32 0, i32* %a1
  %v1 = load i32, i32* %a1

  call void @llvm.lifetime.end(i64 4, i8* %a1.i8)

  %a2.i8 = bitcast float* %a2 to i8*
  call void @llvm.lifetime.start(i64 4, i8* %a2.i8)

  store float 0.0, float* %a2
  %v2 = load float , float * %a2
  %v2.int = bitcast float %v2 to i32
  %sum1 = add i32 %v1, %v2.int

  call void @llvm.lifetime.end(i64 4, i8* %a2.i8)

  ret i32 %sum1
}

define i32 @test1() {
; CHECK-LABEL: @test1(
; CHECK-NOT: alloca
; CHECK: ret i32 0

entry:
  %X = alloca { i32, float }
  %Y = getelementptr { i32, float }, { i32, float }* %X, i64 0, i32 0
  store i32 0, i32* %Y
  %Z = load i32, i32* %Y
  ret i32 %Z
}

define i64 @test2(i64 %X) {
; CHECK-LABEL: @test2(
; CHECK-NOT: alloca
; CHECK: ret i64 %X

entry:
  %A = alloca [8 x i8]
  %B = bitcast [8 x i8]* %A to i64*
  store i64 %X, i64* %B
  br label %L2

L2:
  %Z = load i64, i64* %B
  ret i64 %Z
}

define void @test3(i8* %dst, i8* %src) {
; CHECK-LABEL: @test3(

entry:
  %a = alloca [300 x i8]
; CHECK-NOT:  alloca
; CHECK:      %[[test3_a1:.*]] = alloca [42 x i8]
; CHECK-NEXT: %[[test3_a2:.*]] = alloca [99 x i8]
; CHECK-NEXT: %[[test3_a3:.*]] = alloca [16 x i8]
; CHECK-NEXT: %[[test3_a4:.*]] = alloca [42 x i8]
; CHECK-NEXT: %[[test3_a5:.*]] = alloca [7 x i8]
; CHECK-NEXT: %[[test3_a6:.*]] = alloca [7 x i8]
; CHECK-NEXT: %[[test3_a7:.*]] = alloca [85 x i8]

  %b = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %b, i8* %src, i32 300, i32 1, i1 false)
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [42 x i8], [42 x i8]* %[[test3_a1]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %src, i32 42
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8, i8* %src, i64 42
; CHECK-NEXT: %[[test3_r1:.*]] = load i8, i8* %[[gep]]
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8, i8* %src, i64 43
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [99 x i8], [99 x i8]* %[[test3_a2]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 99
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8, i8* %src, i64 142
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [16 x i8], [16 x i8]* %[[test3_a3]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 16
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8, i8* %src, i64 158
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [42 x i8], [42 x i8]* %[[test3_a4]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 42
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8, i8* %src, i64 200
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test3_a5]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8, i8* %src, i64 207
; CHECK-NEXT: %[[test3_r2:.*]] = load i8, i8* %[[gep]]
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8, i8* %src, i64 208
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test3_a6]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8, i8* %src, i64 215
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [85 x i8], [85 x i8]* %[[test3_a7]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 85

  ; Clobber a single element of the array, this should be promotable.
  %c = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 42
  store i8 0, i8* %c

  ; Make a sequence of overlapping stores to the array. These overlap both in
  ; forward strides and in shrinking accesses.
  %overlap.1.i8 = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 142
  %overlap.2.i8 = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 143
  %overlap.3.i8 = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 144
  %overlap.4.i8 = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 145
  %overlap.5.i8 = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 146
  %overlap.6.i8 = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 147
  %overlap.7.i8 = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 148
  %overlap.8.i8 = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 149
  %overlap.9.i8 = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 150
  %overlap.1.i16 = bitcast i8* %overlap.1.i8 to i16*
  %overlap.1.i32 = bitcast i8* %overlap.1.i8 to i32*
  %overlap.1.i64 = bitcast i8* %overlap.1.i8 to i64*
  %overlap.2.i64 = bitcast i8* %overlap.2.i8 to i64*
  %overlap.3.i64 = bitcast i8* %overlap.3.i8 to i64*
  %overlap.4.i64 = bitcast i8* %overlap.4.i8 to i64*
  %overlap.5.i64 = bitcast i8* %overlap.5.i8 to i64*
  %overlap.6.i64 = bitcast i8* %overlap.6.i8 to i64*
  %overlap.7.i64 = bitcast i8* %overlap.7.i8 to i64*
  %overlap.8.i64 = bitcast i8* %overlap.8.i8 to i64*
  %overlap.9.i64 = bitcast i8* %overlap.9.i8 to i64*
  store i8 1, i8* %overlap.1.i8
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8], [16 x i8]* %[[test3_a3]], i64 0, i64 0
; CHECK-NEXT: store i8 1, i8* %[[gep]]
  store i16 1, i16* %overlap.1.i16
; CHECK-NEXT: %[[bitcast:.*]] = bitcast [16 x i8]* %[[test3_a3]] to i16*
; CHECK-NEXT: store i16 1, i16* %[[bitcast]]
  store i32 1, i32* %overlap.1.i32
; CHECK-NEXT: %[[bitcast:.*]] = bitcast [16 x i8]* %[[test3_a3]] to i32*
; CHECK-NEXT: store i32 1, i32* %[[bitcast]]
  store i64 1, i64* %overlap.1.i64
; CHECK-NEXT: %[[bitcast:.*]] = bitcast [16 x i8]* %[[test3_a3]] to i64*
; CHECK-NEXT: store i64 1, i64* %[[bitcast]]
  store i64 2, i64* %overlap.2.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8], [16 x i8]* %[[test3_a3]], i64 0, i64 1
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 2, i64* %[[bitcast]]
  store i64 3, i64* %overlap.3.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8], [16 x i8]* %[[test3_a3]], i64 0, i64 2
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 3, i64* %[[bitcast]]
  store i64 4, i64* %overlap.4.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8], [16 x i8]* %[[test3_a3]], i64 0, i64 3
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 4, i64* %[[bitcast]]
  store i64 5, i64* %overlap.5.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8], [16 x i8]* %[[test3_a3]], i64 0, i64 4
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 5, i64* %[[bitcast]]
  store i64 6, i64* %overlap.6.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8], [16 x i8]* %[[test3_a3]], i64 0, i64 5
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 6, i64* %[[bitcast]]
  store i64 7, i64* %overlap.7.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8], [16 x i8]* %[[test3_a3]], i64 0, i64 6
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 7, i64* %[[bitcast]]
  store i64 8, i64* %overlap.8.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8], [16 x i8]* %[[test3_a3]], i64 0, i64 7
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 8, i64* %[[bitcast]]
  store i64 9, i64* %overlap.9.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8], [16 x i8]* %[[test3_a3]], i64 0, i64 8
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 9, i64* %[[bitcast]]

  ; Make two sequences of overlapping stores with more gaps and irregularities.
  %overlap2.1.0.i8 = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 200
  %overlap2.1.1.i8 = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 201
  %overlap2.1.2.i8 = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 202
  %overlap2.1.3.i8 = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 203

  %overlap2.2.0.i8 = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 208
  %overlap2.2.1.i8 = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 209
  %overlap2.2.2.i8 = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 210
  %overlap2.2.3.i8 = getelementptr [300 x i8], [300 x i8]* %a, i64 0, i64 211

  %overlap2.1.0.i16 = bitcast i8* %overlap2.1.0.i8 to i16*
  %overlap2.1.0.i32 = bitcast i8* %overlap2.1.0.i8 to i32*
  %overlap2.1.1.i32 = bitcast i8* %overlap2.1.1.i8 to i32*
  %overlap2.1.2.i32 = bitcast i8* %overlap2.1.2.i8 to i32*
  %overlap2.1.3.i32 = bitcast i8* %overlap2.1.3.i8 to i32*
  store i8 1,  i8*  %overlap2.1.0.i8
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test3_a5]], i64 0, i64 0
; CHECK-NEXT: store i8 1, i8* %[[gep]]
  store i16 1, i16* %overlap2.1.0.i16
; CHECK-NEXT: %[[bitcast:.*]] = bitcast [7 x i8]* %[[test3_a5]] to i16*
; CHECK-NEXT: store i16 1, i16* %[[bitcast]]
  store i32 1, i32* %overlap2.1.0.i32
; CHECK-NEXT: %[[bitcast:.*]] = bitcast [7 x i8]* %[[test3_a5]] to i32*
; CHECK-NEXT: store i32 1, i32* %[[bitcast]]
  store i32 2, i32* %overlap2.1.1.i32
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test3_a5]], i64 0, i64 1
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT: store i32 2, i32* %[[bitcast]]
  store i32 3, i32* %overlap2.1.2.i32
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test3_a5]], i64 0, i64 2
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT: store i32 3, i32* %[[bitcast]]
  store i32 4, i32* %overlap2.1.3.i32
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test3_a5]], i64 0, i64 3
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT: store i32 4, i32* %[[bitcast]]

  %overlap2.2.0.i32 = bitcast i8* %overlap2.2.0.i8 to i32*
  %overlap2.2.1.i16 = bitcast i8* %overlap2.2.1.i8 to i16*
  %overlap2.2.1.i32 = bitcast i8* %overlap2.2.1.i8 to i32*
  %overlap2.2.2.i32 = bitcast i8* %overlap2.2.2.i8 to i32*
  %overlap2.2.3.i32 = bitcast i8* %overlap2.2.3.i8 to i32*
  store i32 1, i32* %overlap2.2.0.i32
; CHECK-NEXT: %[[bitcast:.*]] = bitcast [7 x i8]* %[[test3_a6]] to i32*
; CHECK-NEXT: store i32 1, i32* %[[bitcast]]
  store i8 1,  i8*  %overlap2.2.1.i8
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test3_a6]], i64 0, i64 1
; CHECK-NEXT: store i8 1, i8* %[[gep]]
  store i16 1, i16* %overlap2.2.1.i16
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test3_a6]], i64 0, i64 1
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: store i16 1, i16* %[[bitcast]]
  store i32 1, i32* %overlap2.2.1.i32
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test3_a6]], i64 0, i64 1
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT: store i32 1, i32* %[[bitcast]]
  store i32 3, i32* %overlap2.2.2.i32
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test3_a6]], i64 0, i64 2
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT: store i32 3, i32* %[[bitcast]]
  store i32 4, i32* %overlap2.2.3.i32
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test3_a6]], i64 0, i64 3
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT: store i32 4, i32* %[[bitcast]]

  %overlap2.prefix = getelementptr i8, i8* %overlap2.1.1.i8, i64 -4
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %overlap2.prefix, i8* %src, i32 8, i32 1, i1 false)
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [42 x i8], [42 x i8]* %[[test3_a4]], i64 0, i64 39
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %src, i32 3
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8, i8* %src, i64 3
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test3_a5]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 5

  ; Bridge between the overlapping areas
  call void @llvm.memset.p0i8.i32(i8* %overlap2.1.2.i8, i8 42, i32 8, i32 1, i1 false)
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test3_a5]], i64 0, i64 2
; CHECK-NEXT: call void @llvm.memset.p0i8.i32(i8* %[[gep]], i8 42, i32 5
; ...promoted i8 store...
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test3_a6]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memset.p0i8.i32(i8* %[[gep]], i8 42, i32 2

  ; Entirely within the second overlap.
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %overlap2.2.1.i8, i8* %src, i32 5, i32 1, i1 false)
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test3_a6]], i64 0, i64 1
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep]], i8* %src, i32 5

  ; Trailing past the second overlap.
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %overlap2.2.2.i8, i8* %src, i32 8, i32 1, i1 false)
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test3_a6]], i64 0, i64 2
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep]], i8* %src, i32 5
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8, i8* %src, i64 5
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [85 x i8], [85 x i8]* %[[test3_a7]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 3

  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %b, i32 300, i32 1, i1 false)
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [42 x i8], [42 x i8]* %[[test3_a1]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %[[gep]], i32 42
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8, i8* %dst, i64 42
; CHECK-NEXT: store i8 0, i8* %[[gep]]
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8, i8* %dst, i64 43
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [99 x i8], [99 x i8]* %[[test3_a2]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 99
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8, i8* %dst, i64 142
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [16 x i8], [16 x i8]* %[[test3_a3]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 16
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8, i8* %dst, i64 158
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [42 x i8], [42 x i8]* %[[test3_a4]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 42
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8, i8* %dst, i64 200
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test3_a5]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8, i8* %dst, i64 207
; CHECK-NEXT: store i8 42, i8* %[[gep]]
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8, i8* %dst, i64 208
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test3_a6]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8, i8* %dst, i64 215
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [85 x i8], [85 x i8]* %[[test3_a7]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 85

  ret void
}

define void @test4(i8* %dst, i8* %src) {
; CHECK-LABEL: @test4(

entry:
  %a = alloca [100 x i8]
; CHECK-NOT:  alloca
; CHECK:      %[[test4_a1:.*]] = alloca [20 x i8]
; CHECK-NEXT: %[[test4_a2:.*]] = alloca [7 x i8]
; CHECK-NEXT: %[[test4_a3:.*]] = alloca [10 x i8]
; CHECK-NEXT: %[[test4_a4:.*]] = alloca [7 x i8]
; CHECK-NEXT: %[[test4_a5:.*]] = alloca [7 x i8]
; CHECK-NEXT: %[[test4_a6:.*]] = alloca [40 x i8]

  %b = getelementptr [100 x i8], [100 x i8]* %a, i64 0, i64 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %b, i8* %src, i32 100, i32 1, i1 false)
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [20 x i8], [20 x i8]* %[[test4_a1]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep]], i8* %src, i32 20
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8, i8* %src, i64 20
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: %[[test4_r1:.*]] = load i16, i16* %[[bitcast]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8, i8* %src, i64 22
; CHECK-NEXT: %[[test4_r2:.*]] = load i8, i8* %[[gep]]
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8, i8* %src, i64 23
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test4_a2]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8, i8* %src, i64 30
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [10 x i8], [10 x i8]* %[[test4_a3]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 10
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8, i8* %src, i64 40
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: %[[test4_r3:.*]] = load i16, i16* %[[bitcast]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8, i8* %src, i64 42
; CHECK-NEXT: %[[test4_r4:.*]] = load i8, i8* %[[gep]]
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8, i8* %src, i64 43
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test4_a4]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8, i8* %src, i64 50
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: %[[test4_r5:.*]] = load i16, i16* %[[bitcast]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8, i8* %src, i64 52
; CHECK-NEXT: %[[test4_r6:.*]] = load i8, i8* %[[gep]]
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8, i8* %src, i64 53
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test4_a5]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8, i8* %src, i64 60
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [40 x i8], [40 x i8]* %[[test4_a6]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 40

  %a.src.1 = getelementptr [100 x i8], [100 x i8]* %a, i64 0, i64 20
  %a.dst.1 = getelementptr [100 x i8], [100 x i8]* %a, i64 0, i64 40
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a.dst.1, i8* %a.src.1, i32 10, i32 1, i1 false)
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test4_a4]], i64 0, i64 0
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test4_a2]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7

  ; Clobber a single element of the array, this should be promotable, and be deleted.
  %c = getelementptr [100 x i8], [100 x i8]* %a, i64 0, i64 42
  store i8 0, i8* %c

  %a.src.2 = getelementptr [100 x i8], [100 x i8]* %a, i64 0, i64 50
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %a.dst.1, i8* %a.src.2, i32 10, i32 1, i1 false)
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test4_a4]], i64 0, i64 0
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test4_a5]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7

  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %b, i32 100, i32 1, i1 false)
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [20 x i8], [20 x i8]* %[[test4_a1]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %[[gep]], i32 20
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8, i8* %dst, i64 20
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: store i16 %[[test4_r1]], i16* %[[bitcast]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8, i8* %dst, i64 22
; CHECK-NEXT: store i8 %[[test4_r2]], i8* %[[gep]]
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8, i8* %dst, i64 23
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test4_a2]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8, i8* %dst, i64 30
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [10 x i8], [10 x i8]* %[[test4_a3]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 10
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8, i8* %dst, i64 40
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: store i16 %[[test4_r5]], i16* %[[bitcast]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8, i8* %dst, i64 42
; CHECK-NEXT: store i8 %[[test4_r6]], i8* %[[gep]]
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8, i8* %dst, i64 43
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test4_a4]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8, i8* %dst, i64 50
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: store i16 %[[test4_r5]], i16* %[[bitcast]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8, i8* %dst, i64 52
; CHECK-NEXT: store i8 %[[test4_r6]], i8* %[[gep]]
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8, i8* %dst, i64 53
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8], [7 x i8]* %[[test4_a5]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8, i8* %dst, i64 60
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [40 x i8], [40 x i8]* %[[test4_a6]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 40

  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memcpy.p1i8.p0i8.i32(i8 addrspace(1)* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind

define i16 @test5() {
; CHECK-LABEL: @test5(
; CHECK-NOT: alloca float
; CHECK:      %[[cast:.*]] = bitcast float 0.0{{.*}} to i32
; CHECK-NEXT: %[[shr:.*]] = lshr i32 %[[cast]], 16
; CHECK-NEXT: %[[trunc:.*]] = trunc i32 %[[shr]] to i16
; CHECK-NEXT: ret i16 %[[trunc]]

entry:
  %a = alloca [4 x i8]
  %fptr = bitcast [4 x i8]* %a to float*
  store float 0.0, float* %fptr
  %ptr = getelementptr [4 x i8], [4 x i8]* %a, i32 0, i32 2
  %iptr = bitcast i8* %ptr to i16*
  %val = load i16, i16* %iptr
  ret i16 %val
}

define i32 @test6() {
; CHECK-LABEL: @test6(
; CHECK: alloca i32
; CHECK-NEXT: store volatile i32
; CHECK-NEXT: load i32, i32*
; CHECK-NEXT: ret i32

entry:
  %a = alloca [4 x i8]
  %ptr = getelementptr [4 x i8], [4 x i8]* %a, i32 0, i32 0
  call void @llvm.memset.p0i8.i32(i8* %ptr, i8 42, i32 4, i32 1, i1 true)
  %iptr = bitcast i8* %ptr to i32*
  %val = load i32, i32* %iptr
  ret i32 %val
}

define void @test7(i8* %src, i8* %dst) {
; CHECK-LABEL: @test7(
; CHECK: alloca i32
; CHECK-NEXT: bitcast i8* %src to i32*
; CHECK-NEXT: load volatile i32, i32*
; CHECK-NEXT: store volatile i32
; CHECK-NEXT: bitcast i8* %dst to i32*
; CHECK-NEXT: load volatile i32, i32*
; CHECK-NEXT: store volatile i32
; CHECK-NEXT: ret

entry:
  %a = alloca [4 x i8]
  %ptr = getelementptr [4 x i8], [4 x i8]* %a, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %ptr, i8* %src, i32 4, i32 1, i1 true)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %ptr, i32 4, i32 1, i1 true)
  ret void
}


%S1 = type { i32, i32, [16 x i8] }
%S2 = type { %S1*, %S2* }

define %S2 @test8(%S2* %s2) {
; CHECK-LABEL: @test8(
entry:
  %new = alloca %S2
; CHECK-NOT: alloca

  %s2.next.ptr = getelementptr %S2, %S2* %s2, i64 0, i32 1
  %s2.next = load %S2*, %S2** %s2.next.ptr
; CHECK:      %[[gep:.*]] = getelementptr %S2, %S2* %s2, i64 0, i32 1
; CHECK-NEXT: %[[next:.*]] = load %S2*, %S2** %[[gep]]

  %s2.next.s1.ptr = getelementptr %S2, %S2* %s2.next, i64 0, i32 0
  %s2.next.s1 = load %S1*, %S1** %s2.next.s1.ptr
  %new.s1.ptr = getelementptr %S2, %S2* %new, i64 0, i32 0
  store %S1* %s2.next.s1, %S1** %new.s1.ptr
  %s2.next.next.ptr = getelementptr %S2, %S2* %s2.next, i64 0, i32 1
  %s2.next.next = load %S2*, %S2** %s2.next.next.ptr
  %new.next.ptr = getelementptr %S2, %S2* %new, i64 0, i32 1
  store %S2* %s2.next.next, %S2** %new.next.ptr
; CHECK-NEXT: %[[gep:.*]] = getelementptr %S2, %S2* %[[next]], i64 0, i32 0
; CHECK-NEXT: %[[next_s1:.*]] = load %S1*, %S1** %[[gep]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr %S2, %S2* %[[next]], i64 0, i32 1
; CHECK-NEXT: %[[next_next:.*]] = load %S2*, %S2** %[[gep]]

  %new.s1 = load %S1*, %S1** %new.s1.ptr
  %result1 = insertvalue %S2 undef, %S1* %new.s1, 0
; CHECK-NEXT: %[[result1:.*]] = insertvalue %S2 undef, %S1* %[[next_s1]], 0
  %new.next = load %S2*, %S2** %new.next.ptr
  %result2 = insertvalue %S2 %result1, %S2* %new.next, 1
; CHECK-NEXT: %[[result2:.*]] = insertvalue %S2 %[[result1]], %S2* %[[next_next]], 1
  ret %S2 %result2
; CHECK-NEXT: ret %S2 %[[result2]]
}

define i64 @test9() {
; Ensure we can handle loads off the end of an alloca even when wrapped in
; weird bit casts and types. This is valid IR due to the alignment and masking
; off the bits past the end of the alloca.
;
; CHECK-LABEL: @test9(
; CHECK-NOT: alloca
; CHECK:      %[[b2:.*]] = zext i8 26 to i64
; CHECK-NEXT: %[[s2:.*]] = shl i64 %[[b2]], 16
; CHECK-NEXT: %[[m2:.*]] = and i64 undef, -16711681
; CHECK-NEXT: %[[i2:.*]] = or i64 %[[m2]], %[[s2]]
; CHECK-NEXT: %[[b1:.*]] = zext i8 0 to i64
; CHECK-NEXT: %[[s1:.*]] = shl i64 %[[b1]], 8
; CHECK-NEXT: %[[m1:.*]] = and i64 %[[i2]], -65281
; CHECK-NEXT: %[[i1:.*]] = or i64 %[[m1]], %[[s1]]
; CHECK-NEXT: %[[b0:.*]] = zext i8 0 to i64
; CHECK-NEXT: %[[m0:.*]] = and i64 %[[i1]], -256
; CHECK-NEXT: %[[i0:.*]] = or i64 %[[m0]], %[[b0]]
; CHECK-NEXT: %[[result:.*]] = and i64 %[[i0]], 16777215
; CHECK-NEXT: ret i64 %[[result]]

entry:
  %a = alloca { [3 x i8] }, align 8
  %gep1 = getelementptr inbounds { [3 x i8] }, { [3 x i8] }* %a, i32 0, i32 0, i32 0
  store i8 0, i8* %gep1, align 1
  %gep2 = getelementptr inbounds { [3 x i8] }, { [3 x i8] }* %a, i32 0, i32 0, i32 1
  store i8 0, i8* %gep2, align 1
  %gep3 = getelementptr inbounds { [3 x i8] }, { [3 x i8] }* %a, i32 0, i32 0, i32 2
  store i8 26, i8* %gep3, align 1
  %cast = bitcast { [3 x i8] }* %a to { i64 }*
  %elt = getelementptr inbounds { i64 }, { i64 }* %cast, i32 0, i32 0
  %load = load i64, i64* %elt
  %result = and i64 %load, 16777215
  ret i64 %result
}

define %S2* @test10() {
; CHECK-LABEL: @test10(
; CHECK-NOT: alloca %S2*
; CHECK: ret %S2* null

entry:
  %a = alloca [8 x i8]
  %ptr = getelementptr [8 x i8], [8 x i8]* %a, i32 0, i32 0
  call void @llvm.memset.p0i8.i32(i8* %ptr, i8 0, i32 8, i32 1, i1 false)
  %s2ptrptr = bitcast i8* %ptr to %S2**
  %s2ptr = load %S2*, %S2** %s2ptrptr
  ret %S2* %s2ptr
}

define i32 @test11() {
; CHECK-LABEL: @test11(
; CHECK-NOT: alloca
; CHECK: ret i32 0

entry:
  %X = alloca i32
  br i1 undef, label %good, label %bad

good:
  %Y = getelementptr i32, i32* %X, i64 0
  store i32 0, i32* %Y
  %Z = load i32, i32* %Y
  ret i32 %Z

bad:
  %Y2 = getelementptr i32, i32* %X, i64 1
  store i32 0, i32* %Y2
  %Z2 = load i32, i32* %Y2
  ret i32 %Z2
}

define i8 @test12() {
; We fully promote these to the i24 load or store size, resulting in just masks
; and other operations that instcombine will fold, but no alloca.
;
; CHECK-LABEL: @test12(

entry:
  %a = alloca [3 x i8]
  %b = alloca [3 x i8]
; CHECK-NOT: alloca

  %a0ptr = getelementptr [3 x i8], [3 x i8]* %a, i64 0, i32 0
  store i8 0, i8* %a0ptr
  %a1ptr = getelementptr [3 x i8], [3 x i8]* %a, i64 0, i32 1
  store i8 0, i8* %a1ptr
  %a2ptr = getelementptr [3 x i8], [3 x i8]* %a, i64 0, i32 2
  store i8 0, i8* %a2ptr
  %aiptr = bitcast [3 x i8]* %a to i24*
  %ai = load i24, i24* %aiptr
; CHECK-NOT: store
; CHECK-NOT: load
; CHECK:      %[[ext2:.*]] = zext i8 0 to i24
; CHECK-NEXT: %[[shift2:.*]] = shl i24 %[[ext2]], 16
; CHECK-NEXT: %[[mask2:.*]] = and i24 undef, 65535
; CHECK-NEXT: %[[insert2:.*]] = or i24 %[[mask2]], %[[shift2]]
; CHECK-NEXT: %[[ext1:.*]] = zext i8 0 to i24
; CHECK-NEXT: %[[shift1:.*]] = shl i24 %[[ext1]], 8
; CHECK-NEXT: %[[mask1:.*]] = and i24 %[[insert2]], -65281
; CHECK-NEXT: %[[insert1:.*]] = or i24 %[[mask1]], %[[shift1]]
; CHECK-NEXT: %[[ext0:.*]] = zext i8 0 to i24
; CHECK-NEXT: %[[mask0:.*]] = and i24 %[[insert1]], -256
; CHECK-NEXT: %[[insert0:.*]] = or i24 %[[mask0]], %[[ext0]]

  %biptr = bitcast [3 x i8]* %b to i24*
  store i24 %ai, i24* %biptr
  %b0ptr = getelementptr [3 x i8], [3 x i8]* %b, i64 0, i32 0
  %b0 = load i8, i8* %b0ptr
  %b1ptr = getelementptr [3 x i8], [3 x i8]* %b, i64 0, i32 1
  %b1 = load i8, i8* %b1ptr
  %b2ptr = getelementptr [3 x i8], [3 x i8]* %b, i64 0, i32 2
  %b2 = load i8, i8* %b2ptr
; CHECK-NOT: store
; CHECK-NOT: load
; CHECK:      %[[trunc0:.*]] = trunc i24 %[[insert0]] to i8
; CHECK-NEXT: %[[shift1:.*]] = lshr i24 %[[insert0]], 8
; CHECK-NEXT: %[[trunc1:.*]] = trunc i24 %[[shift1]] to i8
; CHECK-NEXT: %[[shift2:.*]] = lshr i24 %[[insert0]], 16
; CHECK-NEXT: %[[trunc2:.*]] = trunc i24 %[[shift2]] to i8

  %bsum0 = add i8 %b0, %b1
  %bsum1 = add i8 %bsum0, %b2
  ret i8 %bsum1
; CHECK:      %[[sum0:.*]] = add i8 %[[trunc0]], %[[trunc1]]
; CHECK-NEXT: %[[sum1:.*]] = add i8 %[[sum0]], %[[trunc2]]
; CHECK-NEXT: ret i8 %[[sum1]]
}

define i32 @test13() {
; Ensure we don't crash and handle undefined loads that straddle the end of the
; allocation.
; CHECK-LABEL: @test13(
; CHECK:      %[[value:.*]] = zext i8 0 to i16
; CHECK-NEXT: %[[ret:.*]] = zext i16 %[[value]] to i32
; CHECK-NEXT: ret i32 %[[ret]]

entry:
  %a = alloca [3 x i8], align 2
  %b0ptr = getelementptr [3 x i8], [3 x i8]* %a, i64 0, i32 0
  store i8 0, i8* %b0ptr
  %b1ptr = getelementptr [3 x i8], [3 x i8]* %a, i64 0, i32 1
  store i8 0, i8* %b1ptr
  %b2ptr = getelementptr [3 x i8], [3 x i8]* %a, i64 0, i32 2
  store i8 0, i8* %b2ptr
  %iptrcast = bitcast [3 x i8]* %a to i16*
  %iptrgep = getelementptr i16, i16* %iptrcast, i64 1
  %i = load i16, i16* %iptrgep
  %ret = zext i16 %i to i32
  ret i32 %ret
}

%test14.struct = type { [3 x i32] }

define void @test14(...) nounwind uwtable {
; This is a strange case where we split allocas into promotable partitions, but
; also gain enough data to prove they must be dead allocas due to GEPs that walk
; across two adjacent allocas. Test that we don't try to promote or otherwise
; do bad things to these dead allocas, they should just be removed.
; CHECK-LABEL: @test14(
; CHECK-NEXT: entry:
; CHECK-NEXT: ret void

entry:
  %a = alloca %test14.struct
  %p = alloca %test14.struct*
  %0 = bitcast %test14.struct* %a to i8*
  %1 = getelementptr i8, i8* %0, i64 12
  %2 = bitcast i8* %1 to %test14.struct*
  %3 = getelementptr inbounds %test14.struct, %test14.struct* %2, i32 0, i32 0
  %4 = getelementptr inbounds %test14.struct, %test14.struct* %a, i32 0, i32 0
  %5 = bitcast [3 x i32]* %3 to i32*
  %6 = bitcast [3 x i32]* %4 to i32*
  %7 = load i32, i32* %6, align 4
  store i32 %7, i32* %5, align 4
  %8 = getelementptr inbounds i32, i32* %5, i32 1
  %9 = getelementptr inbounds i32, i32* %6, i32 1
  %10 = load i32, i32* %9, align 4
  store i32 %10, i32* %8, align 4
  %11 = getelementptr inbounds i32, i32* %5, i32 2
  %12 = getelementptr inbounds i32, i32* %6, i32 2
  %13 = load i32, i32* %12, align 4
  store i32 %13, i32* %11, align 4
  ret void
}

define i32 @test15(i1 %flag) nounwind uwtable {
; Ensure that when there are dead instructions using an alloca that are not
; loads or stores we still delete them during partitioning and rewriting.
; Otherwise we'll go to promote them while thy still have unpromotable uses.
; CHECK-LABEL: @test15(
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %loop
; CHECK:      loop:
; CHECK-NEXT:   br label %loop

entry:
  %l0 = alloca i64
  %l1 = alloca i64
  %l2 = alloca i64
  %l3 = alloca i64
  br label %loop

loop:
  %dead3 = phi i8* [ %gep3, %loop ], [ null, %entry ]

  store i64 1879048192, i64* %l0, align 8
  %bc0 = bitcast i64* %l0 to i8*
  %gep0 = getelementptr i8, i8* %bc0, i64 3
  %dead0 = bitcast i8* %gep0 to i64*

  store i64 1879048192, i64* %l1, align 8
  %bc1 = bitcast i64* %l1 to i8*
  %gep1 = getelementptr i8, i8* %bc1, i64 3
  %dead1 = getelementptr i8, i8* %gep1, i64 1

  store i64 1879048192, i64* %l2, align 8
  %bc2 = bitcast i64* %l2 to i8*
  %gep2.1 = getelementptr i8, i8* %bc2, i64 1
  %gep2.2 = getelementptr i8, i8* %bc2, i64 3
  ; Note that this select should get visited multiple times due to using two
  ; different GEPs off the same alloca. We should only delete it once.
  %dead2 = select i1 %flag, i8* %gep2.1, i8* %gep2.2

  store i64 1879048192, i64* %l3, align 8
  %bc3 = bitcast i64* %l3 to i8*
  %gep3 = getelementptr i8, i8* %bc3, i64 3

  br label %loop
}

define void @test16(i8* %src, i8* %dst) {
; Ensure that we can promote an alloca of [3 x i8] to an i24 SSA value.
; CHECK-LABEL: @test16(
; CHECK-NOT: alloca
; CHECK:      %[[srccast:.*]] = bitcast i8* %src to i24*
; CHECK-NEXT: load i24, i24* %[[srccast]]
; CHECK-NEXT: %[[dstcast:.*]] = bitcast i8* %dst to i24*
; CHECK-NEXT: store i24 0, i24* %[[dstcast]]
; CHECK-NEXT: ret void

entry:
  %a = alloca [3 x i8]
  %ptr = getelementptr [3 x i8], [3 x i8]* %a, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %ptr, i8* %src, i32 4, i32 1, i1 false)
  %cast = bitcast i8* %ptr to i24*
  store i24 0, i24* %cast
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %ptr, i32 4, i32 1, i1 false)
  ret void
}

define void @test17(i8* %src, i8* %dst) {
; Ensure that we can rewrite unpromotable memcpys which extend past the end of
; the alloca.
; CHECK-LABEL: @test17(
; CHECK:      %[[a:.*]] = alloca [3 x i8]
; CHECK-NEXT: %[[ptr:.*]] = getelementptr [3 x i8], [3 x i8]* %[[a]], i32 0, i32 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[ptr]], i8* %src,
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %[[ptr]],
; CHECK-NEXT: ret void

entry:
  %a = alloca [3 x i8]
  %ptr = getelementptr [3 x i8], [3 x i8]* %a, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %ptr, i8* %src, i32 4, i32 1, i1 true)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %ptr, i32 4, i32 1, i1 true)
  ret void
}

define void @test18(i8* %src, i8* %dst, i32 %size) {
; Preserve transfer instrinsics with a variable size, even if they overlap with
; fixed size operations. Further, continue to split and promote allocas preceding
; the variable sized intrinsic.
; CHECK-LABEL: @test18(
; CHECK:      %[[a:.*]] = alloca [34 x i8]
; CHECK:      %[[srcgep1:.*]] = getelementptr inbounds i8, i8* %src, i64 4
; CHECK-NEXT: %[[srccast1:.*]] = bitcast i8* %[[srcgep1]] to i32*
; CHECK-NEXT: %[[srcload:.*]] = load i32, i32* %[[srccast1]]
; CHECK-NEXT: %[[agep1:.*]] = getelementptr inbounds [34 x i8], [34 x i8]* %[[a]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[agep1]], i8* %src, i32 %size,
; CHECK-NEXT: %[[agep2:.*]] = getelementptr inbounds [34 x i8], [34 x i8]* %[[a]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memset.p0i8.i32(i8* %[[agep2]], i8 42, i32 %size,
; CHECK-NEXT: %[[dstcast1:.*]] = bitcast i8* %dst to i32*
; CHECK-NEXT: store i32 42, i32* %[[dstcast1]]
; CHECK-NEXT: %[[dstgep1:.*]] = getelementptr inbounds i8, i8* %dst, i64 4
; CHECK-NEXT: %[[dstcast2:.*]] = bitcast i8* %[[dstgep1]] to i32*
; CHECK-NEXT: store i32 %[[srcload]], i32* %[[dstcast2]]
; CHECK-NEXT: %[[agep3:.*]] = getelementptr inbounds [34 x i8], [34 x i8]* %[[a]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %[[agep3]], i32 %size,
; CHECK-NEXT: ret void

entry:
  %a = alloca [42 x i8]
  %ptr = getelementptr [42 x i8], [42 x i8]* %a, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %ptr, i8* %src, i32 8, i32 1, i1 false)
  %ptr2 = getelementptr [42 x i8], [42 x i8]* %a, i32 0, i32 8
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %ptr2, i8* %src, i32 %size, i32 1, i1 false)
  call void @llvm.memset.p0i8.i32(i8* %ptr2, i8 42, i32 %size, i32 1, i1 false)
  %cast = bitcast i8* %ptr to i32*
  store i32 42, i32* %cast
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %ptr, i32 8, i32 1, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %ptr2, i32 %size, i32 1, i1 false)
  ret void
}

%opaque = type opaque

define i32 @test19(%opaque* %x) {
; This input will cause us to try to compute a natural GEP when rewriting
; pointers in such a way that we try to GEP through the opaque type. Previously,
; a check for an unsized type was missing and this crashed. Ensure it behaves
; reasonably now.
; CHECK-LABEL: @test19(
; CHECK-NOT: alloca
; CHECK: ret i32 undef

entry:
  %a = alloca { i64, i8* }
  %cast1 = bitcast %opaque* %x to i8*
  %cast2 = bitcast { i64, i8* }* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %cast2, i8* %cast1, i32 16, i32 1, i1 false)
  %gep = getelementptr inbounds { i64, i8* }, { i64, i8* }* %a, i32 0, i32 0
  %val = load i64, i64* %gep
  ret i32 undef
}

define i32 @test20() {
; Ensure we can track negative offsets (before the beginning of the alloca) and
; negative relative offsets from offsets starting past the end of the alloca.
; CHECK-LABEL: @test20(
; CHECK-NOT: alloca
; CHECK: %[[sum1:.*]] = add i32 1, 2
; CHECK: %[[sum2:.*]] = add i32 %[[sum1]], 3
; CHECK: ret i32 %[[sum2]]

entry:
  %a = alloca [3 x i32]
  %gep1 = getelementptr [3 x i32], [3 x i32]* %a, i32 0, i32 0
  store i32 1, i32* %gep1
  %gep2.1 = getelementptr [3 x i32], [3 x i32]* %a, i32 0, i32 -2
  %gep2.2 = getelementptr i32, i32* %gep2.1, i32 3
  store i32 2, i32* %gep2.2
  %gep3.1 = getelementptr [3 x i32], [3 x i32]* %a, i32 0, i32 14
  %gep3.2 = getelementptr i32, i32* %gep3.1, i32 -12
  store i32 3, i32* %gep3.2

  %load1 = load i32, i32* %gep1
  %load2 = load i32, i32* %gep2.2
  %load3 = load i32, i32* %gep3.2
  %sum1 = add i32 %load1, %load2
  %sum2 = add i32 %sum1, %load3
  ret i32 %sum2
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind

define i8 @test21() {
; Test allocations and offsets which border on overflow of the int64_t used
; internally. This is really awkward to really test as LLVM doesn't really
; support such extreme constructs cleanly.
; CHECK-LABEL: @test21(
; CHECK-NOT: alloca
; CHECK: or i8 -1, -1

entry:
  %a = alloca [2305843009213693951 x i8]
  %gep0 = getelementptr [2305843009213693951 x i8], [2305843009213693951 x i8]* %a, i64 0, i64 2305843009213693949
  store i8 255, i8* %gep0
  %gep1 = getelementptr [2305843009213693951 x i8], [2305843009213693951 x i8]* %a, i64 0, i64 -9223372036854775807
  %gep2 = getelementptr i8, i8* %gep1, i64 -1
  call void @llvm.memset.p0i8.i64(i8* %gep2, i8 0, i64 18446744073709551615, i32 1, i1 false)
  %gep3 = getelementptr i8, i8* %gep1, i64 9223372036854775807
  %gep4 = getelementptr i8, i8* %gep3, i64 9223372036854775807
  %gep5 = getelementptr i8, i8* %gep4, i64 -6917529027641081857
  store i8 255, i8* %gep5
  %cast1 = bitcast i8* %gep4 to i32*
  store i32 0, i32* %cast1
  %load = load i8, i8* %gep0
  %gep6 = getelementptr i8, i8* %gep0, i32 1
  %load2 = load i8, i8* %gep6
  %result = or i8 %load, %load2
  ret i8 %result
}

%PR13916.struct = type { i8 }

define void @PR13916.1() {
; Ensure that we handle overlapping memcpy intrinsics correctly, especially in
; the case where there is a directly identical value for both source and dest.
; CHECK: @PR13916.1
; CHECK-NOT: alloca
; CHECK: ret void

entry:
  %a = alloca i8
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a, i8* %a, i32 1, i32 1, i1 false)
  %tmp2 = load i8, i8* %a
  ret void
}

define void @PR13916.2() {
; Check whether we continue to handle them correctly when they start off with
; different pointer value chains, but during rewriting we coalesce them into the
; same value.
; CHECK: @PR13916.2
; CHECK-NOT: alloca
; CHECK: ret void

entry:
  %a = alloca %PR13916.struct, align 1
  br i1 undef, label %if.then, label %if.end

if.then:
  %tmp0 = bitcast %PR13916.struct* %a to i8*
  %tmp1 = bitcast %PR13916.struct* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp0, i8* %tmp1, i32 1, i32 1, i1 false)
  br label %if.end

if.end:
  %gep = getelementptr %PR13916.struct, %PR13916.struct* %a, i32 0, i32 0
  %tmp2 = load i8, i8* %gep
  ret void
}

define void @PR13990() {
; Ensure we can handle cases where processing one alloca causes the other
; alloca to become dead and get deleted. This might crash or fail under
; Valgrind if we regress.
; CHECK-LABEL: @PR13990(
; CHECK-NOT: alloca
; CHECK: unreachable
; CHECK: unreachable

entry:
  %tmp1 = alloca i8*
  %tmp2 = alloca i8*
  br i1 undef, label %bb1, label %bb2

bb1:
  store i8* undef, i8** %tmp2
  br i1 undef, label %bb2, label %bb3

bb2:
  %tmp50 = select i1 undef, i8** %tmp2, i8** %tmp1
  br i1 undef, label %bb3, label %bb4

bb3:
  unreachable

bb4:
  unreachable
}

define double @PR13969(double %x) {
; Check that we detect when promotion will un-escape an alloca and iterate to
; re-try running SROA over that alloca. Without that, the two allocas that are
; stored into a dead alloca don't get rewritten and promoted.
; CHECK-LABEL: @PR13969(

entry:
  %a = alloca double
  %b = alloca double*
  %c = alloca double
; CHECK-NOT: alloca

  store double %x, double* %a
  store double* %c, double** %b
  store double* %a, double** %b
  store double %x, double* %c
  %ret = load double, double* %a
; CHECK-NOT: store
; CHECK-NOT: load

  ret double %ret
; CHECK: ret double %x
}

%PR14034.struct = type { { {} }, i32, %PR14034.list }
%PR14034.list = type { %PR14034.list*, %PR14034.list* }

define void @PR14034() {
; This test case tries to form GEPs into the empty leading struct members, and
; subsequently crashed (under valgrind) before we fixed the PR. The important
; thing is to handle empty structs gracefully.
; CHECK-LABEL: @PR14034(

entry:
  %a = alloca %PR14034.struct
  %list = getelementptr %PR14034.struct, %PR14034.struct* %a, i32 0, i32 2
  %prev = getelementptr %PR14034.list, %PR14034.list* %list, i32 0, i32 1
  store %PR14034.list* undef, %PR14034.list** %prev
  %cast0 = bitcast %PR14034.struct* undef to i8*
  %cast1 = bitcast %PR14034.struct* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %cast0, i8* %cast1, i32 12, i32 0, i1 false)
  ret void
}

define i32 @test22(i32 %x) {
; Test that SROA and promotion is not confused by a grab bax mixture of pointer
; types involving wrapper aggregates and zero-length aggregate members.
; CHECK-LABEL: @test22(

entry:
  %a1 = alloca { { [1 x { i32 }] } }
  %a2 = alloca { {}, { float }, [0 x i8] }
  %a3 = alloca { [0 x i8], { [0 x double], [1 x [1 x <4 x i8>]], {} }, { { {} } } }
; CHECK-NOT: alloca

  %wrap1 = insertvalue [1 x { i32 }] undef, i32 %x, 0, 0
  %gep1 = getelementptr { { [1 x { i32 }] } }, { { [1 x { i32 }] } }* %a1, i32 0, i32 0, i32 0
  store [1 x { i32 }] %wrap1, [1 x { i32 }]* %gep1

  %gep2 = getelementptr { { [1 x { i32 }] } }, { { [1 x { i32 }] } }* %a1, i32 0, i32 0
  %ptrcast1 = bitcast { [1 x { i32 }] }* %gep2 to { [1 x { float }] }*
  %load1 = load { [1 x { float }] }, { [1 x { float }] }* %ptrcast1
  %unwrap1 = extractvalue { [1 x { float }] } %load1, 0, 0

  %wrap2 = insertvalue { {}, { float }, [0 x i8] } undef, { float } %unwrap1, 1
  store { {}, { float }, [0 x i8] } %wrap2, { {}, { float }, [0 x i8] }* %a2

  %gep3 = getelementptr { {}, { float }, [0 x i8] }, { {}, { float }, [0 x i8] }* %a2, i32 0, i32 1, i32 0
  %ptrcast2 = bitcast float* %gep3 to <4 x i8>*
  %load3 = load <4 x i8>, <4 x i8>* %ptrcast2
  %valcast1 = bitcast <4 x i8> %load3 to i32

  %wrap3 = insertvalue [1 x [1 x i32]] undef, i32 %valcast1, 0, 0
  %wrap4 = insertvalue { [1 x [1 x i32]], {} } undef, [1 x [1 x i32]] %wrap3, 0
  %gep4 = getelementptr { [0 x i8], { [0 x double], [1 x [1 x <4 x i8>]], {} }, { { {} } } }, { [0 x i8], { [0 x double], [1 x [1 x <4 x i8>]], {} }, { { {} } } }* %a3, i32 0, i32 1
  %ptrcast3 = bitcast { [0 x double], [1 x [1 x <4 x i8>]], {} }* %gep4 to { [1 x [1 x i32]], {} }*
  store { [1 x [1 x i32]], {} } %wrap4, { [1 x [1 x i32]], {} }* %ptrcast3

  %gep5 = getelementptr { [0 x i8], { [0 x double], [1 x [1 x <4 x i8>]], {} }, { { {} } } }, { [0 x i8], { [0 x double], [1 x [1 x <4 x i8>]], {} }, { { {} } } }* %a3, i32 0, i32 1, i32 1, i32 0
  %ptrcast4 = bitcast [1 x <4 x i8>]* %gep5 to { {}, float, {} }*
  %load4 = load { {}, float, {} }, { {}, float, {} }* %ptrcast4
  %unwrap2 = extractvalue { {}, float, {} } %load4, 1
  %valcast2 = bitcast float %unwrap2 to i32

  ret i32 %valcast2
; CHECK: ret i32
}

define void @PR14059.1(double* %d) {
; In PR14059 a peculiar construct was identified as something that is used
; pervasively in ARM's ABI-calling-convention lowering: the passing of a struct
; of doubles via an array of i32 in order to place the data into integer
; registers. This in turn was missed as an optimization by SROA due to the
; partial loads and stores of integers to the double alloca we were trying to
; form and promote. The solution is to widen the integer operations to be
; whole-alloca operations, and perform the appropriate bitcasting on the
; *values* rather than the pointers. When this works, partial reads and writes
; via integers can be promoted away.
; CHECK: @PR14059.1
; CHECK-NOT: alloca
; CHECK: ret void

entry:
  %X.sroa.0.i = alloca double, align 8
  %0 = bitcast double* %X.sroa.0.i to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %0)

  ; Store to the low 32-bits...
  %X.sroa.0.0.cast2.i = bitcast double* %X.sroa.0.i to i32*
  store i32 0, i32* %X.sroa.0.0.cast2.i, align 8

  ; Also use a memset to the middle 32-bits for fun.
  %X.sroa.0.2.raw_idx2.i = getelementptr inbounds i8, i8* %0, i32 2
  call void @llvm.memset.p0i8.i64(i8* %X.sroa.0.2.raw_idx2.i, i8 0, i64 4, i32 1, i1 false)

  ; Or a memset of the whole thing.
  call void @llvm.memset.p0i8.i64(i8* %0, i8 0, i64 8, i32 1, i1 false)

  ; Write to the high 32-bits with a memcpy.
  %X.sroa.0.4.raw_idx4.i = getelementptr inbounds i8, i8* %0, i32 4
  %d.raw = bitcast double* %d to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %X.sroa.0.4.raw_idx4.i, i8* %d.raw, i32 4, i32 1, i1 false)

  ; Store to the high 32-bits...
  %X.sroa.0.4.cast5.i = bitcast i8* %X.sroa.0.4.raw_idx4.i to i32*
  store i32 1072693248, i32* %X.sroa.0.4.cast5.i, align 4

  ; Do the actual math...
  %X.sroa.0.0.load1.i = load double, double* %X.sroa.0.i, align 8
  %accum.real.i = load double, double* %d, align 8
  %add.r.i = fadd double %accum.real.i, %X.sroa.0.0.load1.i
  store double %add.r.i, double* %d, align 8
  call void @llvm.lifetime.end(i64 -1, i8* %0)
  ret void
}

define i64 @PR14059.2({ float, float }* %phi) {
; Check that SROA can split up alloca-wide integer loads and stores where the
; underlying alloca has smaller components that are accessed independently. This
; shows up particularly with ABI lowering patterns coming out of Clang that rely
; on the particular register placement of a single large integer return value.
; CHECK: @PR14059.2

entry:
  %retval = alloca { float, float }, align 4
  ; CHECK-NOT: alloca

  %0 = bitcast { float, float }* %retval to i64*
  store i64 0, i64* %0
  ; CHECK-NOT: store

  %phi.realp = getelementptr inbounds { float, float }, { float, float }* %phi, i32 0, i32 0
  %phi.real = load float, float* %phi.realp
  %phi.imagp = getelementptr inbounds { float, float }, { float, float }* %phi, i32 0, i32 1
  %phi.imag = load float, float* %phi.imagp
  ; CHECK:      %[[realp:.*]] = getelementptr inbounds { float, float }, { float, float }* %phi, i32 0, i32 0
  ; CHECK-NEXT: %[[real:.*]] = load float, float* %[[realp]]
  ; CHECK-NEXT: %[[imagp:.*]] = getelementptr inbounds { float, float }, { float, float }* %phi, i32 0, i32 1
  ; CHECK-NEXT: %[[imag:.*]] = load float, float* %[[imagp]]

  %real = getelementptr inbounds { float, float }, { float, float }* %retval, i32 0, i32 0
  %imag = getelementptr inbounds { float, float }, { float, float }* %retval, i32 0, i32 1
  store float %phi.real, float* %real
  store float %phi.imag, float* %imag
  ; CHECK-NEXT: %[[real_convert:.*]] = bitcast float %[[real]] to i32
  ; CHECK-NEXT: %[[imag_convert:.*]] = bitcast float %[[imag]] to i32
  ; CHECK-NEXT: %[[imag_ext:.*]] = zext i32 %[[imag_convert]] to i64
  ; CHECK-NEXT: %[[imag_shift:.*]] = shl i64 %[[imag_ext]], 32
  ; CHECK-NEXT: %[[imag_mask:.*]] = and i64 undef, 4294967295
  ; CHECK-NEXT: %[[imag_insert:.*]] = or i64 %[[imag_mask]], %[[imag_shift]]
  ; CHECK-NEXT: %[[real_ext:.*]] = zext i32 %[[real_convert]] to i64
  ; CHECK-NEXT: %[[real_mask:.*]] = and i64 %[[imag_insert]], -4294967296
  ; CHECK-NEXT: %[[real_insert:.*]] = or i64 %[[real_mask]], %[[real_ext]]

  %1 = load i64, i64* %0, align 1
  ret i64 %1
  ; CHECK-NEXT: ret i64 %[[real_insert]]
}

define void @PR14105({ [16 x i8] }* %ptr) {
; Ensure that when rewriting the GEP index '-1' for this alloca we preserve is
; sign as negative. We use a volatile memcpy to ensure promotion never actually
; occurs.
; CHECK-LABEL: @PR14105(

entry:
  %a = alloca { [16 x i8] }, align 8
; CHECK: alloca [16 x i8], align 8

  %gep = getelementptr inbounds { [16 x i8] }, { [16 x i8] }* %ptr, i64 -1
; CHECK-NEXT: getelementptr inbounds { [16 x i8] }, { [16 x i8] }* %ptr, i64 -1, i32 0, i64 0

  %cast1 = bitcast { [16 x i8 ] }* %gep to i8*
  %cast2 = bitcast { [16 x i8 ] }* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %cast1, i8* %cast2, i32 16, i32 8, i1 true)
  ret void
; CHECK: ret
}

define void @PR14105_as1({ [16 x i8] } addrspace(1)* %ptr) {
; Make sure this the right address space pointer is used for type check.
; CHECK-LABEL: @PR14105_as1(

entry:
  %a = alloca { [16 x i8] }, align 8
; CHECK: alloca [16 x i8], align 8

  %gep = getelementptr inbounds { [16 x i8] }, { [16 x i8] } addrspace(1)* %ptr, i64 -1
; CHECK-NEXT: getelementptr inbounds { [16 x i8] }, { [16 x i8] } addrspace(1)* %ptr, i16 -1, i32 0, i16 0

  %cast1 = bitcast { [16 x i8 ] } addrspace(1)* %gep to i8 addrspace(1)*
  %cast2 = bitcast { [16 x i8 ] }* %a to i8*
  call void @llvm.memcpy.p1i8.p0i8.i32(i8 addrspace(1)* %cast1, i8* %cast2, i32 16, i32 8, i1 true)
  ret void
; CHECK: ret
}

define void @PR14465() {
; Ensure that we don't crash when analyzing a alloca larger than the maximum
; integer type width (MAX_INT_BITS) supported by llvm (1048576*32 > (1<<23)-1).
; CHECK-LABEL: @PR14465(

  %stack = alloca [1048576 x i32], align 16
; CHECK: alloca [1048576 x i32]
  %cast = bitcast [1048576 x i32]* %stack to i8*
  call void @llvm.memset.p0i8.i64(i8* %cast, i8 -2, i64 4194304, i32 16, i1 false)
  ret void
; CHECK: ret
}

define void @PR14548(i1 %x) {
; Handle a mixture of i1 and i8 loads and stores to allocas. This particular
; pattern caused crashes and invalid output in the PR, and its nature will
; trigger a mixture in several permutations as we resolve each alloca
; iteratively.
; Note that we don't do a particularly good *job* of handling these mixtures,
; but the hope is that this is very rare.
; CHECK-LABEL: @PR14548(

entry:
  %a = alloca <{ i1 }>, align 8
  %b = alloca <{ i1 }>, align 8
; CHECK:      %[[a:.*]] = alloca i8, align 8
; CHECK-NEXT: %[[b:.*]] = alloca i8, align 8

  %b.i1 = bitcast <{ i1 }>* %b to i1*
  store i1 %x, i1* %b.i1, align 8
  %b.i8 = bitcast <{ i1 }>* %b to i8*
  %foo = load i8, i8* %b.i8, align 1
; CHECK-NEXT: %[[b_cast:.*]] = bitcast i8* %[[b]] to i1*
; CHECK-NEXT: store i1 %x, i1* %[[b_cast]], align 8
; CHECK-NEXT: {{.*}} = load i8, i8* %[[b]], align 8

  %a.i8 = bitcast <{ i1 }>* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a.i8, i8* %b.i8, i32 1, i32 1, i1 false) nounwind
  %bar = load i8, i8* %a.i8, align 1
  %a.i1 = getelementptr inbounds <{ i1 }>, <{ i1 }>* %a, i32 0, i32 0
  %baz = load i1, i1* %a.i1, align 1
; CHECK-NEXT: %[[copy:.*]] = load i8, i8* %[[b]], align 8
; CHECK-NEXT: store i8 %[[copy]], i8* %[[a]], align 8
; CHECK-NEXT: {{.*}} = load i8, i8* %[[a]], align 8
; CHECK-NEXT: %[[a_cast:.*]] = bitcast i8* %[[a]] to i1*
; CHECK-NEXT: {{.*}} = load i1, i1* %[[a_cast]], align 8

  ret void
}

define <3 x i8> @PR14572.1(i32 %x) {
; Ensure that a split integer store which is wider than the type size of the
; alloca (relying on the alloc size padding) doesn't trigger an assert.
; CHECK: @PR14572.1

entry:
  %a = alloca <3 x i8>, align 4
; CHECK-NOT: alloca

  %cast = bitcast <3 x i8>* %a to i32*
  store i32 %x, i32* %cast, align 1
  %y = load <3 x i8>, <3 x i8>* %a, align 4
  ret <3 x i8> %y
; CHECK: ret <3 x i8>
}

define i32 @PR14572.2(<3 x i8> %x) {
; Ensure that a split integer load which is wider than the type size of the
; alloca (relying on the alloc size padding) doesn't trigger an assert.
; CHECK: @PR14572.2

entry:
  %a = alloca <3 x i8>, align 4
; CHECK-NOT: alloca

  store <3 x i8> %x, <3 x i8>* %a, align 1
  %cast = bitcast <3 x i8>* %a to i32*
  %y = load i32, i32* %cast, align 4
  ret i32 %y
; CHECK: ret i32
}

define i32 @PR14601(i32 %x) {
; Don't try to form a promotable integer alloca when there is a variable length
; memory intrinsic.
; CHECK-LABEL: @PR14601(

entry:
  %a = alloca i32
; CHECK: alloca

  %a.i8 = bitcast i32* %a to i8*
  call void @llvm.memset.p0i8.i32(i8* %a.i8, i8 0, i32 %x, i32 1, i1 false)
  %v = load i32, i32* %a
  ret i32 %v
}

define void @PR15674(i8* %data, i8* %src, i32 %size) {
; Arrange (via control flow) to have unmerged stores of a particular width to
; an alloca where we incrementally store from the end of the array toward the
; beginning of the array. Ensure that the final integer store, despite being
; convertable to the integer type that we end up promoting this alloca toward,
; doesn't get widened to a full alloca store.
; CHECK-LABEL: @PR15674(

entry:
  %tmp = alloca [4 x i8], align 1
; CHECK: alloca i32

  switch i32 %size, label %end [
    i32 4, label %bb4
    i32 3, label %bb3
    i32 2, label %bb2
    i32 1, label %bb1
  ]

bb4:
  %src.gep3 = getelementptr inbounds i8, i8* %src, i32 3
  %src.3 = load i8, i8* %src.gep3
  %tmp.gep3 = getelementptr inbounds [4 x i8], [4 x i8]* %tmp, i32 0, i32 3
  store i8 %src.3, i8* %tmp.gep3
; CHECK: store i8

  br label %bb3

bb3:
  %src.gep2 = getelementptr inbounds i8, i8* %src, i32 2
  %src.2 = load i8, i8* %src.gep2
  %tmp.gep2 = getelementptr inbounds [4 x i8], [4 x i8]* %tmp, i32 0, i32 2
  store i8 %src.2, i8* %tmp.gep2
; CHECK: store i8

  br label %bb2

bb2:
  %src.gep1 = getelementptr inbounds i8, i8* %src, i32 1
  %src.1 = load i8, i8* %src.gep1
  %tmp.gep1 = getelementptr inbounds [4 x i8], [4 x i8]* %tmp, i32 0, i32 1
  store i8 %src.1, i8* %tmp.gep1
; CHECK: store i8

  br label %bb1

bb1:
  %src.gep0 = getelementptr inbounds i8, i8* %src, i32 0
  %src.0 = load i8, i8* %src.gep0
  %tmp.gep0 = getelementptr inbounds [4 x i8], [4 x i8]* %tmp, i32 0, i32 0
  store i8 %src.0, i8* %tmp.gep0
; CHECK: store i8

  br label %end

end:
  %tmp.raw = bitcast [4 x i8]* %tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %data, i8* %tmp.raw, i32 %size, i32 1, i1 false)
  ret void
; CHECK: ret void
}

define void @PR15805(i1 %a, i1 %b) {
; CHECK-LABEL: @PR15805(
; CHECK-NOT: alloca
; CHECK: ret void

  %c = alloca i64, align 8
  %p.0.c = select i1 undef, i64* %c, i64* %c
  %cond.in = select i1 undef, i64* %p.0.c, i64* %c
  %cond = load i64, i64* %cond.in, align 8
  ret void
}

define void @PR15805.1(i1 %a, i1 %b) {
; Same as the normal PR15805, but rigged to place the use before the def inside
; of looping unreachable code. This helps ensure that we aren't sensitive to the
; order in which the uses of the alloca are visited.
;
; CHECK-LABEL: @PR15805.1(
; CHECK-NOT: alloca
; CHECK: ret void

  %c = alloca i64, align 8
  br label %exit

loop:
  %cond.in = select i1 undef, i64* %c, i64* %p.0.c
  %p.0.c = select i1 undef, i64* %c, i64* %c
  %cond = load i64, i64* %cond.in, align 8
  br i1 undef, label %loop, label %exit

exit:
  ret void
}

define void @PR16651.1(i8* %a) {
; This test case caused a crash due to the volatile memcpy in combination with
; lowering to integer loads and stores of a width other than that of the original
; memcpy.
;
; CHECK-LABEL: @PR16651.1(
; CHECK: alloca i16
; CHECK: alloca i8
; CHECK: alloca i8
; CHECK: unreachable

entry:
  %b = alloca i32, align 4
  %b.cast = bitcast i32* %b to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %b.cast, i8* %a, i32 4, i32 4, i1 true)
  %b.gep = getelementptr inbounds i8, i8* %b.cast, i32 2
  load i8, i8* %b.gep, align 2
  unreachable
}

define void @PR16651.2() {
; This test case caused a crash due to failing to promote given a select that
; can't be speculated. It shouldn't be promoted, but we missed that fact when
; analyzing whether we could form a vector promotion because that code didn't
; bail on select instructions.
;
; CHECK-LABEL: @PR16651.2(
; CHECK: alloca <2 x float>
; CHECK: ret void

entry:
  %tv1 = alloca { <2 x float>, <2 x float> }, align 8
  %0 = getelementptr { <2 x float>, <2 x float> }, { <2 x float>, <2 x float> }* %tv1, i64 0, i32 1
  store <2 x float> undef, <2 x float>* %0, align 8
  %1 = getelementptr inbounds { <2 x float>, <2 x float> }, { <2 x float>, <2 x float> }* %tv1, i64 0, i32 1, i64 0
  %cond105.in.i.i = select i1 undef, float* null, float* %1
  %cond105.i.i = load float, float* %cond105.in.i.i, align 8
  ret void
}

define void @test23(i32 %x) {
; CHECK-LABEL: @test23(
; CHECK-NOT: alloca
; CHECK: ret void
entry:
  %a = alloca i32, align 4
  store i32 %x, i32* %a, align 4
  %gep1 = getelementptr inbounds i32, i32* %a, i32 1
  %gep0 = getelementptr inbounds i32, i32* %a, i32 0
  %cast1 = bitcast i32* %gep1 to i8*
  %cast0 = bitcast i32* %gep0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %cast1, i8* %cast0, i32 4, i32 1, i1 false)
  ret void
}

define void @PR18615() {
; CHECK-LABEL: @PR18615(
; CHECK-NOT: alloca
; CHECK: ret void
entry:
  %f = alloca i8
  %gep = getelementptr i8, i8* %f, i64 -1
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* undef, i8* %gep, i32 1, i32 1, i1 false)
  ret void
}

define void @test24(i8* %src, i8* %dst) {
; CHECK-LABEL: @test24(
; CHECK: alloca i64, align 16
; CHECK: load volatile i64, i64* %{{[^,]*}}, align 1
; CHECK: store volatile i64 %{{[^,]*}}, i64* %{{[^,]*}}, align 16
; CHECK: load volatile i64, i64* %{{[^,]*}}, align 16
; CHECK: store volatile i64 %{{[^,]*}}, i64* %{{[^,]*}}, align 1

entry:
  %a = alloca i64, align 16
  %ptr = bitcast i64* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %ptr, i8* %src, i32 8, i32 1, i1 true)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %ptr, i32 8, i32 1, i1 true)
  ret void
}

define float @test25() {
; Check that we split up stores in order to promote the smaller SSA values.. These types
; of patterns can arise because LLVM maps small memcpy's to integer load and
; stores. If we get a memcpy of an aggregate (such as C and C++ frontends would
; produce, but so might any language frontend), this will in many cases turn into
; an integer load and store. SROA needs to be extremely powerful to correctly
; handle these cases and form splitable and promotable SSA values.
;
; CHECK-LABEL: @test25(
; CHECK-NOT: alloca
; CHECK: %[[F1:.*]] = bitcast i32 0 to float
; CHECK: %[[F2:.*]] = bitcast i32 1065353216 to float
; CHECK: %[[SUM:.*]] = fadd float %[[F1]], %[[F2]]
; CHECK: ret float %[[SUM]]

entry:
  %a = alloca i64
  %b = alloca i64
  %a.cast = bitcast i64* %a to [2 x float]*
  %a.gep1 = getelementptr [2 x float], [2 x float]* %a.cast, i32 0, i32 0
  %a.gep2 = getelementptr [2 x float], [2 x float]* %a.cast, i32 0, i32 1
  %b.cast = bitcast i64* %b to [2 x float]*
  %b.gep1 = getelementptr [2 x float], [2 x float]* %b.cast, i32 0, i32 0
  %b.gep2 = getelementptr [2 x float], [2 x float]* %b.cast, i32 0, i32 1
  store float 0.0, float* %a.gep1
  store float 1.0, float* %a.gep2
  %v = load i64, i64* %a
  store i64 %v, i64* %b
  %f1 = load float, float* %b.gep1
  %f2 = load float, float* %b.gep2
  %ret = fadd float %f1, %f2
  ret float %ret
}

@complex1 = external global [2 x float]
@complex2 = external global [2 x float]

define void @test26() {
; Test a case of splitting up loads and stores against a globals.
;
; CHECK-LABEL: @test26(
; CHECK-NOT: alloca
; CHECK: %[[L1:.*]] = load i32, i32* bitcast
; CHECK: %[[L2:.*]] = load i32, i32* bitcast
; CHECK: %[[F1:.*]] = bitcast i32 %[[L1]] to float
; CHECK: %[[F2:.*]] = bitcast i32 %[[L2]] to float
; CHECK: %[[SUM:.*]] = fadd float %[[F1]], %[[F2]]
; CHECK: %[[C1:.*]] = bitcast float %[[SUM]] to i32
; CHECK: %[[C2:.*]] = bitcast float %[[SUM]] to i32
; CHECK: store i32 %[[C1]], i32* bitcast
; CHECK: store i32 %[[C2]], i32* bitcast
; CHECK: ret void

entry:
  %a = alloca i64
  %a.cast = bitcast i64* %a to [2 x float]*
  %a.gep1 = getelementptr [2 x float], [2 x float]* %a.cast, i32 0, i32 0
  %a.gep2 = getelementptr [2 x float], [2 x float]* %a.cast, i32 0, i32 1
  %v1 = load i64, i64* bitcast ([2 x float]* @complex1 to i64*)
  store i64 %v1, i64* %a
  %f1 = load float, float* %a.gep1
  %f2 = load float, float* %a.gep2
  %sum = fadd float %f1, %f2
  store float %sum, float* %a.gep1
  store float %sum, float* %a.gep2
  %v2 = load i64, i64* %a
  store i64 %v2, i64* bitcast ([2 x float]* @complex2 to i64*)
  ret void
}

define float @test27() {
; Another, more complex case of splittable i64 loads and stores. This example
; is a particularly challenging one because the load and store both point into
; the alloca SROA is processing, and they overlap but at an offset.
;
; CHECK-LABEL: @test27(
; CHECK-NOT: alloca
; CHECK: %[[F1:.*]] = bitcast i32 0 to float
; CHECK: %[[F2:.*]] = bitcast i32 1065353216 to float
; CHECK: %[[SUM:.*]] = fadd float %[[F1]], %[[F2]]
; CHECK: ret float %[[SUM]]

entry:
  %a = alloca [12 x i8]
  %gep1 = getelementptr [12 x i8], [12 x i8]* %a, i32 0, i32 0
  %gep2 = getelementptr [12 x i8], [12 x i8]* %a, i32 0, i32 4
  %gep3 = getelementptr [12 x i8], [12 x i8]* %a, i32 0, i32 8
  %iptr1 = bitcast i8* %gep1 to i64*
  %iptr2 = bitcast i8* %gep2 to i64*
  %fptr1 = bitcast i8* %gep1 to float*
  %fptr2 = bitcast i8* %gep2 to float*
  %fptr3 = bitcast i8* %gep3 to float*
  store float 0.0, float* %fptr1
  store float 1.0, float* %fptr2
  %v = load i64, i64* %iptr1
  store i64 %v, i64* %iptr2
  %f1 = load float, float* %fptr2
  %f2 = load float, float* %fptr3
  %ret = fadd float %f1, %f2
  ret float %ret
}

define i32 @PR22093() {
; Test that we don't try to pre-split a splittable store of a splittable but
; not pre-splittable load over the same alloca. We "handle" this case when the
; load is unsplittable but unrelated to this alloca by just generating extra
; loads without touching the original, but when the original load was out of
; this alloca we need to handle it specially to ensure the splits line up
; properly for rewriting.
;
; CHECK-LABEL: @PR22093(
; CHECK-NOT: alloca
; CHECK: alloca i16
; CHECK-NOT: alloca
; CHECK: store volatile i16

entry:
  %a = alloca i32
  %a.cast = bitcast i32* %a to i16*
  store volatile i16 42, i16* %a.cast
  %load = load i32, i32* %a
  store i32 %load, i32* %a
  ret i32 %load
}

define void @PR22093.2() {
; Another way that we end up being unable to split a particular set of loads
; and stores can even have ordering importance. Here we have a load which is
; pre-splittable by itself, and the first store is also compatible. But the
; second store of the load makes the load unsplittable because of a mismatch of
; splits. Because this makes the load unsplittable, we also have to go back and
; remove the first store from the presplit candidates as its load won't be
; presplit.
;
; CHECK-LABEL: @PR22093.2(
; CHECK-NOT: alloca
; CHECK: alloca i16
; CHECK-NEXT: alloca i8
; CHECK-NOT: alloca
; CHECK: store volatile i16
; CHECK: store volatile i8

entry:
  %a = alloca i64
  %a.cast1 = bitcast i64* %a to i32*
  %a.cast2 = bitcast i64* %a to i16*
  store volatile i16 42, i16* %a.cast2
  %load = load i32, i32* %a.cast1
  store i32 %load, i32* %a.cast1
  %a.gep1 = getelementptr i32, i32* %a.cast1, i32 1
  %a.cast3 = bitcast i32* %a.gep1 to i8*
  store volatile i8 13, i8* %a.cast3
  store i32 %load, i32* %a.gep1
  ret void
}

define void @PR23737() {
; CHECK-LABEL: @PR23737(
; CHECK: store atomic volatile {{.*}} seq_cst
; CHECK: load atomic volatile {{.*}} seq_cst
entry:
  %ptr = alloca i64, align 8
  store atomic volatile i64 0, i64* %ptr seq_cst, align 8
  %load = load atomic volatile i64, i64* %ptr seq_cst, align 8
  ret void
}

define i16 @PR24463() {
; Ensure we can handle a very interesting case where there is an integer-based
; rewrite of the uses of the alloca, but where one of the integers in that is
; a sub-integer that requires extraction *and* extends past the end of the
; alloca. In this case, we should extract the i8 and then zext it to i16.
;
; CHECK-LABEL: @PR24463(
; CHECK-NOT: alloca
; CHECK: %[[SHIFT:.*]] = lshr i16 0, 8
; CHECK: %[[TRUNC:.*]] = trunc i16 %[[SHIFT]] to i8
; CHECK: %[[ZEXT:.*]] = zext i8 %[[TRUNC]] to i16
; CHECK: ret i16 %[[ZEXT]]
entry:
  %alloca = alloca [3 x i8]
  %gep1 = getelementptr inbounds [3 x i8], [3 x i8]* %alloca, i64 0, i64 1
  %bc1 = bitcast i8* %gep1 to i16*
  store i16 0, i16* %bc1
  %gep2 = getelementptr inbounds [3 x i8], [3 x i8]* %alloca, i64 0, i64 2
  %bc2 = bitcast i8* %gep2 to i16*
  %load = load i16, i16* %bc2
  ret i16 %load
}

%struct.STest = type { %struct.SPos, %struct.SPos }
%struct.SPos = type { float, float }

define void @PR25873(%struct.STest* %outData) {
; CHECK-LABEL: @PR25873(
; CHECK: store i32 1123418112
; CHECK: store i32 1139015680
; CHECK: %[[HIZEXT:.*]] = zext i32 1139015680 to i64
; CHECK: %[[HISHL:.*]] = shl i64 %[[HIZEXT]], 32
; CHECK: %[[HIMASK:.*]] = and i64 undef, 4294967295
; CHECK: %[[HIINSERT:.*]] = or i64 %[[HIMASK]], %[[HISHL]]
; CHECK: %[[LOZEXT:.*]] = zext i32 1123418112 to i64
; CHECK: %[[LOMASK:.*]] = and i64 %[[HIINSERT]], -4294967296
; CHECK: %[[LOINSERT:.*]] = or i64 %[[LOMASK]], %[[LOZEXT]]
; CHECK: store i64 %[[LOINSERT]]
entry:
  %tmpData = alloca %struct.STest, align 8
  %0 = bitcast %struct.STest* %tmpData to i8*
  call void @llvm.lifetime.start(i64 16, i8* %0)
  %x = getelementptr inbounds %struct.STest, %struct.STest* %tmpData, i64 0, i32 0, i32 0
  store float 1.230000e+02, float* %x, align 8
  %y = getelementptr inbounds %struct.STest, %struct.STest* %tmpData, i64 0, i32 0, i32 1
  store float 4.560000e+02, float* %y, align 4
  %m_posB = getelementptr inbounds %struct.STest, %struct.STest* %tmpData, i64 0, i32 1
  %1 = bitcast %struct.STest* %tmpData to i64*
  %2 = bitcast %struct.SPos* %m_posB to i64*
  %3 = load i64, i64* %1, align 8
  store i64 %3, i64* %2, align 8
  %4 = bitcast %struct.STest* %outData to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %4, i8* %0, i64 16, i32 4, i1 false)
  call void @llvm.lifetime.end(i64 16, i8* %0)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind
