; RUN: opt < %s -sroa -S | FileCheck %s
; RUN: opt < %s -sroa -force-ssa-updater -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

declare void @llvm.lifetime.start(i64, i8* nocapture)
declare void @llvm.lifetime.end(i64, i8* nocapture)

define i32 @test0() {
; CHECK: @test0
; CHECK-NOT: alloca
; CHECK: ret i32

entry:
  %a1 = alloca i32
  %a2 = alloca float

  %a1.i8 = bitcast i32* %a1 to i8*
  call void @llvm.lifetime.start(i64 4, i8* %a1.i8)

  store i32 0, i32* %a1
  %v1 = load i32* %a1

  call void @llvm.lifetime.end(i64 4, i8* %a1.i8)

  %a2.i8 = bitcast float* %a2 to i8*
  call void @llvm.lifetime.start(i64 4, i8* %a2.i8)

  store float 0.0, float* %a2
  %v2 = load float * %a2
  %v2.int = bitcast float %v2 to i32
  %sum1 = add i32 %v1, %v2.int

  call void @llvm.lifetime.end(i64 4, i8* %a2.i8)

  ret i32 %sum1
}

define i32 @test1() {
; CHECK: @test1
; CHECK-NOT: alloca
; CHECK: ret i32 0

entry:
  %X = alloca { i32, float }
  %Y = getelementptr { i32, float }* %X, i64 0, i32 0
  store i32 0, i32* %Y
  %Z = load i32* %Y
  ret i32 %Z
}

define i64 @test2(i64 %X) {
; CHECK: @test2
; CHECK-NOT: alloca
; CHECK: ret i64 %X

entry:
  %A = alloca [8 x i8]
  %B = bitcast [8 x i8]* %A to i64*
  store i64 %X, i64* %B
  br label %L2

L2:
  %Z = load i64* %B
  ret i64 %Z
}

define void @test3(i8* %dst, i8* %src) {
; CHECK: @test3

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

  %b = getelementptr [300 x i8]* %a, i64 0, i64 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %b, i8* %src, i32 300, i32 1, i1 false)
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [42 x i8]* %[[test3_a1]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %src, i32 42
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %src, i64 42
; CHECK-NEXT: %[[test3_r1:.*]] = load i8* %[[gep]]
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 43
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [99 x i8]* %[[test3_a2]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 99
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 142
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 16
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 158
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [42 x i8]* %[[test3_a4]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 42
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 200
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a5]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %src, i64 207
; CHECK-NEXT: %[[test3_r2:.*]] = load i8* %[[gep]]
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 208
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 215
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [85 x i8]* %[[test3_a7]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 85

  ; Clobber a single element of the array, this should be promotable.
  %c = getelementptr [300 x i8]* %a, i64 0, i64 42
  store i8 0, i8* %c

  ; Make a sequence of overlapping stores to the array. These overlap both in
  ; forward strides and in shrinking accesses.
  %overlap.1.i8 = getelementptr [300 x i8]* %a, i64 0, i64 142
  %overlap.2.i8 = getelementptr [300 x i8]* %a, i64 0, i64 143
  %overlap.3.i8 = getelementptr [300 x i8]* %a, i64 0, i64 144
  %overlap.4.i8 = getelementptr [300 x i8]* %a, i64 0, i64 145
  %overlap.5.i8 = getelementptr [300 x i8]* %a, i64 0, i64 146
  %overlap.6.i8 = getelementptr [300 x i8]* %a, i64 0, i64 147
  %overlap.7.i8 = getelementptr [300 x i8]* %a, i64 0, i64 148
  %overlap.8.i8 = getelementptr [300 x i8]* %a, i64 0, i64 149
  %overlap.9.i8 = getelementptr [300 x i8]* %a, i64 0, i64 150
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
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 0
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
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 1
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 2, i64* %[[bitcast]]
  store i64 3, i64* %overlap.3.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 2
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 3, i64* %[[bitcast]]
  store i64 4, i64* %overlap.4.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 3
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 4, i64* %[[bitcast]]
  store i64 5, i64* %overlap.5.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 4
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 5, i64* %[[bitcast]]
  store i64 6, i64* %overlap.6.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 5
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 6, i64* %[[bitcast]]
  store i64 7, i64* %overlap.7.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 6
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 7, i64* %[[bitcast]]
  store i64 8, i64* %overlap.8.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 7
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 8, i64* %[[bitcast]]
  store i64 9, i64* %overlap.9.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 8
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 9, i64* %[[bitcast]]

  ; Make two sequences of overlapping stores with more gaps and irregularities.
  %overlap2.1.0.i8 = getelementptr [300 x i8]* %a, i64 0, i64 200
  %overlap2.1.1.i8 = getelementptr [300 x i8]* %a, i64 0, i64 201
  %overlap2.1.2.i8 = getelementptr [300 x i8]* %a, i64 0, i64 202
  %overlap2.1.3.i8 = getelementptr [300 x i8]* %a, i64 0, i64 203

  %overlap2.2.0.i8 = getelementptr [300 x i8]* %a, i64 0, i64 208
  %overlap2.2.1.i8 = getelementptr [300 x i8]* %a, i64 0, i64 209
  %overlap2.2.2.i8 = getelementptr [300 x i8]* %a, i64 0, i64 210
  %overlap2.2.3.i8 = getelementptr [300 x i8]* %a, i64 0, i64 211

  %overlap2.1.0.i16 = bitcast i8* %overlap2.1.0.i8 to i16*
  %overlap2.1.0.i32 = bitcast i8* %overlap2.1.0.i8 to i32*
  %overlap2.1.1.i32 = bitcast i8* %overlap2.1.1.i8 to i32*
  %overlap2.1.2.i32 = bitcast i8* %overlap2.1.2.i8 to i32*
  %overlap2.1.3.i32 = bitcast i8* %overlap2.1.3.i8 to i32*
  store i8 1,  i8*  %overlap2.1.0.i8
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a5]], i64 0, i64 0
; CHECK-NEXT: store i8 1, i8* %[[gep]]
  store i16 1, i16* %overlap2.1.0.i16
; CHECK-NEXT: %[[bitcast:.*]] = bitcast [7 x i8]* %[[test3_a5]] to i16*
; CHECK-NEXT: store i16 1, i16* %[[bitcast]]
  store i32 1, i32* %overlap2.1.0.i32
; CHECK-NEXT: %[[bitcast:.*]] = bitcast [7 x i8]* %[[test3_a5]] to i32*
; CHECK-NEXT: store i32 1, i32* %[[bitcast]]
  store i32 2, i32* %overlap2.1.1.i32
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a5]], i64 0, i64 1
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT: store i32 2, i32* %[[bitcast]]
  store i32 3, i32* %overlap2.1.2.i32
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a5]], i64 0, i64 2
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT: store i32 3, i32* %[[bitcast]]
  store i32 4, i32* %overlap2.1.3.i32
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a5]], i64 0, i64 3
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
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 1
; CHECK-NEXT: store i8 1, i8* %[[gep]]
  store i16 1, i16* %overlap2.2.1.i16
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 1
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: store i16 1, i16* %[[bitcast]]
  store i32 1, i32* %overlap2.2.1.i32
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 1
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT: store i32 1, i32* %[[bitcast]]
  store i32 3, i32* %overlap2.2.2.i32
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 2
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT: store i32 3, i32* %[[bitcast]]
  store i32 4, i32* %overlap2.2.3.i32
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 3
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT: store i32 4, i32* %[[bitcast]]

  %overlap2.prefix = getelementptr i8* %overlap2.1.1.i8, i64 -4
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %overlap2.prefix, i8* %src, i32 8, i32 1, i1 false)
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [42 x i8]* %[[test3_a4]], i64 0, i64 39
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %src, i32 3
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 3
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a5]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 5

  ; Bridge between the overlapping areas
  call void @llvm.memset.p0i8.i32(i8* %overlap2.1.2.i8, i8 42, i32 8, i32 1, i1 false)
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a5]], i64 0, i64 2
; CHECK-NEXT: call void @llvm.memset.p0i8.i32(i8* %[[gep]], i8 42, i32 5
; ...promoted i8 store...
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memset.p0i8.i32(i8* %[[gep]], i8 42, i32 2

  ; Entirely within the second overlap.
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %overlap2.2.1.i8, i8* %src, i32 5, i32 1, i1 false)
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 1
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep]], i8* %src, i32 5

  ; Trailing past the second overlap.
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %overlap2.2.2.i8, i8* %src, i32 8, i32 1, i1 false)
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 2
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep]], i8* %src, i32 5
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 5
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [85 x i8]* %[[test3_a7]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 3

  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %b, i32 300, i32 1, i1 false)
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [42 x i8]* %[[test3_a1]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %[[gep]], i32 42
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %dst, i64 42
; CHECK-NEXT: store i8 0, i8* %[[gep]]
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 43
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [99 x i8]* %[[test3_a2]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 99
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 142
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 16
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 158
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [42 x i8]* %[[test3_a4]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 42
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 200
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a5]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %dst, i64 207
; CHECK-NEXT: store i8 42, i8* %[[gep]]
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 208
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 215
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [85 x i8]* %[[test3_a7]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 85

  ret void
}

define void @test4(i8* %dst, i8* %src) {
; CHECK: @test4

entry:
  %a = alloca [100 x i8]
; CHECK-NOT:  alloca
; CHECK:      %[[test4_a1:.*]] = alloca [20 x i8]
; CHECK-NEXT: %[[test4_a2:.*]] = alloca [7 x i8]
; CHECK-NEXT: %[[test4_a3:.*]] = alloca [10 x i8]
; CHECK-NEXT: %[[test4_a4:.*]] = alloca [7 x i8]
; CHECK-NEXT: %[[test4_a5:.*]] = alloca [7 x i8]
; CHECK-NEXT: %[[test4_a6:.*]] = alloca [40 x i8]

  %b = getelementptr [100 x i8]* %a, i64 0, i64 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %b, i8* %src, i32 100, i32 1, i1 false)
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [20 x i8]* %[[test4_a1]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep]], i8* %src, i32 20
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %src, i64 20
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: %[[test4_r1:.*]] = load i16* %[[bitcast]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %src, i64 22
; CHECK-NEXT: %[[test4_r2:.*]] = load i8* %[[gep]]
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 23
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a2]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 30
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [10 x i8]* %[[test4_a3]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 10
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %src, i64 40
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: %[[test4_r3:.*]] = load i16* %[[bitcast]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %src, i64 42
; CHECK-NEXT: %[[test4_r4:.*]] = load i8* %[[gep]]
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 43
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a4]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %src, i64 50
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: %[[test4_r5:.*]] = load i16* %[[bitcast]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %src, i64 52
; CHECK-NEXT: %[[test4_r6:.*]] = load i8* %[[gep]]
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 53
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a5]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 60
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [40 x i8]* %[[test4_a6]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 40

  %a.src.1 = getelementptr [100 x i8]* %a, i64 0, i64 20
  %a.dst.1 = getelementptr [100 x i8]* %a, i64 0, i64 40
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a.dst.1, i8* %a.src.1, i32 10, i32 1, i1 false)
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a4]], i64 0, i64 0
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a2]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7

  ; Clobber a single element of the array, this should be promotable, and be deleted.
  %c = getelementptr [100 x i8]* %a, i64 0, i64 42
  store i8 0, i8* %c

  %a.src.2 = getelementptr [100 x i8]* %a, i64 0, i64 50
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %a.dst.1, i8* %a.src.2, i32 10, i32 1, i1 false)
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a4]], i64 0, i64 0
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a5]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7

  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %b, i32 100, i32 1, i1 false)
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [20 x i8]* %[[test4_a1]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %[[gep]], i32 20
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %dst, i64 20
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: store i16 %[[test4_r1]], i16* %[[bitcast]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %dst, i64 22
; CHECK-NEXT: store i8 %[[test4_r2]], i8* %[[gep]]
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 23
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a2]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 30
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [10 x i8]* %[[test4_a3]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 10
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %dst, i64 40
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: store i16 %[[test4_r5]], i16* %[[bitcast]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %dst, i64 42
; CHECK-NEXT: store i8 %[[test4_r6]], i8* %[[gep]]
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 43
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a4]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %dst, i64 50
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: store i16 %[[test4_r5]], i16* %[[bitcast]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %dst, i64 52
; CHECK-NEXT: store i8 %[[test4_r6]], i8* %[[gep]]
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 53
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a5]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 60
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [40 x i8]* %[[test4_a6]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 40

  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind

define i16 @test5() {
; CHECK: @test5
; CHECK-NOT: alloca float
; CHECK:      %[[cast:.*]] = bitcast float 0.0{{.*}} to i32
; CHECK-NEXT: %[[shr:.*]] = lshr i32 %[[cast]], 16
; CHECK-NEXT: %[[trunc:.*]] = trunc i32 %[[shr]] to i16
; CHECK-NEXT: ret i16 %[[trunc]]

entry:
  %a = alloca [4 x i8]
  %fptr = bitcast [4 x i8]* %a to float*
  store float 0.0, float* %fptr
  %ptr = getelementptr [4 x i8]* %a, i32 0, i32 2
  %iptr = bitcast i8* %ptr to i16*
  %val = load i16* %iptr
  ret i16 %val
}

define i32 @test6() {
; CHECK: @test6
; CHECK: alloca i32
; CHECK-NEXT: store volatile i32
; CHECK-NEXT: load i32*
; CHECK-NEXT: ret i32

entry:
  %a = alloca [4 x i8]
  %ptr = getelementptr [4 x i8]* %a, i32 0, i32 0
  call void @llvm.memset.p0i8.i32(i8* %ptr, i8 42, i32 4, i32 1, i1 true)
  %iptr = bitcast i8* %ptr to i32*
  %val = load i32* %iptr
  ret i32 %val
}

define void @test7(i8* %src, i8* %dst) {
; CHECK: @test7
; CHECK: alloca i32
; CHECK-NEXT: bitcast i8* %src to i32*
; CHECK-NEXT: load volatile i32*
; CHECK-NEXT: store volatile i32
; CHECK-NEXT: bitcast i8* %dst to i32*
; CHECK-NEXT: load volatile i32*
; CHECK-NEXT: store volatile i32
; CHECK-NEXT: ret

entry:
  %a = alloca [4 x i8]
  %ptr = getelementptr [4 x i8]* %a, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %ptr, i8* %src, i32 4, i32 1, i1 true)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %ptr, i32 4, i32 1, i1 true)
  ret void
}


%S1 = type { i32, i32, [16 x i8] }
%S2 = type { %S1*, %S2* }

define %S2 @test8(%S2* %s2) {
; CHECK: @test8
entry:
  %new = alloca %S2
; CHECK-NOT: alloca

  %s2.next.ptr = getelementptr %S2* %s2, i64 0, i32 1
  %s2.next = load %S2** %s2.next.ptr
; CHECK:      %[[gep:.*]] = getelementptr %S2* %s2, i64 0, i32 1
; CHECK-NEXT: %[[next:.*]] = load %S2** %[[gep]]

  %s2.next.s1.ptr = getelementptr %S2* %s2.next, i64 0, i32 0
  %s2.next.s1 = load %S1** %s2.next.s1.ptr
  %new.s1.ptr = getelementptr %S2* %new, i64 0, i32 0
  store %S1* %s2.next.s1, %S1** %new.s1.ptr
  %s2.next.next.ptr = getelementptr %S2* %s2.next, i64 0, i32 1
  %s2.next.next = load %S2** %s2.next.next.ptr
  %new.next.ptr = getelementptr %S2* %new, i64 0, i32 1
  store %S2* %s2.next.next, %S2** %new.next.ptr
; CHECK-NEXT: %[[gep:.*]] = getelementptr %S2* %[[next]], i64 0, i32 0
; CHECK-NEXT: %[[next_s1:.*]] = load %S1** %[[gep]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr %S2* %[[next]], i64 0, i32 1
; CHECK-NEXT: %[[next_next:.*]] = load %S2** %[[gep]]

  %new.s1 = load %S1** %new.s1.ptr
  %result1 = insertvalue %S2 undef, %S1* %new.s1, 0
; CHECK-NEXT: %[[result1:.*]] = insertvalue %S2 undef, %S1* %[[next_s1]], 0
  %new.next = load %S2** %new.next.ptr
  %result2 = insertvalue %S2 %result1, %S2* %new.next, 1
; CHECK-NEXT: %[[result2:.*]] = insertvalue %S2 %[[result1]], %S2* %[[next_next]], 1
  ret %S2 %result2
; CHECK-NEXT: ret %S2 %[[result2]]
}

define i64 @test9() {
; Ensure we can handle loads off the end of an alloca even when wrapped in
; weird bit casts and types. The result is undef, but this shouldn't crash
; anything.
; CHECK: @test9
; CHECK-NOT: alloca
; CHECK: ret i64 undef

entry:
  %a = alloca { [3 x i8] }
  %gep1 = getelementptr inbounds { [3 x i8] }* %a, i32 0, i32 0, i32 0
  store i8 0, i8* %gep1, align 1
  %gep2 = getelementptr inbounds { [3 x i8] }* %a, i32 0, i32 0, i32 1
  store i8 0, i8* %gep2, align 1
  %gep3 = getelementptr inbounds { [3 x i8] }* %a, i32 0, i32 0, i32 2
  store i8 26, i8* %gep3, align 1
  %cast = bitcast { [3 x i8] }* %a to { i64 }*
  %elt = getelementptr inbounds { i64 }* %cast, i32 0, i32 0
  %result = load i64* %elt
  ret i64 %result
}

define %S2* @test10() {
; CHECK: @test10
; CHECK-NOT: alloca %S2*
; CHECK: ret %S2* null

entry:
  %a = alloca [8 x i8]
  %ptr = getelementptr [8 x i8]* %a, i32 0, i32 0
  call void @llvm.memset.p0i8.i32(i8* %ptr, i8 0, i32 8, i32 1, i1 false)
  %s2ptrptr = bitcast i8* %ptr to %S2**
  %s2ptr = load %S2** %s2ptrptr
  ret %S2* %s2ptr
}

define i32 @test11() {
; CHECK: @test11
; CHECK-NOT: alloca
; CHECK: ret i32 0

entry:
  %X = alloca i32
  br i1 undef, label %good, label %bad

good:
  %Y = getelementptr i32* %X, i64 0
  store i32 0, i32* %Y
  %Z = load i32* %Y
  ret i32 %Z

bad:
  %Y2 = getelementptr i32* %X, i64 1
  store i32 0, i32* %Y2
  %Z2 = load i32* %Y2
  ret i32 %Z2
}

define i8 @test12() {
; We fully promote these to the i24 load or store size, resulting in just masks
; and other operations that instcombine will fold, but no alloca.
;
; CHECK: @test12

entry:
  %a = alloca [3 x i8]
  %b = alloca [3 x i8]
; CHECK-NOT: alloca

  %a0ptr = getelementptr [3 x i8]* %a, i64 0, i32 0
  store i8 0, i8* %a0ptr
  %a1ptr = getelementptr [3 x i8]* %a, i64 0, i32 1
  store i8 0, i8* %a1ptr
  %a2ptr = getelementptr [3 x i8]* %a, i64 0, i32 2
  store i8 0, i8* %a2ptr
  %aiptr = bitcast [3 x i8]* %a to i24*
  %ai = load i24* %aiptr
; CHCEK-NOT: store
; CHCEK-NOT: load
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
  %b0ptr = getelementptr [3 x i8]* %b, i64 0, i32 0
  %b0 = load i8* %b0ptr
  %b1ptr = getelementptr [3 x i8]* %b, i64 0, i32 1
  %b1 = load i8* %b1ptr
  %b2ptr = getelementptr [3 x i8]* %b, i64 0, i32 2
  %b2 = load i8* %b2ptr
; CHCEK-NOT: store
; CHCEK-NOT: load
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
; CHECK: @test13
; CHECK: %[[ret:.*]] = zext i16 undef to i32
; CHECK: ret i32 %[[ret]]

entry:
  %a = alloca [3 x i8]
  %b0ptr = getelementptr [3 x i8]* %a, i64 0, i32 0
  store i8 0, i8* %b0ptr
  %b1ptr = getelementptr [3 x i8]* %a, i64 0, i32 1
  store i8 0, i8* %b1ptr
  %b2ptr = getelementptr [3 x i8]* %a, i64 0, i32 2
  store i8 0, i8* %b2ptr
  %iptrcast = bitcast [3 x i8]* %a to i16*
  %iptrgep = getelementptr i16* %iptrcast, i64 1
  %i = load i16* %iptrgep
  %ret = zext i16 %i to i32
  ret i32 %ret
}

%test14.struct = type { [3 x i32] }

define void @test14(...) nounwind uwtable {
; This is a strange case where we split allocas into promotable partitions, but
; also gain enough data to prove they must be dead allocas due to GEPs that walk
; across two adjacent allocas. Test that we don't try to promote or otherwise
; do bad things to these dead allocas, they should just be removed.
; CHECK: @test14
; CHECK-NEXT: entry:
; CHECK-NEXT: ret void

entry:
  %a = alloca %test14.struct
  %p = alloca %test14.struct*
  %0 = bitcast %test14.struct* %a to i8*
  %1 = getelementptr i8* %0, i64 12
  %2 = bitcast i8* %1 to %test14.struct*
  %3 = getelementptr inbounds %test14.struct* %2, i32 0, i32 0
  %4 = getelementptr inbounds %test14.struct* %a, i32 0, i32 0
  %5 = bitcast [3 x i32]* %3 to i32*
  %6 = bitcast [3 x i32]* %4 to i32*
  %7 = load i32* %6, align 4
  store i32 %7, i32* %5, align 4
  %8 = getelementptr inbounds i32* %5, i32 1
  %9 = getelementptr inbounds i32* %6, i32 1
  %10 = load i32* %9, align 4
  store i32 %10, i32* %8, align 4
  %11 = getelementptr inbounds i32* %5, i32 2
  %12 = getelementptr inbounds i32* %6, i32 2
  %13 = load i32* %12, align 4
  store i32 %13, i32* %11, align 4
  ret void
}

define i32 @test15(i1 %flag) nounwind uwtable {
; Ensure that when there are dead instructions using an alloca that are not
; loads or stores we still delete them during partitioning and rewriting.
; Otherwise we'll go to promote them while thy still have unpromotable uses.
; CHECK: @test15
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
  %gep0 = getelementptr i8* %bc0, i64 3
  %dead0 = bitcast i8* %gep0 to i64*

  store i64 1879048192, i64* %l1, align 8
  %bc1 = bitcast i64* %l1 to i8*
  %gep1 = getelementptr i8* %bc1, i64 3
  %dead1 = getelementptr i8* %gep1, i64 1

  store i64 1879048192, i64* %l2, align 8
  %bc2 = bitcast i64* %l2 to i8*
  %gep2.1 = getelementptr i8* %bc2, i64 1
  %gep2.2 = getelementptr i8* %bc2, i64 3
  ; Note that this select should get visited multiple times due to using two
  ; different GEPs off the same alloca. We should only delete it once.
  %dead2 = select i1 %flag, i8* %gep2.1, i8* %gep2.2

  store i64 1879048192, i64* %l3, align 8
  %bc3 = bitcast i64* %l3 to i8*
  %gep3 = getelementptr i8* %bc3, i64 3

  br label %loop
}

define void @test16(i8* %src, i8* %dst) {
; Ensure that we can promote an alloca of [3 x i8] to an i24 SSA value.
; CHECK: @test16
; CHECK-NOT: alloca
; CHECK:      %[[srccast:.*]] = bitcast i8* %src to i24*
; CHECK-NEXT: load i24* %[[srccast]]
; CHECK-NEXT: %[[dstcast:.*]] = bitcast i8* %dst to i24*
; CHECK-NEXT: store i24 0, i24* %[[dstcast]]
; CHECK-NEXT: ret void

entry:
  %a = alloca [3 x i8]
  %ptr = getelementptr [3 x i8]* %a, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %ptr, i8* %src, i32 4, i32 1, i1 false)
  %cast = bitcast i8* %ptr to i24*
  store i24 0, i24* %cast
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %ptr, i32 4, i32 1, i1 false)
  ret void
}

define void @test17(i8* %src, i8* %dst) {
; Ensure that we can rewrite unpromotable memcpys which extend past the end of
; the alloca.
; CHECK: @test17
; CHECK:      %[[a:.*]] = alloca [3 x i8]
; CHECK-NEXT: %[[ptr:.*]] = getelementptr [3 x i8]* %[[a]], i32 0, i32 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[ptr]], i8* %src,
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %[[ptr]],
; CHECK-NEXT: ret void

entry:
  %a = alloca [3 x i8]
  %ptr = getelementptr [3 x i8]* %a, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %ptr, i8* %src, i32 4, i32 1, i1 true)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %ptr, i32 4, i32 1, i1 true)
  ret void
}

define void @test18(i8* %src, i8* %dst, i32 %size) {
; Preserve transfer instrinsics with a variable size, even if they overlap with
; fixed size operations. Further, continue to split and promote allocas preceding
; the variable sized intrinsic.
; CHECK: @test18
; CHECK:      %[[a:.*]] = alloca [34 x i8]
; CHECK:      %[[srcgep1:.*]] = getelementptr inbounds i8* %src, i64 4
; CHECK-NEXT: %[[srccast1:.*]] = bitcast i8* %[[srcgep1]] to i32*
; CHECK-NEXT: %[[srcload:.*]] = load i32* %[[srccast1]]
; CHECK-NEXT: %[[agep1:.*]] = getelementptr inbounds [34 x i8]* %[[a]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[agep1]], i8* %src, i32 %size,
; CHECK-NEXT: %[[agep2:.*]] = getelementptr inbounds [34 x i8]* %[[a]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memset.p0i8.i32(i8* %[[agep2]], i8 42, i32 %size,
; CHECK-NEXT: %[[dstcast1:.*]] = bitcast i8* %dst to i32*
; CHECK-NEXT: store i32 42, i32* %[[dstcast1]]
; CHECK-NEXT: %[[dstgep1:.*]] = getelementptr inbounds i8* %dst, i64 4
; CHECK-NEXT: %[[dstcast2:.*]] = bitcast i8* %[[dstgep1]] to i32*
; CHECK-NEXT: store i32 %[[srcload]], i32* %[[dstcast2]]
; CHECK-NEXT: %[[agep3:.*]] = getelementptr inbounds [34 x i8]* %[[a]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %[[agep3]], i32 %size,
; CHECK-NEXT: ret void

entry:
  %a = alloca [42 x i8]
  %ptr = getelementptr [42 x i8]* %a, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %ptr, i8* %src, i32 8, i32 1, i1 false)
  %ptr2 = getelementptr [42 x i8]* %a, i32 0, i32 8
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
; CHECK: @test19
; CHECK-NOT: alloca
; CHECK: ret i32 undef

entry:
  %a = alloca { i64, i8* }
  %cast1 = bitcast %opaque* %x to i8*
  %cast2 = bitcast { i64, i8* }* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %cast2, i8* %cast1, i32 16, i32 1, i1 false)
  %gep = getelementptr inbounds { i64, i8* }* %a, i32 0, i32 0
  %val = load i64* %gep
  ret i32 undef
}

define i32 @test20() {
; Ensure we can track negative offsets (before the beginning of the alloca) and
; negative relative offsets from offsets starting past the end of the alloca.
; CHECK: @test20
; CHECK-NOT: alloca
; CHECK: %[[sum1:.*]] = add i32 1, 2
; CHECK: %[[sum2:.*]] = add i32 %[[sum1]], 3
; CHECK: ret i32 %[[sum2]]

entry:
  %a = alloca [3 x i32]
  %gep1 = getelementptr [3 x i32]* %a, i32 0, i32 0
  store i32 1, i32* %gep1
  %gep2.1 = getelementptr [3 x i32]* %a, i32 0, i32 -2
  %gep2.2 = getelementptr i32* %gep2.1, i32 3
  store i32 2, i32* %gep2.2
  %gep3.1 = getelementptr [3 x i32]* %a, i32 0, i32 14
  %gep3.2 = getelementptr i32* %gep3.1, i32 -12
  store i32 3, i32* %gep3.2

  %load1 = load i32* %gep1
  %load2 = load i32* %gep2.2
  %load3 = load i32* %gep3.2
  %sum1 = add i32 %load1, %load2
  %sum2 = add i32 %sum1, %load3
  ret i32 %sum2
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind

define i8 @test21() {
; Test allocations and offsets which border on overflow of the int64_t used
; internally. This is really awkward to really test as LLVM doesn't really
; support such extreme constructs cleanly.
; CHECK: @test21
; CHECK-NOT: alloca
; CHECK: or i8 -1, -1

entry:
  %a = alloca [2305843009213693951 x i8]
  %gep0 = getelementptr [2305843009213693951 x i8]* %a, i64 0, i64 2305843009213693949
  store i8 255, i8* %gep0
  %gep1 = getelementptr [2305843009213693951 x i8]* %a, i64 0, i64 -9223372036854775807
  %gep2 = getelementptr i8* %gep1, i64 -1
  call void @llvm.memset.p0i8.i64(i8* %gep2, i8 0, i64 18446744073709551615, i32 1, i1 false)
  %gep3 = getelementptr i8* %gep1, i64 9223372036854775807
  %gep4 = getelementptr i8* %gep3, i64 9223372036854775807
  %gep5 = getelementptr i8* %gep4, i64 -6917529027641081857
  store i8 255, i8* %gep5
  %cast1 = bitcast i8* %gep4 to i32*
  store i32 0, i32* %cast1
  %load = load i8* %gep0
  %gep6 = getelementptr i8* %gep0, i32 1
  %load2 = load i8* %gep6
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
  %tmp2 = load i8* %a
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
  %gep = getelementptr %PR13916.struct* %a, i32 0, i32 0
  %tmp2 = load i8* %gep
  ret void
}

define void @PR13990() {
; Ensure we can handle cases where processing one alloca causes the other
; alloca to become dead and get deleted. This might crash or fail under
; Valgrind if we regress.
; CHECK: @PR13990
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
; CHECK: @PR13969

entry:
  %a = alloca double
  %b = alloca double*
  %c = alloca double
; CHECK-NOT: alloca

  store double %x, double* %a
  store double* %c, double** %b
  store double* %a, double** %b
  store double %x, double* %c
  %ret = load double* %a
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
; CHECK: @PR14034

entry:
  %a = alloca %PR14034.struct
  %list = getelementptr %PR14034.struct* %a, i32 0, i32 2
  %prev = getelementptr %PR14034.list* %list, i32 0, i32 1
  store %PR14034.list* undef, %PR14034.list** %prev
  %cast0 = bitcast %PR14034.struct* undef to i8*
  %cast1 = bitcast %PR14034.struct* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %cast0, i8* %cast1, i32 12, i32 0, i1 false)
  ret void
}

define i32 @test22(i32 %x) {
; Test that SROA and promotion is not confused by a grab bax mixture of pointer
; types involving wrapper aggregates and zero-length aggregate members.
; CHECK: @test22

entry:
  %a1 = alloca { { [1 x { i32 }] } }
  %a2 = alloca { {}, { float }, [0 x i8] }
  %a3 = alloca { [0 x i8], { [0 x double], [1 x [1 x <4 x i8>]], {} }, { { {} } } }
; CHECK-NOT: alloca

  %wrap1 = insertvalue [1 x { i32 }] undef, i32 %x, 0, 0
  %gep1 = getelementptr { { [1 x { i32 }] } }* %a1, i32 0, i32 0, i32 0
  store [1 x { i32 }] %wrap1, [1 x { i32 }]* %gep1

  %gep2 = getelementptr { { [1 x { i32 }] } }* %a1, i32 0, i32 0
  %ptrcast1 = bitcast { [1 x { i32 }] }* %gep2 to { [1 x { float }] }*
  %load1 = load { [1 x { float }] }* %ptrcast1
  %unwrap1 = extractvalue { [1 x { float }] } %load1, 0, 0

  %wrap2 = insertvalue { {}, { float }, [0 x i8] } undef, { float } %unwrap1, 1
  store { {}, { float }, [0 x i8] } %wrap2, { {}, { float }, [0 x i8] }* %a2

  %gep3 = getelementptr { {}, { float }, [0 x i8] }* %a2, i32 0, i32 1, i32 0
  %ptrcast2 = bitcast float* %gep3 to <4 x i8>*
  %load3 = load <4 x i8>* %ptrcast2
  %valcast1 = bitcast <4 x i8> %load3 to i32

  %wrap3 = insertvalue [1 x [1 x i32]] undef, i32 %valcast1, 0, 0
  %wrap4 = insertvalue { [1 x [1 x i32]], {} } undef, [1 x [1 x i32]] %wrap3, 0
  %gep4 = getelementptr { [0 x i8], { [0 x double], [1 x [1 x <4 x i8>]], {} }, { { {} } } }* %a3, i32 0, i32 1
  %ptrcast3 = bitcast { [0 x double], [1 x [1 x <4 x i8>]], {} }* %gep4 to { [1 x [1 x i32]], {} }*
  store { [1 x [1 x i32]], {} } %wrap4, { [1 x [1 x i32]], {} }* %ptrcast3

  %gep5 = getelementptr { [0 x i8], { [0 x double], [1 x [1 x <4 x i8>]], {} }, { { {} } } }* %a3, i32 0, i32 1, i32 1, i32 0
  %ptrcast4 = bitcast [1 x <4 x i8>]* %gep5 to { {}, float, {} }*
  %load4 = load { {}, float, {} }* %ptrcast4
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
  %X.sroa.0.2.raw_idx2.i = getelementptr inbounds i8* %0, i32 2
  call void @llvm.memset.p0i8.i64(i8* %X.sroa.0.2.raw_idx2.i, i8 0, i64 4, i32 1, i1 false)

  ; Or a memset of the whole thing.
  call void @llvm.memset.p0i8.i64(i8* %0, i8 0, i64 8, i32 1, i1 false)

  ; Write to the high 32-bits with a memcpy.
  %X.sroa.0.4.raw_idx4.i = getelementptr inbounds i8* %0, i32 4
  %d.raw = bitcast double* %d to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %X.sroa.0.4.raw_idx4.i, i8* %d.raw, i32 4, i32 1, i1 false)

  ; Store to the high 32-bits...
  %X.sroa.0.4.cast5.i = bitcast i8* %X.sroa.0.4.raw_idx4.i to i32*
  store i32 1072693248, i32* %X.sroa.0.4.cast5.i, align 4

  ; Do the actual math...
  %X.sroa.0.0.load1.i = load double* %X.sroa.0.i, align 8
  %accum.real.i = load double* %d, align 8
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

  %phi.realp = getelementptr inbounds { float, float }* %phi, i32 0, i32 0
  %phi.real = load float* %phi.realp
  %phi.imagp = getelementptr inbounds { float, float }* %phi, i32 0, i32 1
  %phi.imag = load float* %phi.imagp
  ; CHECK:      %[[realp:.*]] = getelementptr inbounds { float, float }* %phi, i32 0, i32 0
  ; CHECK-NEXT: %[[real:.*]] = load float* %[[realp]]
  ; CHECK-NEXT: %[[imagp:.*]] = getelementptr inbounds { float, float }* %phi, i32 0, i32 1
  ; CHECK-NEXT: %[[imag:.*]] = load float* %[[imagp]]

  %real = getelementptr inbounds { float, float }* %retval, i32 0, i32 0
  %imag = getelementptr inbounds { float, float }* %retval, i32 0, i32 1
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

  %1 = load i64* %0, align 1
  ret i64 %1
  ; CHECK-NEXT: ret i64 %[[real_insert]]
}

define void @PR14105({ [16 x i8] }* %ptr) {
; Ensure that when rewriting the GEP index '-1' for this alloca we preserve is
; sign as negative. We use a volatile memcpy to ensure promotion never actually
; occurs.
; CHECK: @PR14105

entry:
  %a = alloca { [16 x i8] }, align 8
; CHECK: alloca [16 x i8], align 8

  %gep = getelementptr inbounds { [16 x i8] }* %ptr, i64 -1
; CHECK-NEXT: getelementptr inbounds { [16 x i8] }* %ptr, i64 -1, i32 0, i64 0

  %cast1 = bitcast { [16 x i8 ] }* %gep to i8*
  %cast2 = bitcast { [16 x i8 ] }* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %cast1, i8* %cast2, i32 16, i32 8, i1 true)
  ret void
; CHECK: ret
}

define void @PR14465() {
; Ensure that we don't crash when analyzing a alloca larger than the maximum
; integer type width (MAX_INT_BITS) supported by llvm (1048576*32 > (1<<23)-1).
; CHECK: @PR14465

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
; CHECK: @PR14548

entry:
  %a = alloca <{ i1 }>, align 8
  %b = alloca <{ i1 }>, align 8
; Nothing of interest is simplified here.
; CHECK: alloca
; CHECK: alloca

  %b.i1 = bitcast <{ i1 }>* %b to i1*
  store i1 %x, i1* %b.i1, align 8
  %b.i8 = bitcast <{ i1 }>* %b to i8*
  %foo = load i8* %b.i8, align 1

  %a.i8 = bitcast <{ i1 }>* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a.i8, i8* %b.i8, i32 1, i32 1, i1 false) nounwind
  %bar = load i8* %a.i8, align 1
  %a.i1 = getelementptr inbounds <{ i1 }>* %a, i32 0, i32 0
  %baz = load i1* %a.i1, align 1
  ret void
}
