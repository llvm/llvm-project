; RUN: llc -march=hexagon < %s | FileCheck %s

; Test for saturating add instructions.

; CHECK-LABEL: test1
; CHECK: v{{.*}}.ub = vadd(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub):sat
define <128 x i8> @test1(<128 x i8>* %a0, <128 x i8>* %a1) #0 {
entry:
  %wide.load = load <128 x i8>, <128 x i8>* %a0, align 1
  %wide.load62 = load <128 x i8>, <128 x i8>* %a1, align 1
  %add = call <128 x i8> @llvm.uadd.sat.v128i8(<128 x i8> %wide.load, <128 x i8> %wide.load62)
  ret <128 x i8> %add
}

; CHECK-LABEL: test2
; CHECK: v{{.*}}.b = vadd(v{{[0-9]+}}.b,v{{[0-9]+}}.b):sat
define <128 x i8> @test2(<128 x i8>* %a0, <128 x i8>* %a1) #0 {
entry:
  %wide.load = load <128 x i8>, <128 x i8>* %a0, align 1
  %wide.load62 = load <128 x i8>, <128 x i8>* %a1, align 1
  %add = call <128 x i8> @llvm.sadd.sat.v128i8(<128 x i8> %wide.load, <128 x i8> %wide.load62)
  ret <128 x i8> %add
}

; CHECK-LABEL: test3
; CHECK: v{{.*}}.uh = vadd(v{{[0-9]+}}.uh,v{{[0-9]+}}.uh):sat
define <64 x i16> @test3(<64 x i16>* %a0, <64 x i16>* %a1) #0 {
entry:
  %wide.load = load <64 x i16>, <64 x i16>* %a0, align 1
  %wide.load62 = load <64 x i16>, <64 x i16>* %a1, align 1
  %add = call <64 x i16> @llvm.uadd.sat.v64i16(<64 x i16> %wide.load, <64 x i16> %wide.load62)
  ret <64 x i16> %add
}

; CHECK-LABEL: test4
; CHECK: v{{.*}}.h = vadd(v{{[0-9]+}}.h,v{{[0-9]+}}.h):sat
define <64 x i16> @test4(<64 x i16>* %a0, <64 x i16>* %a1) #0 {
entry:
  %wide.load = load <64 x i16>, <64 x i16>* %a0, align 1
  %wide.load62 = load <64 x i16>, <64 x i16>* %a1, align 1
  %add = call <64 x i16> @llvm.sadd.sat.v64i16(<64 x i16> %wide.load, <64 x i16> %wide.load62)
  ret <64 x i16> %add
}

; CHECK-LABEL: test5
; CHECK: v{{.*}}.uw = vadd(v{{[0-9]+}}.uw,v{{[0-9]+}}.uw):sat
define <32 x i32> @test5(<32 x i32>* %a0, <32 x i32>* %a1) #0 {
entry:
  %wide.load = load <32 x i32>, <32 x i32>* %a0, align 1
  %wide.load62 = load <32 x i32>, <32 x i32>* %a1, align 1
  %add = call <32 x i32> @llvm.uadd.sat.v32i32(<32 x i32> %wide.load, <32 x i32> %wide.load62)
  ret <32 x i32> %add
}

; CHECK-LABEL: test6
; CHECK: v{{.*}}.w = vadd(v{{[0-9]+}}.w,v{{[0-9]+}}.w):sat
define <32 x i32> @test6(<32 x i32>* %a0, <32 x i32>* %a1) #0 {
entry:
  %wide.load = load <32 x i32>, <32 x i32>* %a0, align 1
  %wide.load62 = load <32 x i32>, <32 x i32>* %a1, align 1
  %add = call <32 x i32> @llvm.sadd.sat.v32i32(<32 x i32> %wide.load, <32 x i32> %wide.load62)
  ret <32 x i32> %add
}

; CHECK-LABEL: test7
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.ub = vadd(v{{[0-9]+}}:{{[0-9]+}}.ub,v{{[0-9]+}}:{{[0-9]+}}.ub):sat
define <256 x i8> @test7(<256 x i8>* %a0, <256 x i8>* %a1) #0 {
entry:
  %wide.load = load <256 x i8>, <256 x i8>* %a0, align 1
  %wide.load62 = load <256 x i8>, <256 x i8>* %a1, align 1
  %add = call <256 x i8> @llvm.uadd.sat.v256i8(<256 x i8> %wide.load, <256 x i8> %wide.load62)
  ret <256 x i8> %add
}

; CHECK-LABEL: test8
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.b = vadd(v{{[0-9]+}}:{{[0-9]+}}.b,v{{[0-9]+}}:{{[0-9]+}}.b):sat
define <256 x i8> @test8(<256 x i8>* %a0, <256 x i8>* %a1) #0 {
entry:
  %wide.load = load <256 x i8>, <256 x i8>* %a0, align 1
  %wide.load62 = load <256 x i8>, <256 x i8>* %a1, align 1
  %add = call <256 x i8> @llvm.sadd.sat.v256i8(<256 x i8> %wide.load, <256 x i8> %wide.load62)
  ret <256 x i8> %add
}

; CHECK-LABEL: test9
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uh = vadd(v{{[0-9]+}}:{{[0-9]+}}.uh,v{{[0-9]+}}:{{[0-9]+}}.uh):sat
define <128 x i16> @test9(<128 x i16>* %a0, <128 x i16>* %a1) #0 {
entry:
  %wide.load = load <128 x i16>, <128 x i16>* %a0, align 1
  %wide.load62 = load <128 x i16>, <128 x i16>* %a1, align 1
  %add = call <128 x i16> @llvm.uadd.sat.v128i16(<128 x i16> %wide.load, <128 x i16> %wide.load62)
  ret <128 x i16> %add
}

; CHECK-LABEL: test10
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vadd(v{{[0-9]+}}:{{[0-9]+}}.h,v{{[0-9]+}}:{{[0-9]+}}.h):sat
define <128 x i16> @test10(<128 x i16>* %a0, <128 x i16>* %a1) #0 {
entry:
  %wide.load = load <128 x i16>, <128 x i16>* %a0, align 1
  %wide.load62 = load <128 x i16>, <128 x i16>* %a1, align 1
  %add = call <128 x i16> @llvm.sadd.sat.v128i16(<128 x i16> %wide.load, <128 x i16> %wide.load62)
  ret <128 x i16> %add
}

; CHECK-LABEL: test11
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uw = vadd(v{{[0-9]+}}:{{[0-9]+}}.uw,v{{[0-9]+}}:{{[0-9]+}}.uw):sat
define <64 x i32> @test11(<64 x i32>* %a0, <64 x i32>* %a1) #0 {
entry:
  %wide.load = load <64 x i32>, <64 x i32>* %a0, align 1
  %wide.load62 = load <64 x i32>, <64 x i32>* %a1, align 1
  %add = call <64 x i32> @llvm.uadd.sat.v64i32(<64 x i32> %wide.load, <64 x i32> %wide.load62)
  ret <64 x i32> %add
}

; CHECK-LABEL: test12
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vadd(v{{[0-9]+}}:{{[0-9]+}}.w,v{{[0-9]+}}:{{[0-9]+}}.w):sat
define <64 x i32> @test12(<64 x i32>* %a0, <64 x i32>* %a1) #0 {
entry:
  %wide.load = load <64 x i32>, <64 x i32>* %a0, align 1
  %wide.load62 = load <64 x i32>, <64 x i32>* %a1, align 1
  %add = call <64 x i32> @llvm.sadd.sat.v64i32(<64 x i32> %wide.load, <64 x i32> %wide.load62)
  ret <64 x i32> %add
}

; CHECK-LABEL: test13
; CHECK: r{{[0-9]+}} = add(r{{[0-9]+}},r{{[0-9]+}}):sat
define i32 @test13(i32 %a0, i32 %a1) #0 {
entry:
  %add = call i32 @llvm.sadd.sat.i32(i32 %a0, i32 %a1)
  ret i32 %add
}

; CHECK-LABEL: test14
; CHECK: r{{[0-9]+}}:{{[0-9]+}} = add(r{{[0-9]+}}:{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}}):sat
define i64 @test14(i64 %a0, i64 %a1) #0 {
entry:
  %add = call i64 @llvm.sadd.sat.i64(i64 %a0, i64 %a1)
  ret i64 %add
}

declare <128 x i8> @llvm.uadd.sat.v128i8(<128 x i8>, <128 x i8>) #1
declare <128 x i8> @llvm.sadd.sat.v128i8(<128 x i8>, <128 x i8>) #1
declare <64 x i16> @llvm.uadd.sat.v64i16(<64 x i16>, <64 x i16>) #1
declare <64 x i16> @llvm.sadd.sat.v64i16(<64 x i16>, <64 x i16>) #1
declare <32 x i32> @llvm.uadd.sat.v32i32(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.sadd.sat.v32i32(<32 x i32>, <32 x i32>) #1
declare <256 x i8> @llvm.uadd.sat.v256i8(<256 x i8>, <256 x i8>) #1
declare <256 x i8> @llvm.sadd.sat.v256i8(<256 x i8>, <256 x i8>) #1
declare <128 x i16> @llvm.uadd.sat.v128i16(<128 x i16>, <128 x i16>) #1
declare <128 x i16> @llvm.sadd.sat.v128i16(<128 x i16>, <128 x i16>) #1
declare <64 x i32> @llvm.uadd.sat.v64i32(<64 x i32>, <64 x i32>) #1
declare <64 x i32> @llvm.sadd.sat.v64i32(<64 x i32>, <64 x i32>) #1
declare i32 @llvm.sadd.sat.i32(i32, i32)
declare i64 @llvm.sadd.sat.i64(i64, i64)

attributes #0 = { nounwind "target-cpu"="hexagonv73" "target-features"="+hvxv73,+hvx-length128b" }
attributes #1 = { nounwind readnone speculatable willreturn }
