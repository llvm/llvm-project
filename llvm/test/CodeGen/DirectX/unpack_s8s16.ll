; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.6-library %s | FileCheck %s

define noundef <4 x i16> @test_unpack_s8s16(i32 noundef %a) {
  ; CHECK: %{{.*}} = call { i16, i16, i16, i16 } @dx.op.unpack4x8.i16(i32 219, i8 1, i32 {{.*}})
  %unpacked = call { i16, i16, i16, i16 } @llvm.dx.unpack.s8s16(i32 %a)
  %1 = extractvalue { i16, i16, i16, i16 } %unpacked, 0
  %2 = extractvalue { i16, i16, i16, i16 } %unpacked, 1
  %3 = extractvalue { i16, i16, i16, i16 } %unpacked, 2
  %4 = extractvalue { i16, i16, i16, i16 } %unpacked, 3
  %5 = insertelement <4 x i16> poison, i16 %1, i32 0
  %6 = insertelement <4 x i16> %5, i16 %2, i32 1
  %7 = insertelement <4 x i16> %6, i16 %3, i32 2
  %8 = insertelement <4 x i16> %7, i16 %4, i32 3
  ret <4 x i16> %8
}

; CHECK-DAG: declare { i16, i16, i16, i16 } @dx.op.unpack4x8.i16(i32, i8, i32)
declare { i16, i16, i16, i16 } @llvm.dx.unpack.s8s16(i32)
