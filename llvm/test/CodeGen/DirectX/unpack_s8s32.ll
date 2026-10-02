; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.6-library %s | FileCheck %s

define noundef <4 x i32> @test_unpack_s8s32(i32 noundef %a) {
  ; CHECK: %{{.*}} = call { i32, i32, i32, i32 } @dx.op.unpack4x8.i32(i32 219, i8 1, i32 {{.*}})
  %unpacked = call { i32, i32, i32, i32 } @llvm.dx.unpack.s8s32(i32 %a)
  %1 = extractvalue { i32, i32, i32, i32 } %unpacked, 0
  %2 = extractvalue { i32, i32, i32, i32 } %unpacked, 1
  %3 = extractvalue { i32, i32, i32, i32 } %unpacked, 2
  %4 = extractvalue { i32, i32, i32, i32 } %unpacked, 3
  %5 = insertelement <4 x i32> poison, i32 %1, i32 0
  %6 = insertelement <4 x i32> %5, i32 %2, i32 1
  %7 = insertelement <4 x i32> %6, i32 %3, i32 2
  %8 = insertelement <4 x i32> %7, i32 %4, i32 3
  ret <4 x i32> %8
}

; CHECK-DAG: declare { i32, i32, i32, i32 } @dx.op.unpack4x8.i32(i32, i8, i32)
declare { i32, i32, i32, i32 } @llvm.dx.unpack.s8s32(i32)
