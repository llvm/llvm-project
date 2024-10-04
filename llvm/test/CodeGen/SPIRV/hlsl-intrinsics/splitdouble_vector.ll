; RUN: opt -S -scalarizer -mtriple=spirv-vulkan-library %s 2>&1 | llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown -o - | FileCheck %s

; SPIRV lowering for splitdouble should relly on the scalarizer.

define spir_func noundef <3 x i32> @test_vector(<3 x double> noundef %D) local_unnamed_addr {
entry:
  ; CHECK-COUNT-3: %[[#]] = OpBitcast %[[#]] %[[#]]
  ; CHECK-COUNT-3: %[[#]] = OpCompositeExtract %[[#]] %[[#]] [[0-2]]
  %0 = bitcast <3 x double> %D to <6 x i32>
  %1 = shufflevector <6 x i32> %0, <6 x i32> poison, <3 x i32> <i32 0, i32 2, i32 4>
  %2 = shufflevector <6 x i32> %0, <6 x i32> poison, <3 x i32> <i32 1, i32 3, i32 5>
  %add = add <3 x i32> %1, %2
  ret <3 x i32> %add
}
