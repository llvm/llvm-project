; RUN: llc  -mtriple=x86_64-unknown-unknown -mattr=+avx512f,+avx512bw,+avx512vl,+avx512dq -mcpu=znver5 < %s | FileCheck %s
; RUN: llc -update-baseIndex -mtriple=x86_64-unknown-unknown -mattr=+avx512f,+avx512bw,+avx512vl,+avx512dq -mcpu=znver5 < %s | FileCheck %s
; RUN: llc -update-baseIndex=false -mtriple=x86_64-unknown-unknown -mattr=+avx512f,+avx512bw,+avx512vl,+avx512dq -mcpu=znver5 < %s | FileCheck %s -check-prefix=OLD

%struct.pt = type { float, float, float, i32 }

; CHECK-LABEL: test_gather_16f32_1:
; CHECK:   vgatherdps

; OLD-LABEL: test_gather_16f32_1:
; OLD:  vgatherqps
; OLD:  vgatherqps

define <16 x float> @test_gather_16f32_1(ptr %x, ptr %arr, <16 x i1> %mask, <16 x float> %src0)  {
  %wide.load = load <16 x i32>, ptr %arr, align 4
  %4 = and <16 x i32> %wide.load, <i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911>
  %5 = zext <16 x i32> %4 to <16 x i64>
  %ptrs = getelementptr inbounds %struct.pt, ptr %x, <16 x i64> %5
  %res = call <16 x float> @llvm.masked.gather.v16f32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x float> %src0)
  ret <16 x float> %res
  }

; CHECK-LABEL: test_gather_16f32_2:
; CHECK:   vgatherdps

; OLD-LABEL: test_gather_16f32_2:
; OLD:  vgatherqps
; OLD:  vgatherqps

define <16 x float> @test_gather_16f32_2(ptr %x, ptr %arr, <16 x i1> %mask, <16 x float> %src0)  {
  %wide.load = load <16 x i32>, ptr %arr, align 4
  %4 = and <16 x i32> %wide.load, <i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911>
  %5 = zext <16 x i32> %4 to <16 x i64>
  %ptrs = getelementptr inbounds %struct.pt, ptr %x, <16 x i64> %5, i32 1
  %res = call <16 x float> @llvm.masked.gather.v16f32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x float> %src0)
  ret <16 x float> %res
  }
