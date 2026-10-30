; NOTE: Assertions are hand-written and only check spills and ScratchSize; do
; NOTE: not regenerate them with utils/update_llc_test_checks.py.
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -O3 -amdgpu-remat-extend-deps -verify-machineinstrs < %s | FileCheck -check-prefix=EXTEND %s
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -O3 < %s | FileCheck -check-prefix=NOEXTEND %s
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -O3 < %s | grep "Folded Spill" | count 1
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -O3 < %s | grep "Folded Reload" | count 1

; Reduced with llvm-reduce from a real fp8 GEMM kernel (256x256x128 tile, MFMA
; accumulators written back with 16-bit global stores). The address
; computations of the stores (V_LSHL_ADD_U64 on a shared base) are defined
; early and used late, across MFMA code with high register pressure.
;
; "amdgpu-num-vgpr"="112" (224 unified VGPRs on gfx950) sets the register
; budget explicitly; the kernel's LDS usage caps its occupancy at 2, so
; "amdgpu-waves-per-eu" cannot be used to tighten it.
;
; Without -amdgpu-remat-extend-deps, the pre-RA rematerialization stage cannot
; rematerialize the addresses because their base (or another rematerializable
; operand) is not live at the later uses, and the kernel spills. With the flag,
; the addresses are rematerialized next to their later uses and the live range
; of the base is extended, which brings pressure under the budget: no spill.
;
; Interestingness, with -greedy-regclass-priority-trumps-globalness on all llc
; runs (it removed in-loop spill noise in the original kernel, whose loop did
; not survive the reduction; this file gives the same results without it):
;   - the budget attributes are kept;
;   - flag off: the stage runs to reduce spilling, rematerializes no
;     V_LSHL_ADD_U64, and ScratchSize > 0;
;   - flag on, with -verify-machineinstrs: at least 10 V_LSHL_ADD_U64
;     rematerializations and a smaller ScratchSize;
;   - the same number of spills/reloads inside loops with and without the flag.

; EXTEND-NOT: Folded Spill
; EXTEND-NOT: Folded Reload
; EXTEND: ; ScratchSize: 0
; NOEXTEND: ; ScratchSize: 20

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgpu9.50-amd-amdhsa"

define amdgpu_kernel void @_Z20fp8_gemm_256x256x128ILi8192ELi8192ELi8192EEvP14__hip_fp8_e4m3S1_P14__hip_bfloat16(ptr addrspace(1) %0, i32 %1, i128 %2, <4 x i32> %3, <8 x i32> %4, <4 x i32> %5, i32 %6, i32 %7, ptr addrspace(1) %8, i32 %9, i32 %10, i32 %11, i64 %12, i64 %13, <8 x i32> %14, <4 x float> %15, <4 x float> %16, <8 x i32> %17, <4 x float> %18, <4 x float> %19, <8 x i32> %20, <4 x float> %21, <4 x float> %22, <4 x float> %23, <4 x float> %24, <4 x float> %25, <4 x float> %26, ptr addrspace(1) %27, ptr addrspace(1) %28, ptr addrspace(1) %29, ptr addrspace(1) %30) #0 {
  %32 = shufflevector <4 x i32> %3, <4 x i32> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %33 = tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> splat (i32 1), <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0>, <4 x float> zeroinitializer, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %34 = tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> splat (i32 1), <8 x i32> zeroinitializer, <4 x float> zeroinitializer, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  %35 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %36 = and i32 %35, 1
  %37 = or i32 %7, %36
  %38 = zext i32 %37 to i64
  %39 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %38
  store i16 0, ptr addrspace(1) %39, align 2
  %40 = or i32 %36, %6
  %41 = zext i32 %40 to i64
  %42 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %41
  store i16 0, ptr addrspace(1) %42, align 2
  %43 = or i32 %1, %36
  %44 = zext i32 %43 to i64
  %45 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %44
  store i16 0, ptr addrspace(1) %45, align 2
  %46 = or i32 %35, 16384
  %47 = or i32 %46, %1
  %48 = zext i32 %47 to i64
  %49 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %48
  store i16 0, ptr addrspace(1) %49, align 2
  %50 = or i32 %36, 8208
  %51 = or i32 %50, %1
  %52 = zext i32 %51 to i64
  %53 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %52
  store i16 0, ptr addrspace(1) %53, align 2
  %54 = or i32 %35, 16
  %55 = or i32 %1, %54
  %56 = zext i32 %55 to i64
  %57 = getelementptr [2 x i8], ptr addrspace(1) %29, i64 %56
  store i16 0, ptr addrspace(1) %57, align 2
  %58 = or i32 %9, %35
  %59 = zext i32 %58 to i64
  %60 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %59
  store i16 0, ptr addrspace(1) %60, align 2
  %61 = or i32 %35, 131072
  %62 = or i32 %61, %1
  %63 = tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %4, <8 x i32> zeroinitializer, <4 x float> zeroinitializer, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %64 = bitcast <4 x float> %63 to <8 x i16>
  %65 = extractelement <8 x i16> %64, i64 0
  %66 = zext i32 %62 to i64
  %67 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %66
  store i16 %65, ptr addrspace(1) %67, align 2
  %68 = or disjoint i32 %36, 8192
  %69 = or i32 %1, %68
  %70 = zext i32 %69 to i64
  %71 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %70
  store i16 0, ptr addrspace(1) %71, align 2
  %72 = or i32 %35, %11
  %73 = zext i32 %72 to i64
  %74 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %73
  store i16 0, ptr addrspace(1) %74, align 2
  %75 = or i32 %35, 147456
  %76 = or i32 %75, %1
  %77 = zext i32 %76 to i64
  %78 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %77
  store i16 0, ptr addrspace(1) %78, align 2
  %79 = or i32 %10, %35
  %80 = zext i32 %79 to i64
  %81 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %80
  store i16 0, ptr addrspace(1) %81, align 2
  %82 = or i32 %35, %7
  %83 = zext i32 %82 to i64
  %84 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %83
  store i16 0, ptr addrspace(1) %84, align 2
  %85 = shl i32 %35, 11
  %86 = or i32 %85, %6
  %87 = zext i32 %86 to i64
  %88 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %87
  store i16 0, ptr addrspace(1) %88, align 2
  %89 = zext i32 %35 to i64
  %90 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %89
  %91 = or i32 %75, %50
  %92 = zext i32 %91 to i64
  %93 = getelementptr [2 x i8], ptr addrspace(1) %90, i64 %92
  %94 = or i32 %35, 262144
  %95 = or i32 %94, %1
  %96 = zext i32 %95 to i64
  %97 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %96
  store i16 1, ptr addrspace(1) %97, align 2
  %98 = and i32 %85, 98304
  %99 = or i32 %98, %1
  %100 = zext i32 %99 to i64
  %101 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %100
  store i16 0, ptr addrspace(1) %101, align 2
  %102 = or i32 %98, %35
  %103 = zext i32 %102 to i64
  %104 = getelementptr [2 x i8], ptr addrspace(1) %90, i64 %103
  %105 = zext i32 %36 to i64
  %106 = getelementptr [2 x i8], ptr addrspace(1) %90, i64 %105
  store i16 0, ptr addrspace(1) null, align 2
  %107 = zext i32 %94 to i64
  %108 = getelementptr [2 x i8], ptr addrspace(1) %90, i64 %107
  %109 = or i32 %35, %1
  %110 = zext i32 %109 to i64
  %111 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %110
  store i16 0, ptr addrspace(1) %111, align 2
  %112 = or i32 %85, %1
  %113 = zext i32 %112 to i64
  %114 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %113
  store i16 0, ptr addrspace(1) %114, align 2
  %115 = or i32 %85, %36
  %116 = zext i32 %115 to i64
  %117 = getelementptr [2 x i8], ptr addrspace(1) %90, i64 %116
  %118 = or i32 %35, 1
  %119 = or i32 %118, %1
  %120 = zext i32 %119 to i64
  %121 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %120
  store i16 0, ptr addrspace(1) %121, align 2
  %122 = getelementptr [2 x i8], ptr addrspace(1) %90, i64 %12
  store i16 0, ptr addrspace(1) %122, align 2
  %123 = zext i32 %68 to i64
  %124 = getelementptr [2 x i8], ptr addrspace(1) %90, i64 %123
  store i16 0, ptr addrspace(1) %124, align 2
  %125 = or i32 %35, %6
  %126 = zext i32 %125 to i64
  %127 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %126
  store i16 0, ptr addrspace(1) %127, align 2
  %128 = zext i32 %85 to i64
  %129 = getelementptr [2 x i8], ptr addrspace(1) %90, i64 %128
  store i16 0, ptr addrspace(1) %129, align 2
  %130 = getelementptr [2 x i8], ptr addrspace(1) %90, i64 %13
  store i16 0, ptr addrspace(1) %130, align 2
  %131 = tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0>, <4 x float> zeroinitializer, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %132 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0)
  %133 = insertelement <2 x i128> zeroinitializer, i128 %132, i64 0
  %134 = bitcast <2 x i128> %133 to <8 x i32>
  %135 = tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %14, <8 x i32> %134, <4 x float> splat (float +qnan), i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %136 = tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %17, <8 x i32> splat (i32 1), <4 x float> %18, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %137 = tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %19, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %138 = tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %20, <8 x i32> zeroinitializer, <4 x float> %21, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  %139 = bitcast <4 x float> %131 to <8 x i16>
  %140 = extractelement <8 x i16> %139, i64 0
  store i16 %140, ptr addrspace(1) %39, align 2
  %141 = extractelement <8 x i16> %139, i64 3
  store i16 %141, ptr addrspace(1) %42, align 2
  store i16 1, ptr addrspace(1) %45, align 2
  store i16 0, ptr addrspace(1) %49, align 2
  store i16 0, ptr addrspace(1) %53, align 2
  %142 = bitcast <4 x float> %135 to <8 x i16>
  %143 = extractelement <8 x i16> %142, i64 0
  store i16 %143, ptr addrspace(1) %57, align 2
  store i16 0, ptr addrspace(1) %60, align 2
  %144 = tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %32, <8 x i32> zeroinitializer, <4 x float> %15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %145 = bitcast <4 x float> %144 to <8 x i16>
  %146 = extractelement <8 x i16> %145, i64 0
  store i16 %146, ptr addrspace(1) %67, align 2
  store i16 0, ptr addrspace(1) %71, align 2
  store i16 0, ptr addrspace(1) %74, align 2
  store i16 0, ptr addrspace(1) %78, align 2
  %147 = tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %16, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %148 = bitcast <4 x float> %147 to <8 x i16>
  store i16 0, ptr addrspace(1) %81, align 2
  store i16 0, ptr addrspace(1) %84, align 2
  %149 = extractelement <8 x i16> %148, i64 0
  store i16 %149, ptr addrspace(1) %88, align 2
  store i16 0, ptr addrspace(1) %93, align 2
  %150 = bitcast <4 x float> %136 to <8 x i16>
  %151 = extractelement <8 x i16> %150, i64 0
  store i16 %151, ptr addrspace(1) %97, align 2
  store i16 0, ptr addrspace(1) %101, align 2
  store i16 0, ptr addrspace(1) %104, align 2
  %152 = bitcast <4 x float> %137 to <8 x i16>
  store i16 0, ptr addrspace(1) %106, align 2
  %153 = extractelement <8 x i16> %152, i64 0
  store i16 %153, ptr addrspace(1) %108, align 2
  store i16 0, ptr addrspace(1) %111, align 2
  store i16 0, ptr addrspace(1) %114, align 2
  %154 = bitcast <4 x float> %138 to <8 x i16>
  %155 = extractelement <8 x i16> %154, i64 0
  store i16 %155, ptr addrspace(1) %117, align 2
  store i16 0, ptr addrspace(1) %121, align 2
  store i16 0, ptr addrspace(1) %122, align 2
  store i16 0, ptr addrspace(1) %124, align 2
  %156 = tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %22, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %157 = bitcast <4 x float> %156 to <8 x i16>
  %158 = extractelement <8 x i16> %157, i64 0
  store i16 %158, ptr addrspace(1) %127, align 2
  store i16 0, ptr addrspace(1) %129, align 2
  %159 = zext i32 %54 to i64
  %160 = getelementptr [2 x i8], ptr addrspace(1) %8, i64 %159
  store i16 0, ptr addrspace(1) %160, align 2
  store i16 0, ptr addrspace(1) %130, align 2
  %161 = tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %23, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %162 = bitcast <4 x float> %161 to <8 x i16>
  %163 = extractelement <8 x i16> %162, i64 0
  %164 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %38
  store i16 %163, ptr addrspace(1) %164, align 2
  %165 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %41
  store i16 0, ptr addrspace(1) %165, align 2
  %166 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %44
  store i16 0, ptr addrspace(1) %166, align 2
  %167 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %48
  store i16 0, ptr addrspace(1) %167, align 2
  %168 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %52
  store i16 0, ptr addrspace(1) %168, align 2
  %169 = tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %24, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %170 = bitcast <4 x float> %169 to <8 x i16>
  %171 = extractelement <8 x i16> %170, i64 0
  %172 = getelementptr [2 x i8], ptr addrspace(1) %30, i64 %56
  store i16 %171, ptr addrspace(1) %172, align 2
  %173 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %59
  store i16 0, ptr addrspace(1) %173, align 2
  %174 = tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %33, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %175 = bitcast <4 x float> %174 to <8 x i16>
  %176 = extractelement <8 x i16> %175, i64 0
  %177 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %66
  store i16 %176, ptr addrspace(1) %177, align 2
  %178 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %70
  store i16 0, ptr addrspace(1) %178, align 2
  %179 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %73
  store i16 0, ptr addrspace(1) %179, align 2
  %180 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %77
  store i16 0, ptr addrspace(1) %180, align 2
  %181 = tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %34, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %182 = bitcast <4 x float> %181 to <8 x i16>
  %183 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %80
  store i16 0, ptr addrspace(1) %183, align 2
  %184 = extractelement <8 x i16> %182, i64 0
  %185 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %83
  store i16 %184, ptr addrspace(1) %185, align 2
  %186 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %87
  store i16 0, ptr addrspace(1) %186, align 2
  %187 = getelementptr [2 x i8], ptr addrspace(1) %27, i64 %89
  %188 = getelementptr [2 x i8], ptr addrspace(1) %187, i64 %92
  store i16 0, ptr addrspace(1) %188, align 2
  %189 = tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %25, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %190 = bitcast <4 x float> %189 to <8 x i16>
  %191 = extractelement <8 x i16> %190, i64 0
  %192 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %96
  store i16 %191, ptr addrspace(1) %192, align 2
  %193 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %100
  store i16 0, ptr addrspace(1) %193, align 2
  %194 = getelementptr [2 x i8], ptr addrspace(1) %187, i64 %103
  store i16 0, ptr addrspace(1) %194, align 2
  %195 = tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %26, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %196 = bitcast <4 x float> %195 to <8 x i16>
  %197 = extractelement <8 x i16> %196, i64 0
  %198 = getelementptr [2 x i8], ptr addrspace(1) %187, i64 %105
  store i16 %197, ptr addrspace(1) %198, align 2
  %199 = getelementptr [2 x i8], ptr addrspace(1) %187, i64 %107
  store i16 0, ptr addrspace(1) %199, align 2
  %200 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %110
  store i16 0, ptr addrspace(1) %200, align 2
  %201 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %113
  store i16 0, ptr addrspace(1) %201, align 2
  %202 = getelementptr [2 x i8], ptr addrspace(1) %187, i64 %116
  store i16 0, ptr addrspace(1) %202, align 2
  %203 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %120
  store i16 0, ptr addrspace(1) %203, align 2
  %204 = getelementptr [2 x i8], ptr addrspace(1) null, i64 %126
  store i16 0, ptr addrspace(1) %204, align 2
  %205 = getelementptr [2 x i8], ptr addrspace(1) null, i64 %128
  store i16 0, ptr addrspace(1) %205, align 2
  store i16 0, ptr addrspace(1) null, align 2
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.amdgcn.s.setprio(i16 immarg) #2

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32>, <8 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #3

; uselistorder directives
uselistorder ptr @llvm.amdgcn.s.setprio, { 1, 0 }
uselistorder ptr @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32, { 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }

attributes #0 = { "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,512" "amdgpu-num-vgpr"="112" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
