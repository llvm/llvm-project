; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O3 -o - %s | FileCheck %s

; This is a reduced testcase that demonstrates a bug fix in AMDGPURewriteAGPRCopyMFMA's
; eliminateSpillsOfReassignedVGPRs(). When a spill slot has multiple stores in
; different blocks that don't dominate all loads, the pass must skip elimination
; to avoid creating invalid SSA by replacing all references with a single vreg.

; CHECK-LABEL: kernel_crash:

source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@global_smem = external addrspace(3) global [0 x i8]

define amdgpu_kernel void @kernel_crash(i32 inreg %0, ptr addrspace(1) inreg readonly captures(none) %1, i32 inreg %2, i32 %3, i32 %4, i32 %5, i32 %6, ptr addrspace(3) %7, ptr addrspace(3) %global_smem, i1 %8, i32 %9, i32 %10, i32 %11, i32 %12, <8 x half> %13, <8 x half> %14, <8 x half> %15, <8 x half> %16, <8 x half> %17, <8 x half> %18, <8 x half> %19, <8 x half> %20, <8 x half> %21, <8 x half> %22, <8 x half> %23, <8 x half> %24, <8 x half> %25, <8 x half> %26, float %27, ptr addrspace(3) %28, ptr addrspace(3) %29, ptr addrspace(3) %30, ptr addrspace(3) %31, ptr addrspace(3) %32, ptr addrspace(3) %33, ptr addrspace(3) %34, ptr addrspace(3) %35, ptr addrspace(3) %36, ptr addrspace(3) %37, ptr addrspace(3) %38, i1 %exitcond.not, <2 x float> %39, <2 x float> %40, ptr addrspace(3) %41, ptr addrspace(3) %42, ptr addrspace(3) %43, ptr addrspace(3) %44, ptr addrspace(3) %45, ptr addrspace(3) %46, ptr addrspace(3) %47, ptr addrspace(3) %48, ptr addrspace(3) %49, ptr addrspace(3) %50, ptr addrspace(3) %51, ptr addrspace(3) %52, ptr addrspace(3) %53, ptr addrspace(3) %54, ptr addrspace(3) %55, ptr addrspace(3) %56, ptr addrspace(3) %57, ptr addrspace(3) %58, ptr addrspace(3) %59, ptr addrspace(3) %60, i1 %61, <8 x half> %62, <8 x half> %63, <4 x float> %64, <4 x float> %65, <2 x float> %66, <2 x float> %67, i32 %68, <2 x float> %69, <2 x float> %70, <2 x float> %71, <2 x float> %72, <2 x float> %73, <2 x float> %74) #0 {
  %76 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %77 = and i32 %76, 127
  %78 = srem i32 %77, %3
  %79 = or disjoint i32 %77, 1
  %80 = srem i32 %77, %2
  %81 = srem i32 %79, %2
  %82 = add i32 %80, %4
  %83 = add i32 %80, %5
  %84 = add i32 %81, %5
  %85 = add i32 %81, %6
  %86 = shl i32 %0, 0
  %87 = shl nuw nsw i32 %77, 1
  %88 = getelementptr inbounds nuw i8, ptr addrspace(3) %7, i32 %87
  store i16 0, ptr addrspace(3) %88, align 2
  %89 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %87
  %90 = xor i32 %87, 1040
  %91 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %90
  %92 = getelementptr inbounds nuw i8, ptr addrspace(3) %91, i32 33024
  store i16 0, ptr addrspace(3) %92, align 2
  %93 = getelementptr inbounds nuw i8, ptr addrspace(3) %91, i32 49664
  %94 = getelementptr inbounds nuw i8, ptr addrspace(3) %91, i32 49920
  %95 = xor i32 %87, 13520
  %96 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %95
  %97 = getelementptr inbounds nuw i8, ptr addrspace(3) %96, i32 16384
  store i16 0, ptr addrspace(3) %97, align 2
  %98 = getelementptr inbounds nuw i8, ptr addrspace(3) %96, i32 16896
  %99 = getelementptr inbounds nuw i8, ptr addrspace(3) %96, i32 32768
  br i1 %8, label %.lr.ph, label %.._crit_edge_crit_edge

.._crit_edge_crit_edge:                           ; preds = %75
  %.pre = and i32 %76, 1
  br label %._crit_edge

.lr.ph:                                           ; preds = %75
  %100 = and i32 %76, 152
  %101 = xor i32 %100, 0
  %102 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %9
  %103 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 4160
  %104 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 32768
  %105 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 36928
  %106 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 64
  %107 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 4096
  %108 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 256
  %109 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 4416
  %110 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 33024
  %111 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 37184
  %112 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 320
  %113 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 4352
  %114 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 33088
  %115 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 37120
  %116 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 512
  %117 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 4672
  %118 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 33280
  %119 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 37440
  %120 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 576
  %121 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 4608
  %122 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 33344
  %123 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 37376
  %124 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 768
  %125 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 4928
  %126 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 33536
  %127 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 37696
  %128 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 832
  %129 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 4864
  %130 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 33600
  %131 = getelementptr inbounds nuw i8, ptr addrspace(3) %102, i32 37632
  %132 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %10
  %133 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 32768
  %134 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 36928
  %135 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 64
  %136 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 4096
  %137 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 32832
  %138 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 36864
  %139 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 33088
  %140 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 37120
  %141 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 512
  %142 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 37440
  %143 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 576
  %144 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 4608
  %145 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 33344
  %146 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 37376
  %147 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 768
  %148 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 4928
  %149 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 33536
  %150 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 37696
  %151 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 832
  %152 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 4864
  %153 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 33600
  %154 = getelementptr inbounds nuw i8, ptr addrspace(3) %132, i32 37632
  %155 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 65536), i32 %9
  %156 = getelementptr inbounds nuw i8, ptr addrspace(3) %155, i32 1088
  %157 = getelementptr inbounds nuw i8, ptr addrspace(3) %155, i32 8192
  %158 = getelementptr inbounds nuw i8, ptr addrspace(3) %155, i32 9280
  %159 = getelementptr inbounds nuw i8, ptr addrspace(3) %155, i32 64
  %160 = getelementptr inbounds nuw i8, ptr addrspace(3) %155, i32 1024
  %161 = getelementptr inbounds nuw i8, ptr addrspace(3) %155, i32 8256
  %162 = getelementptr inbounds nuw i8, ptr addrspace(3) %155, i32 9216
  %163 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 65536), i32 %11
  %164 = getelementptr inbounds nuw i8, ptr addrspace(3) %163, i32 1088
  %165 = getelementptr inbounds nuw i8, ptr addrspace(3) %163, i32 8192
  %166 = getelementptr inbounds nuw i8, ptr addrspace(3) %163, i32 9280
  %167 = getelementptr inbounds nuw i8, ptr addrspace(3) %163, i32 64
  %168 = getelementptr inbounds nuw i8, ptr addrspace(3) %163, i32 1024
  %169 = getelementptr inbounds nuw i8, ptr addrspace(3) %163, i32 8256
  %170 = getelementptr inbounds nuw i8, ptr addrspace(3) %163, i32 9216
  %171 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 65536), i32 %10
  %172 = getelementptr inbounds nuw i8, ptr addrspace(3) %171, i32 1088
  %173 = getelementptr inbounds nuw i8, ptr addrspace(3) %171, i32 8192
  %174 = getelementptr inbounds nuw i8, ptr addrspace(3) %171, i32 9280
  %175 = getelementptr inbounds nuw i8, ptr addrspace(3) %171, i32 64
  %176 = getelementptr inbounds nuw i8, ptr addrspace(3) %171, i32 1024
  %177 = getelementptr inbounds nuw i8, ptr addrspace(3) %171, i32 8256
  %178 = getelementptr inbounds nuw i8, ptr addrspace(3) %171, i32 9216
  %179 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 65536), i32 %101
  %180 = getelementptr inbounds nuw i8, ptr addrspace(3) %179, i32 1088
  %181 = getelementptr inbounds nuw i8, ptr addrspace(3) %179, i32 8192
  %182 = getelementptr inbounds nuw i8, ptr addrspace(3) %179, i32 9280
  %183 = getelementptr inbounds nuw i8, ptr addrspace(3) %179, i32 64
  %184 = getelementptr inbounds nuw i8, ptr addrspace(3) %179, i32 1024
  %185 = getelementptr inbounds nuw i8, ptr addrspace(3) %179, i32 8256
  %186 = getelementptr inbounds nuw i8, ptr addrspace(3) %179, i32 9216
  br label %187

187:                                              ; preds = %187, %.lr.ph
  %.pn5751951 = phi i32 [ %78, %.lr.ph ], [ %343, %187 ]
  %188 = phi float [ 0.000000e+00, %.lr.ph ], [ %560, %187 ]
  %189 = phi float [ 0.000000e+00, %.lr.ph ], [ %561, %187 ]
  %190 = phi float [ 0.000000e+00, %.lr.ph ], [ %562, %187 ]
  %191 = phi float [ 0.000000e+00, %.lr.ph ], [ %569, %187 ]
  %192 = phi float [ 0.000000e+00, %.lr.ph ], [ %570, %187 ]
  %193 = phi float [ 0.000000e+00, %.lr.ph ], [ %571, %187 ]
  %194 = phi float [ 0.000000e+00, %.lr.ph ], [ %572, %187 ]
  %195 = phi float [ 0.000000e+00, %.lr.ph ], [ %579, %187 ]
  %196 = phi float [ 0.000000e+00, %.lr.ph ], [ %580, %187 ]
  %197 = phi float [ 0.000000e+00, %.lr.ph ], [ %581, %187 ]
  %198 = phi float [ 0.000000e+00, %.lr.ph ], [ %585, %187 ]
  %199 = phi float [ 0.000000e+00, %.lr.ph ], [ %590, %187 ]
  %200 = phi float [ 0.000000e+00, %.lr.ph ], [ %591, %187 ]
  %201 = phi float [ 0.000000e+00, %.lr.ph ], [ %598, %187 ]
  %202 = phi float [ 0.000000e+00, %.lr.ph ], [ %599, %187 ]
  %203 = phi float [ 0.000000e+00, %.lr.ph ], [ %600, %187 ]
  %204 = phi float [ 0.000000e+00, %.lr.ph ], [ %601, %187 ]
  %205 = phi float [ 0.000000e+00, %.lr.ph ], [ %606, %187 ]
  %206 = phi float [ 0.000000e+00, %.lr.ph ], [ %607, %187 ]
  %207 = phi float [ 0.000000e+00, %.lr.ph ], [ %614, %187 ]
  %208 = phi float [ 0.000000e+00, %.lr.ph ], [ %615, %187 ]
  %209 = phi float [ 0.000000e+00, %.lr.ph ], [ %616, %187 ]
  %210 = phi float [ 0.000000e+00, %.lr.ph ], [ %622, %187 ]
  %211 = phi float [ 0.000000e+00, %.lr.ph ], [ %623, %187 ]
  %212 = phi float [ 0.000000e+00, %.lr.ph ], [ %630, %187 ]
  %213 = phi float [ 0.000000e+00, %.lr.ph ], [ %631, %187 ]
  %214 = phi float [ 0.000000e+00, %.lr.ph ], [ %632, %187 ]
  %215 = phi float [ 0.000000e+00, %.lr.ph ], [ %633, %187 ]
  %216 = phi float [ 0.000000e+00, %.lr.ph ], [ %637, %187 ]
  %217 = phi float [ 0.000000e+00, %.lr.ph ], [ %644, %187 ]
  %218 = phi float [ 0.000000e+00, %.lr.ph ], [ %645, %187 ]
  %219 = phi float [ 0.000000e+00, %.lr.ph ], [ %646, %187 ]
  %220 = phi float [ 0.000000e+00, %.lr.ph ], [ %647, %187 ]
  %221 = phi float [ 0.000000e+00, %.lr.ph ], [ %654, %187 ]
  %222 = phi float [ 0.000000e+00, %.lr.ph ], [ %655, %187 ]
  %223 = phi float [ 0.000000e+00, %.lr.ph ], [ %656, %187 ]
  %224 = phi float [ 0.000000e+00, %.lr.ph ], [ %657, %187 ]
  %225 = phi float [ 0.000000e+00, %.lr.ph ], [ %664, %187 ]
  %226 = phi float [ 0.000000e+00, %.lr.ph ], [ %665, %187 ]
  %227 = phi float [ 0.000000e+00, %.lr.ph ], [ %666, %187 ]
  %228 = phi float [ 0.000000e+00, %.lr.ph ], [ %667, %187 ]
  %229 = phi float [ 0.000000e+00, %.lr.ph ], [ %674, %187 ]
  %230 = phi float [ 0.000000e+00, %.lr.ph ], [ %675, %187 ]
  %231 = phi float [ 0.000000e+00, %.lr.ph ], [ %676, %187 ]
  %232 = phi float [ 0.000000e+00, %.lr.ph ], [ %677, %187 ]
  %233 = phi float [ 0.000000e+00, %.lr.ph ], [ %684, %187 ]
  %234 = phi float [ 0.000000e+00, %.lr.ph ], [ %685, %187 ]
  %235 = phi float [ 0.000000e+00, %.lr.ph ], [ %686, %187 ]
  %236 = phi float [ 0.000000e+00, %.lr.ph ], [ %691, %187 ]
  %237 = phi float [ 0.000000e+00, %.lr.ph ], [ %692, %187 ]
  %238 = phi float [ 0.000000e+00, %.lr.ph ], [ %696, %187 ]
  %239 = phi float [ 0.000000e+00, %.lr.ph ], [ %703, %187 ]
  %240 = phi float [ 0.000000e+00, %.lr.ph ], [ %704, %187 ]
  %241 = phi float [ 0.000000e+00, %.lr.ph ], [ %705, %187 ]
  %242 = phi float [ 0.000000e+00, %.lr.ph ], [ %706, %187 ]
  %243 = phi float [ 0.000000e+00, %.lr.ph ], [ %714, %187 ]
  %244 = phi float [ 0.000000e+00, %.lr.ph ], [ %715, %187 ]
  %245 = phi float [ 0.000000e+00, %.lr.ph ], [ %716, %187 ]
  %246 = phi float [ 0.000000e+00, %.lr.ph ], [ %717, %187 ]
  %247 = phi float [ 0.000000e+00, %.lr.ph ], [ %724, %187 ]
  %248 = phi float [ 0.000000e+00, %.lr.ph ], [ %725, %187 ]
  %249 = phi float [ 0.000000e+00, %.lr.ph ], [ %726, %187 ]
  %250 = phi float [ 0.000000e+00, %.lr.ph ], [ %727, %187 ]
  %251 = phi float [ 0.000000e+00, %.lr.ph ], [ %734, %187 ]
  %252 = phi float [ 0.000000e+00, %.lr.ph ], [ %735, %187 ]
  %253 = phi float [ 0.000000e+00, %.lr.ph ], [ %736, %187 ]
  %254 = phi float [ 0.000000e+00, %.lr.ph ], [ %737, %187 ]
  %255 = phi float [ 0.000000e+00, %.lr.ph ], [ %744, %187 ]
  %256 = phi float [ 0.000000e+00, %.lr.ph ], [ %745, %187 ]
  %257 = phi float [ 0.000000e+00, %.lr.ph ], [ %746, %187 ]
  %258 = phi float [ 0.000000e+00, %.lr.ph ], [ %747, %187 ]
  %259 = phi float [ 0.000000e+00, %.lr.ph ], [ %754, %187 ]
  %260 = phi float [ 0.000000e+00, %.lr.ph ], [ %755, %187 ]
  %261 = phi float [ 0.000000e+00, %.lr.ph ], [ %756, %187 ]
  %262 = phi float [ 0.000000e+00, %.lr.ph ], [ %757, %187 ]
  %263 = phi float [ 0.000000e+00, %.lr.ph ], [ %764, %187 ]
  %264 = phi float [ 0.000000e+00, %.lr.ph ], [ %765, %187 ]
  %265 = phi float [ 0.000000e+00, %.lr.ph ], [ %766, %187 ]
  %266 = phi float [ 0.000000e+00, %.lr.ph ], [ %767, %187 ]
  %267 = phi float [ 0.000000e+00, %.lr.ph ], [ %774, %187 ]
  %268 = phi float [ 0.000000e+00, %.lr.ph ], [ %775, %187 ]
  %269 = phi float [ 0.000000e+00, %.lr.ph ], [ %776, %187 ]
  %270 = phi float [ 0.000000e+00, %.lr.ph ], [ %777, %187 ]
  %271 = phi float [ 0.000000e+00, %.lr.ph ], [ %784, %187 ]
  %272 = phi float [ 0.000000e+00, %.lr.ph ], [ %785, %187 ]
  %273 = phi float [ 0.000000e+00, %.lr.ph ], [ %786, %187 ]
  %274 = phi float [ 0.000000e+00, %.lr.ph ], [ %787, %187 ]
  %275 = phi float [ 0.000000e+00, %.lr.ph ], [ %794, %187 ]
  %276 = phi float [ 0.000000e+00, %.lr.ph ], [ %795, %187 ]
  %277 = phi float [ 0.000000e+00, %.lr.ph ], [ %796, %187 ]
  %278 = phi float [ 0.000000e+00, %.lr.ph ], [ %797, %187 ]
  %279 = phi float [ 0.000000e+00, %.lr.ph ], [ %804, %187 ]
  %280 = phi float [ 0.000000e+00, %.lr.ph ], [ %805, %187 ]
  %281 = phi float [ 0.000000e+00, %.lr.ph ], [ %806, %187 ]
  %282 = phi float [ 0.000000e+00, %.lr.ph ], [ %807, %187 ]
  %283 = phi float [ 0.000000e+00, %.lr.ph ], [ %814, %187 ]
  %284 = phi float [ 0.000000e+00, %.lr.ph ], [ %815, %187 ]
  %285 = phi float [ 0.000000e+00, %.lr.ph ], [ %816, %187 ]
  %286 = phi float [ 0.000000e+00, %.lr.ph ], [ %817, %187 ]
  %287 = phi float [ 0.000000e+00, %.lr.ph ], [ %824, %187 ]
  %288 = phi float [ 0.000000e+00, %.lr.ph ], [ %825, %187 ]
  %289 = phi float [ 0.000000e+00, %.lr.ph ], [ %826, %187 ]
  %290 = phi float [ 0.000000e+00, %.lr.ph ], [ %827, %187 ]
  %291 = phi float [ 0.000000e+00, %.lr.ph ], [ %834, %187 ]
  %292 = phi float [ 0.000000e+00, %.lr.ph ], [ %835, %187 ]
  %293 = phi float [ 0.000000e+00, %.lr.ph ], [ %836, %187 ]
  %294 = phi float [ 0.000000e+00, %.lr.ph ], [ %837, %187 ]
  %295 = phi float [ 0.000000e+00, %.lr.ph ], [ %844, %187 ]
  %296 = phi float [ 0.000000e+00, %.lr.ph ], [ %855, %187 ]
  %297 = phi float [ 0.000000e+00, %.lr.ph ], [ %856, %187 ]
  %298 = phi float [ 0.000000e+00, %.lr.ph ], [ %857, %187 ]
  %299 = phi float [ 0.000000e+00, %.lr.ph ], [ %858, %187 ]
  %300 = phi float [ 0.000000e+00, %.lr.ph ], [ %867, %187 ]
  %301 = phi float [ 0.000000e+00, %.lr.ph ], [ %868, %187 ]
  %302 = phi float [ 0.000000e+00, %.lr.ph ], [ %869, %187 ]
  %303 = phi float [ 0.000000e+00, %.lr.ph ], [ %870, %187 ]
  %304 = phi float [ 0.000000e+00, %.lr.ph ], [ %877, %187 ]
  %305 = phi float [ 0.000000e+00, %.lr.ph ], [ %878, %187 ]
  %306 = phi float [ 0.000000e+00, %.lr.ph ], [ %883, %187 ]
  %307 = phi float [ 0.000000e+00, %.lr.ph ], [ %884, %187 ]
  %308 = phi float [ 0.000000e+00, %.lr.ph ], [ %891, %187 ]
  %309 = phi float [ 0.000000e+00, %.lr.ph ], [ %892, %187 ]
  %310 = phi float [ 0.000000e+00, %.lr.ph ], [ %893, %187 ]
  %311 = phi float [ 0.000000e+00, %.lr.ph ], [ %894, %187 ]
  %312 = phi float [ 0.000000e+00, %.lr.ph ], [ %901, %187 ]
  %313 = phi float [ 0.000000e+00, %.lr.ph ], [ %902, %187 ]
  %314 = phi float [ 0.000000e+00, %.lr.ph ], [ %903, %187 ]
  %315 = phi float [ 0.000000e+00, %.lr.ph ], [ %904, %187 ]
  %316 = phi float [ 0.000000e+00, %.lr.ph ], [ %911, %187 ]
  %317 = phi float [ 0.000000e+00, %.lr.ph ], [ %912, %187 ]
  %318 = phi float [ 0.000000e+00, %.lr.ph ], [ %913, %187 ]
  %319 = phi float [ 0.000000e+00, %.lr.ph ], [ %914, %187 ]
  %320 = phi float [ 0.000000e+00, %.lr.ph ], [ %921, %187 ]
  %321 = phi float [ 0.000000e+00, %.lr.ph ], [ %922, %187 ]
  %322 = phi float [ 0.000000e+00, %.lr.ph ], [ %923, %187 ]
  %323 = phi float [ 0.000000e+00, %.lr.ph ], [ %924, %187 ]
  %324 = phi float [ 0.000000e+00, %.lr.ph ], [ %931, %187 ]
  %325 = phi float [ 0.000000e+00, %.lr.ph ], [ %932, %187 ]
  %326 = phi float [ 0.000000e+00, %.lr.ph ], [ %933, %187 ]
  %327 = phi float [ 0.000000e+00, %.lr.ph ], [ %934, %187 ]
  %328 = phi float [ 0.000000e+00, %.lr.ph ], [ %941, %187 ]
  %329 = phi float [ 0.000000e+00, %.lr.ph ], [ %942, %187 ]
  %330 = phi float [ 0.000000e+00, %.lr.ph ], [ %943, %187 ]
  %331 = phi float [ 0.000000e+00, %.lr.ph ], [ %944, %187 ]
  %332 = phi float [ 0.000000e+00, %.lr.ph ], [ %951, %187 ]
  %333 = phi float [ 0.000000e+00, %.lr.ph ], [ %952, %187 ]
  %334 = phi float [ 0.000000e+00, %.lr.ph ], [ %953, %187 ]
  %335 = phi float [ 0.000000e+00, %.lr.ph ], [ %954, %187 ]
  %336 = phi float [ 0.000000e+00, %.lr.ph ], [ %962, %187 ]
  %337 = phi float [ 0.000000e+00, %.lr.ph ], [ %969, %187 ]
  %338 = phi float [ 0.000000e+00, %.lr.ph ], [ %970, %187 ]
  %339 = phi float [ 0.000000e+00, %.lr.ph ], [ %971, %187 ]
  %340 = phi float [ 0.000000e+00, %.lr.ph ], [ %972, %187 ]
  %341 = phi float [ 0.000000e+00, %.lr.ph ], [ %978, %187 ]
  %342 = phi float [ 0.000000e+00, %.lr.ph ], [ %979, %187 ]
  %.pn1151861 = phi i32 [ %6, %.lr.ph ], [ %354, %187 ]
  %.pn1171860 = phi i32 [ %85, %.lr.ph ], [ %353, %187 ]
  %.pn1691834 = phi i32 [ %0, %.lr.ph ], [ %352, %187 ]
  %.pn2691784 = phi i32 [ %81, %.lr.ph ], [ %351, %187 ]
  %.pn2711783 = phi i32 [ %80, %.lr.ph ], [ %350, %187 ]
  %.pn2751781 = phi i32 [ %5, %.lr.ph ], [ %349, %187 ]
  %.pn2771780 = phi i32 [ %84, %.lr.ph ], [ %348, %187 ]
  %.pn2791779 = phi i32 [ %83, %.lr.ph ], [ %347, %187 ]
  %.pn2951771 = phi i32 [ %82, %.lr.ph ], [ %346, %187 ]
  %.pn3471745 = phi i32 [ %12, %.lr.ph ], [ %345, %187 ]
  %.pn3491744 = phi i32 [ 0, %.lr.ph ], [ %344, %187 ]
  %343 = add i32 %.pn5751951, 1
  %344 = add i32 %.pn3491744, %86
  %345 = add i32 %.pn3471745, %86
  %346 = add i32 %.pn2951771, %86
  %347 = add i32 %.pn2791779, %86
  %348 = add i32 %.pn2771780, %86
  %349 = add i32 %.pn2751781, %86
  %350 = add i32 %.pn2711783, %86
  %351 = add i32 %.pn2691784, %86
  %352 = add i32 %.pn1691834, %86
  %353 = add i32 %.pn1171860, %86
  %354 = add i32 %.pn1151861, %86
  %355 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %343, i32 0, i32 0)
  tail call void @llvm.amdgcn.s.waitcnt(i32 0)
  tail call void @llvm.amdgcn.s.barrier()
  %356 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) %155)
  %357 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %156)
  %358 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %157)
  %359 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %158)
  %360 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %159)
  %361 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %160)
  %362 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %161)
  %363 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %162)
  %364 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) %163)
  %365 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %164)
  %366 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %165)
  %367 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %166)
  %368 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %167)
  %369 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %168)
  %370 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %169)
  %371 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %170)
  %372 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) %171)
  %373 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %172)
  %374 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %173)
  %375 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %174)
  %376 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %175)
  %377 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %176)
  %378 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %177)
  %379 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %178)
  %380 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) %179)
  %381 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %180)
  %382 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %181)
  %383 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %182)
  %384 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %183)
  %385 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %184)
  %386 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %185)
  %387 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %186)
  %388 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) %102)
  %389 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %103)
  %390 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %104)
  %391 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %105)
  %392 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %106)
  %393 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %107)
  %394 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %108)
  %395 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %109)
  %396 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %110)
  %397 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %111)
  %398 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %112)
  %399 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %113)
  %400 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %114)
  %401 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %115)
  %402 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %116)
  %403 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %117)
  %404 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %118)
  %405 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %119)
  %406 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %120)
  %407 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %121)
  %408 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %122)
  %409 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %123)
  %410 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %124)
  %411 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %125)
  %412 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %126)
  %413 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %127)
  %414 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %128)
  %415 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %129)
  %416 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %130)
  %417 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %131)
  %418 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %133)
  %419 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %134)
  %420 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %135)
  %421 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %136)
  %422 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %137)
  %423 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %138)
  %424 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %139)
  %425 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %140)
  %426 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %141)
  %427 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %142)
  %428 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %143)
  %429 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %144)
  %430 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %145)
  %431 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %146)
  %432 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %147)
  %433 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %148)
  %434 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %149)
  %435 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %150)
  %436 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %151)
  %437 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %152)
  %438 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %153)
  %439 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %154)
  %440 = shl i32 %344, 1
  %441 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %440, i32 0, i32 0)
  %442 = shl i32 %345, 1
  %443 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %442, i32 0, i32 0)
  %444 = shl i32 %346, 1
  %445 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %444, i32 0, i32 0)
  %446 = shl i32 %347, 1
  %447 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %446, i32 0, i32 0)
  %448 = shl i32 %348, 1
  %449 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %448, i32 0, i32 0)
  %450 = shl i32 %349, 1
  %451 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %450, i32 0, i32 0)
  %452 = shl i32 %350, 1
  %453 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %452, i32 0, i32 0)
  %454 = shl i32 %351, 1
  %455 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %454, i32 0, i32 0)
  %456 = shl i32 %352, 1
  %457 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %456, i32 0, i32 0)
  %458 = shl i32 %353, 1
  %459 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %458, i32 0, i32 0)
  %460 = shl i32 %354, 1
  %461 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %460, i32 0, i32 0)
  %462 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %463 = shufflevector <4 x half> %356, <4 x half> %357, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %464 = shufflevector <4 x half> %358, <4 x half> %359, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %465 = shufflevector <4 x half> %364, <4 x half> %365, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %466 = shufflevector <4 x half> %366, <4 x half> %367, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %467 = shufflevector <4 x half> %360, <4 x half> %361, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %468 = shufflevector <4 x half> %362, <4 x half> %363, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %469 = shufflevector <4 x half> %368, <4 x half> %369, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %470 = shufflevector <4 x half> %370, <4 x half> %371, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %471 = shufflevector <4 x half> %372, <4 x half> %373, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %472 = shufflevector <4 x half> %374, <4 x half> %375, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %473 = shufflevector <4 x half> %380, <4 x half> %381, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %474 = shufflevector <4 x half> %382, <4 x half> %383, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %475 = shufflevector <4 x half> %376, <4 x half> %377, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %476 = shufflevector <4 x half> %378, <4 x half> %379, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %477 = shufflevector <4 x half> %384, <4 x half> %385, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %478 = shufflevector <4 x half> %386, <4 x half> %387, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %479 = shufflevector <4 x half> %388, <4 x half> %389, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %480 = shufflevector <4 x half> %390, <4 x half> %391, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %481 = shufflevector <4 x half> %392, <4 x half> %393, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %482 = shufflevector <4 x half> %418, <4 x half> %419, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %483 = shufflevector <4 x half> %420, <4 x half> %421, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %484 = shufflevector <4 x half> %422, <4 x half> %423, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %485 = shufflevector <4 x half> %394, <4 x half> %395, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %486 = shufflevector <4 x half> %396, <4 x half> %397, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %487 = shufflevector <4 x half> %398, <4 x half> %399, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %488 = shufflevector <4 x half> %400, <4 x half> %401, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %489 = shufflevector <4 x half> %424, <4 x half> %425, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %490 = shufflevector <4 x half> %402, <4 x half> %403, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %491 = shufflevector <4 x half> %404, <4 x half> %405, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %492 = shufflevector <4 x half> %406, <4 x half> %407, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %493 = shufflevector <4 x half> %408, <4 x half> %409, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %494 = shufflevector <4 x half> %426, <4 x half> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %495 = shufflevector <4 x half> zeroinitializer, <4 x half> %427, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %496 = shufflevector <4 x half> %428, <4 x half> %429, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %497 = shufflevector <4 x half> %430, <4 x half> %431, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %498 = shufflevector <4 x half> %410, <4 x half> %411, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %499 = shufflevector <4 x half> %412, <4 x half> %413, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %500 = shufflevector <4 x half> %414, <4 x half> %415, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %501 = shufflevector <4 x half> %416, <4 x half> %417, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %502 = shufflevector <4 x half> %432, <4 x half> %433, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %503 = shufflevector <4 x half> %434, <4 x half> %435, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %504 = shufflevector <4 x half> %436, <4 x half> %437, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %505 = shufflevector <4 x half> %438, <4 x half> %439, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %506 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %463, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %507 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %506, i32 0, i32 0, i32 0)
  %508 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %490, <8 x half> zeroinitializer, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %509 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %508, i32 0, i32 0, i32 0)
  %510 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %493, <8 x half> zeroinitializer, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %511 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %494, <8 x half> zeroinitializer, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %512 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %464, <4 x float> %511, i32 0, i32 0, i32 0)
  %513 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %14, <8 x half> %463, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %514 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %497, <8 x half> zeroinitializer, <4 x float> %513, i32 0, i32 0, i32 0)
  %515 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %15, <8 x half> %463, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %516 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %515, i32 0, i32 0, i32 0)
  %517 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %16, <8 x half> %463, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %518 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %517, i32 0, i32 0, i32 0)
  %519 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %17, <8 x half> %463, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %520 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %519, i32 0, i32 0, i32 0)
  %521 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %18, <8 x half> %463, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %522 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %521, i32 0, i32 0, i32 0)
  %523 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %19, <8 x half> %465, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %524 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %466, <4 x float> %523, i32 0, i32 0, i32 0)
  %525 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %465, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %526 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %525, i32 0, i32 0, i32 0)
  %527 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %20, <8 x half> %465, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %528 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %527, i32 0, i32 0, i32 0)
  %529 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %494, <8 x half> %21, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %530 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %495, <8 x half> zeroinitializer, <4 x float> %529, i32 0, i32 0, i32 0)
  %531 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %14, <8 x half> %465, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %532 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %531, i32 0, i32 0, i32 0)
  %533 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %17, <8 x half> %465, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %534 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %533, i32 0, i32 0, i32 0)
  %535 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %18, <8 x half> %465, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %536 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %505, <8 x half> %466, <4 x float> %535, i32 0, i32 0, i32 0)
  %537 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %467, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %538 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %22, <8 x half> zeroinitializer, <4 x float> %537, i32 0, i32 0, i32 0)
  %539 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %23, <8 x half> %467, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %540 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %539, i32 0, i32 0, i32 0)
  %541 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %19, <8 x half> %467, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %542 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %541, i32 0, i32 0, i32 0)
  %543 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %467, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %544 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %24, <4 x float> %543, i32 0, i32 0, i32 0)
  %545 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %467, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %546 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %545, i32 0, i32 0, i32 0)
  %547 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %20, <8 x half> %467, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %548 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %547, i32 0, i32 0, i32 0)
  %549 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %492, <8 x half> %467, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %550 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %468, <4 x float> %549, i32 0, i32 0, i32 0)
  %551 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %494, <8 x half> %467, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %552 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %495, <8 x half> zeroinitializer, <4 x float> %551, i32 0, i32 0, i32 0)
  %553 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %496, <8 x half> %25, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %554 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %553, i32 0, i32 0, i32 0)
  %555 = insertelement <4 x float> zeroinitializer, float %188, i64 1
  %556 = insertelement <4 x float> %555, float %189, i64 2
  %557 = insertelement <4 x float> %556, float %190, i64 3
  %558 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %498, <8 x half> zeroinitializer, <4 x float> %557, i32 0, i32 0, i32 0)
  %559 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %499, <8 x half> %468, <4 x float> %558, i32 0, i32 0, i32 0)
  %560 = extractelement <4 x float> %559, i64 1
  %561 = extractelement <4 x float> %559, i64 2
  %562 = extractelement <4 x float> %559, i64 3
  %563 = insertelement <4 x float> zeroinitializer, float %191, i64 0
  %564 = insertelement <4 x float> %563, float %192, i64 1
  %565 = insertelement <4 x float> %564, float %193, i64 2
  %566 = insertelement <4 x float> %565, float %194, i64 3
  %567 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %467, <4 x float> %566, i32 0, i32 0, i32 0)
  %568 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %468, <4 x float> %567, i32 0, i32 0, i32 0)
  %569 = extractelement <4 x float> %568, i64 0
  %570 = extractelement <4 x float> %568, i64 1
  %571 = extractelement <4 x float> %568, i64 2
  %572 = extractelement <4 x float> %568, i64 3
  %573 = insertelement <4 x float> zeroinitializer, float %195, i64 0
  %574 = insertelement <4 x float> %573, float %196, i64 1
  %575 = insertelement <4 x float> %574, float %197, i64 2
  %576 = insertelement <4 x float> %575, float 0.000000e+00, i64 3
  %577 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %576, i32 0, i32 0, i32 0)
  %578 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %577, i32 0, i32 0, i32 0)
  %579 = extractelement <4 x float> %578, i64 0
  %580 = extractelement <4 x float> %578, i64 1
  %581 = extractelement <4 x float> %578, i64 2
  %582 = insertelement <4 x float> zeroinitializer, float %198, i64 3
  %583 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %504, <8 x half> zeroinitializer, <4 x float> %582, i32 0, i32 0, i32 0)
  %584 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %583, i32 0, i32 0, i32 0)
  %585 = extractelement <4 x float> %584, i64 3
  %586 = insertelement <4 x float> zeroinitializer, float %199, i64 2
  %587 = insertelement <4 x float> %586, float %200, i64 3
  %588 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %587, i32 0, i32 0, i32 0)
  %589 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %470, <4 x float> %588, i32 0, i32 0, i32 0)
  %590 = extractelement <4 x float> %589, i64 2
  %591 = extractelement <4 x float> %589, i64 3
  %592 = insertelement <4 x float> zeroinitializer, float %201, i64 0
  %593 = insertelement <4 x float> %592, float %202, i64 1
  %594 = insertelement <4 x float> %593, float %203, i64 2
  %595 = insertelement <4 x float> %594, float %204, i64 3
  %596 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %469, <4 x float> %595, i32 0, i32 0, i32 0)
  %597 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %470, <4 x float> %596, i32 0, i32 0, i32 0)
  %598 = extractelement <4 x float> %597, i64 0
  %599 = extractelement <4 x float> %597, i64 1
  %600 = extractelement <4 x float> %597, i64 2
  %601 = extractelement <4 x float> %597, i64 3
  %602 = insertelement <4 x float> zeroinitializer, float %205, i64 0
  %603 = insertelement <4 x float> zeroinitializer, float %206, i64 3
  %604 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %603, i32 0, i32 0, i32 0)
  %605 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %604, i32 0, i32 0, i32 0)
  %606 = extractelement <4 x float> %605, i64 0
  %607 = extractelement <4 x float> %605, i64 3
  %608 = insertelement <4 x float> zeroinitializer, float %207, i64 0
  %609 = insertelement <4 x float> %608, float %208, i64 1
  %610 = insertelement <4 x float> %609, float %209, i64 2
  %611 = insertelement <4 x float> %610, float 0.000000e+00, i64 3
  %612 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %611, i32 0, i32 0, i32 0)
  %613 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %484, <8 x half> zeroinitializer, <4 x float> %612, i32 0, i32 0, i32 0)
  %614 = extractelement <4 x float> %613, i64 0
  %615 = extractelement <4 x float> %613, i64 1
  %616 = extractelement <4 x float> %613, i64 2
  %617 = insertelement <4 x float> zeroinitializer, float %210, i64 1
  %618 = insertelement <4 x float> %617, float %211, i64 2
  %619 = insertelement <4 x float> %618, float %27, i64 0
  %620 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %485, <8 x half> zeroinitializer, <4 x float> %619, i32 0, i32 0, i32 0)
  %621 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %620, i32 0, i32 0, i32 0)
  %622 = extractelement <4 x float> %621, i64 1
  %623 = extractelement <4 x float> %621, i64 2
  %624 = insertelement <4 x float> zeroinitializer, float %212, i64 0
  %625 = insertelement <4 x float> %624, float %213, i64 1
  %626 = insertelement <4 x float> %625, float %214, i64 2
  %627 = insertelement <4 x float> %626, float %215, i64 3
  %628 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %487, <8 x half> %469, <4 x float> %627, i32 0, i32 0, i32 0)
  %629 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %488, <8 x half> %470, <4 x float> %628, i32 0, i32 0, i32 0)
  %630 = extractelement <4 x float> %629, i64 0
  %631 = extractelement <4 x float> %629, i64 1
  %632 = extractelement <4 x float> %629, i64 2
  %633 = extractelement <4 x float> %629, i64 3
  %634 = insertelement <4 x float> zeroinitializer, float %216, i64 3
  %635 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %634, i32 0, i32 0, i32 0)
  %636 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %635, i32 0, i32 0, i32 0)
  %637 = extractelement <4 x float> %636, i64 3
  %638 = insertelement <4 x float> zeroinitializer, float %217, i64 0
  %639 = insertelement <4 x float> %638, float %218, i64 1
  %640 = insertelement <4 x float> %639, float %219, i64 2
  %641 = insertelement <4 x float> %640, float %220, i64 3
  %642 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %469, <4 x float> %641, i32 0, i32 0, i32 0)
  %643 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %470, <4 x float> %642, i32 0, i32 0, i32 0)
  %644 = extractelement <4 x float> %643, i64 0
  %645 = extractelement <4 x float> %643, i64 1
  %646 = extractelement <4 x float> %643, i64 2
  %647 = extractelement <4 x float> %643, i64 3
  %648 = insertelement <4 x float> zeroinitializer, float %221, i64 0
  %649 = insertelement <4 x float> %648, float %222, i64 1
  %650 = insertelement <4 x float> %649, float %223, i64 2
  %651 = insertelement <4 x float> %650, float %224, i64 3
  %652 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %469, <4 x float> %651, i32 0, i32 0, i32 0)
  %653 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %491, <8 x half> %470, <4 x float> %652, i32 0, i32 0, i32 0)
  %654 = extractelement <4 x float> %653, i64 0
  %655 = extractelement <4 x float> %653, i64 1
  %656 = extractelement <4 x float> %653, i64 2
  %657 = extractelement <4 x float> %653, i64 3
  %658 = insertelement <4 x float> zeroinitializer, float %225, i64 0
  %659 = insertelement <4 x float> %658, float %226, i64 1
  %660 = insertelement <4 x float> %659, float %227, i64 2
  %661 = insertelement <4 x float> %660, float %228, i64 3
  %662 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %469, <4 x float> %661, i32 0, i32 0, i32 0)
  %663 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %470, <4 x float> %662, i32 0, i32 0, i32 0)
  %664 = extractelement <4 x float> %663, i64 0
  %665 = extractelement <4 x float> %663, i64 1
  %666 = extractelement <4 x float> %663, i64 2
  %667 = extractelement <4 x float> %663, i64 3
  %668 = insertelement <4 x float> zeroinitializer, float %229, i64 0
  %669 = insertelement <4 x float> %668, float %230, i64 1
  %670 = insertelement <4 x float> %669, float %231, i64 2
  %671 = insertelement <4 x float> %670, float %232, i64 3
  %672 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %494, <8 x half> %469, <4 x float> %671, i32 0, i32 0, i32 0)
  %673 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %495, <8 x half> zeroinitializer, <4 x float> %672, i32 0, i32 0, i32 0)
  %674 = extractelement <4 x float> %673, i64 0
  %675 = extractelement <4 x float> %673, i64 1
  %676 = extractelement <4 x float> %673, i64 2
  %677 = extractelement <4 x float> %673, i64 3
  %678 = insertelement <4 x float> zeroinitializer, float %233, i64 0
  %679 = insertelement <4 x float> %678, float %234, i64 1
  %680 = insertelement <4 x float> %679, float %235, i64 2
  %681 = insertelement <4 x float> %680, float 0.000000e+00, i64 3
  %682 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %681, i32 0, i32 0, i32 0)
  %683 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %682, i32 0, i32 0, i32 0)
  %684 = extractelement <4 x float> %683, i64 0
  %685 = extractelement <4 x float> %683, i64 1
  %686 = extractelement <4 x float> %683, i64 2
  %687 = insertelement <4 x float> zeroinitializer, float %236, i64 2
  %688 = insertelement <4 x float> %687, float %237, i64 3
  %689 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %502, <8 x half> zeroinitializer, <4 x float> %688, i32 0, i32 0, i32 0)
  %690 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %503, <8 x half> zeroinitializer, <4 x float> %689, i32 0, i32 0, i32 0)
  %691 = extractelement <4 x float> %690, i64 2
  %692 = extractelement <4 x float> %690, i64 3
  %693 = insertelement <4 x float> zeroinitializer, float %238, i64 3
  %694 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %693, i32 0, i32 0, i32 0)
  %695 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %694, i32 0, i32 0, i32 0)
  %696 = extractelement <4 x float> %695, i64 3
  %697 = insertelement <4 x float> zeroinitializer, float %239, i64 0
  %698 = insertelement <4 x float> %697, float %240, i64 1
  %699 = insertelement <4 x float> %698, float %241, i64 2
  %700 = insertelement <4 x float> %699, float %242, i64 3
  %701 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %479, <8 x half> zeroinitializer, <4 x float> %700, i32 0, i32 0, i32 0)
  %702 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %480, <8 x half> %472, <4 x float> %701, i32 0, i32 0, i32 0)
  %703 = extractelement <4 x float> %702, i64 0
  %704 = extractelement <4 x float> %702, i64 1
  %705 = extractelement <4 x float> %702, i64 2
  %706 = extractelement <4 x float> %702, i64 3
  %707 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %472, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %708 = insertelement <4 x float> zeroinitializer, float %243, i64 0
  %709 = insertelement <4 x float> %708, float %244, i64 1
  %710 = insertelement <4 x float> %709, float %245, i64 2
  %711 = insertelement <4 x float> %710, float %246, i64 3
  %712 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %471, <4 x float> %711, i32 0, i32 0, i32 0)
  %713 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %482, <8 x half> %472, <4 x float> %712, i32 0, i32 0, i32 0)
  %714 = extractelement <4 x float> %713, i64 0
  %715 = extractelement <4 x float> %713, i64 1
  %716 = extractelement <4 x float> %713, i64 2
  %717 = extractelement <4 x float> %713, i64 3
  %718 = insertelement <4 x float> zeroinitializer, float %247, i64 0
  %719 = insertelement <4 x float> %718, float %248, i64 1
  %720 = insertelement <4 x float> %719, float %249, i64 2
  %721 = insertelement <4 x float> %720, float %250, i64 3
  %722 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %483, <8 x half> zeroinitializer, <4 x float> %721, i32 0, i32 0, i32 0)
  %723 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %722, i32 0, i32 0, i32 0)
  %724 = extractelement <4 x float> %723, i64 0
  %725 = extractelement <4 x float> %723, i64 1
  %726 = extractelement <4 x float> %723, i64 2
  %727 = extractelement <4 x float> %723, i64 3
  %728 = insertelement <4 x float> zeroinitializer, float %251, i64 0
  %729 = insertelement <4 x float> %728, float %252, i64 1
  %730 = insertelement <4 x float> %729, float %253, i64 2
  %731 = insertelement <4 x float> %730, float %254, i64 3
  %732 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %471, <4 x float> %731, i32 0, i32 0, i32 0)
  %733 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %486, <8 x half> %472, <4 x float> %732, i32 0, i32 0, i32 0)
  %734 = extractelement <4 x float> %733, i64 0
  %735 = extractelement <4 x float> %733, i64 1
  %736 = extractelement <4 x float> %733, i64 2
  %737 = extractelement <4 x float> %733, i64 3
  %738 = insertelement <4 x float> zeroinitializer, float %255, i64 0
  %739 = insertelement <4 x float> %738, float %256, i64 1
  %740 = insertelement <4 x float> %739, float %257, i64 2
  %741 = insertelement <4 x float> %740, float %258, i64 3
  %742 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %487, <8 x half> %471, <4 x float> %741, i32 0, i32 0, i32 0)
  %743 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %472, <4 x float> %742, i32 0, i32 0, i32 0)
  %744 = extractelement <4 x float> %743, i64 0
  %745 = extractelement <4 x float> %743, i64 1
  %746 = extractelement <4 x float> %743, i64 2
  %747 = extractelement <4 x float> %743, i64 3
  %748 = insertelement <4 x float> zeroinitializer, float %259, i64 0
  %749 = insertelement <4 x float> %748, float %260, i64 1
  %750 = insertelement <4 x float> %749, float %261, i64 2
  %751 = insertelement <4 x float> %750, float %262, i64 3
  %752 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %471, <4 x float> %751, i32 0, i32 0, i32 0)
  %753 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %472, <4 x float> %752, i32 0, i32 0, i32 0)
  %754 = extractelement <4 x float> %753, i64 0
  %755 = extractelement <4 x float> %753, i64 1
  %756 = extractelement <4 x float> %753, i64 2
  %757 = extractelement <4 x float> %753, i64 3
  %758 = insertelement <4 x float> zeroinitializer, float %263, i64 0
  %759 = insertelement <4 x float> %758, float %264, i64 1
  %760 = insertelement <4 x float> %759, float %265, i64 2
  %761 = insertelement <4 x float> %760, float %266, i64 3
  %762 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %471, <4 x float> %761, i32 0, i32 0, i32 0)
  %763 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %489, <8 x half> %472, <4 x float> %762, i32 0, i32 0, i32 0)
  %764 = extractelement <4 x float> %763, i64 0
  %765 = extractelement <4 x float> %763, i64 1
  %766 = extractelement <4 x float> %763, i64 2
  %767 = extractelement <4 x float> %763, i64 3
  %768 = insertelement <4 x float> zeroinitializer, float %267, i64 0
  %769 = insertelement <4 x float> %768, float %268, i64 1
  %770 = insertelement <4 x float> %769, float %269, i64 2
  %771 = insertelement <4 x float> %770, float %270, i64 3
  %772 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %471, <4 x float> %771, i32 0, i32 0, i32 0)
  %773 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %472, <4 x float> %772, i32 0, i32 0, i32 0)
  %774 = extractelement <4 x float> %773, i64 0
  %775 = extractelement <4 x float> %773, i64 1
  %776 = extractelement <4 x float> %773, i64 2
  %777 = extractelement <4 x float> %773, i64 3
  %778 = insertelement <4 x float> zeroinitializer, float %271, i64 0
  %779 = insertelement <4 x float> %778, float %272, i64 1
  %780 = insertelement <4 x float> %779, float %273, i64 2
  %781 = insertelement <4 x float> %780, float %274, i64 3
  %782 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %471, <4 x float> %781, i32 0, i32 0, i32 0)
  %783 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %472, <4 x float> %782, i32 0, i32 0, i32 0)
  %784 = extractelement <4 x float> %783, i64 0
  %785 = extractelement <4 x float> %783, i64 1
  %786 = extractelement <4 x float> %783, i64 2
  %787 = extractelement <4 x float> %783, i64 3
  %788 = insertelement <4 x float> zeroinitializer, float %275, i64 0
  %789 = insertelement <4 x float> %788, float %276, i64 1
  %790 = insertelement <4 x float> %789, float %277, i64 2
  %791 = insertelement <4 x float> %790, float %278, i64 3
  %792 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %494, <8 x half> %471, <4 x float> %791, i32 0, i32 0, i32 0)
  %793 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %472, <4 x float> %792, i32 0, i32 0, i32 0)
  %794 = extractelement <4 x float> %793, i64 0
  %795 = extractelement <4 x float> %793, i64 1
  %796 = extractelement <4 x float> %793, i64 2
  %797 = extractelement <4 x float> %793, i64 3
  %798 = insertelement <4 x float> zeroinitializer, float %279, i64 0
  %799 = insertelement <4 x float> %798, float %280, i64 1
  %800 = insertelement <4 x float> %799, float %281, i64 2
  %801 = insertelement <4 x float> %800, float %282, i64 3
  %802 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %471, <4 x float> %801, i32 0, i32 0, i32 0)
  %803 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %472, <4 x float> %802, i32 0, i32 0, i32 0)
  %804 = extractelement <4 x float> %803, i64 0
  %805 = extractelement <4 x float> %803, i64 1
  %806 = extractelement <4 x float> %803, i64 2
  %807 = extractelement <4 x float> %803, i64 3
  %808 = insertelement <4 x float> zeroinitializer, float %283, i64 0
  %809 = insertelement <4 x float> %808, float %284, i64 1
  %810 = insertelement <4 x float> %809, float %285, i64 2
  %811 = insertelement <4 x float> %810, float %286, i64 3
  %812 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %471, <4 x float> %811, i32 0, i32 0, i32 0)
  %813 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %812, i32 0, i32 0, i32 0)
  %814 = extractelement <4 x float> %813, i64 0
  %815 = extractelement <4 x float> %813, i64 1
  %816 = extractelement <4 x float> %813, i64 2
  %817 = extractelement <4 x float> %813, i64 3
  %818 = insertelement <4 x float> zeroinitializer, float %287, i64 0
  %819 = insertelement <4 x float> %818, float %288, i64 1
  %820 = insertelement <4 x float> %819, float %289, i64 2
  %821 = insertelement <4 x float> %820, float %290, i64 3
  %822 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %500, <8 x half> %471, <4 x float> %821, i32 0, i32 0, i32 0)
  %823 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %501, <8 x half> zeroinitializer, <4 x float> %822, i32 0, i32 0, i32 0)
  %824 = extractelement <4 x float> %823, i64 0
  %825 = extractelement <4 x float> %823, i64 1
  %826 = extractelement <4 x float> %823, i64 2
  %827 = extractelement <4 x float> %823, i64 3
  %828 = insertelement <4 x float> zeroinitializer, float %291, i64 0
  %829 = insertelement <4 x float> %828, float %292, i64 1
  %830 = insertelement <4 x float> %829, float %293, i64 2
  %831 = insertelement <4 x float> %830, float %294, i64 3
  %832 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %471, <4 x float> %831, i32 0, i32 0, i32 0)
  %833 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %472, <4 x float> %832, i32 0, i32 0, i32 0)
  %834 = extractelement <4 x float> %833, i64 0
  %835 = extractelement <4 x float> %833, i64 1
  %836 = extractelement <4 x float> %833, i64 2
  %837 = extractelement <4 x float> %833, i64 3
  %838 = insertelement <4 x float> zeroinitializer, float %295, i64 0
  %839 = insertelement <4 x float> %838, float 0.000000e+00, i64 1
  %840 = insertelement <4 x float> %839, float 0.000000e+00, i64 1
  %841 = insertelement <4 x float> %840, float 0.000000e+00, i64 1
  %842 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %841, i32 0, i32 0, i32 0)
  %843 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %842, i32 0, i32 0, i32 0)
  %844 = extractelement <4 x float> %843, i64 0
  %845 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %479, <8 x half> zeroinitializer, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %846 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %845, i32 0, i32 0, i32 0)
  %847 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %19, <8 x half> %473, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %848 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %847, i32 0, i32 0, i32 0)
  %849 = insertelement <4 x float> zeroinitializer, float %296, i64 0
  %850 = insertelement <4 x float> %849, float %297, i64 1
  %851 = insertelement <4 x float> %850, float %298, i64 2
  %852 = insertelement <4 x float> %851, float %299, i64 3
  %853 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %473, <4 x float> %852, i32 0, i32 0, i32 0)
  %854 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %474, <4 x float> %853, i32 0, i32 0, i32 0)
  %855 = extractelement <4 x float> %854, i64 0
  %856 = extractelement <4 x float> %854, i64 1
  %857 = extractelement <4 x float> %854, i64 2
  %858 = extractelement <4 x float> %854, i64 3
  %859 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %473, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %860 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %859, i32 0, i32 0, i32 0)
  %861 = insertelement <4 x float> zeroinitializer, float %300, i64 0
  %862 = insertelement <4 x float> %861, float %301, i64 1
  %863 = insertelement <4 x float> %862, float %302, i64 2
  %864 = insertelement <4 x float> %863, float %303, i64 3
  %865 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %473, <4 x float> %864, i32 0, i32 0, i32 0)
  %866 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %865, i32 0, i32 0, i32 0)
  %867 = extractelement <4 x float> %866, i64 0
  %868 = extractelement <4 x float> %866, i64 1
  %869 = extractelement <4 x float> %866, i64 2
  %870 = extractelement <4 x float> %866, i64 3
  %871 = insertelement <4 x float> zeroinitializer, float %304, i64 0
  %872 = insertelement <4 x float> %871, float %305, i64 1
  %873 = insertelement <4 x float> %872, float 0.000000e+00, i64 2
  %874 = insertelement <4 x float> %873, float 0.000000e+00, i64 3
  %875 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %874, i32 0, i32 0, i32 0)
  %876 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %875, i32 0, i32 0, i32 0)
  %877 = extractelement <4 x float> %876, i64 0
  %878 = extractelement <4 x float> %876, i64 1
  %879 = insertelement <4 x float> zeroinitializer, float %306, i64 0
  %880 = insertelement <4 x float> zeroinitializer, float %307, i64 3
  %881 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %475, <4 x float> %880, i32 0, i32 0, i32 0)
  %882 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %476, <4 x float> %881, i32 0, i32 0, i32 0)
  %883 = extractelement <4 x float> %882, i64 0
  %884 = extractelement <4 x float> %882, i64 3
  %885 = insertelement <4 x float> zeroinitializer, float %308, i64 0
  %886 = insertelement <4 x float> %885, float %309, i64 1
  %887 = insertelement <4 x float> %886, float %310, i64 2
  %888 = insertelement <4 x float> %887, float %311, i64 3
  %889 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %477, <4 x float> %888, i32 0, i32 0, i32 0)
  %890 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %478, <4 x float> %889, i32 0, i32 0, i32 0)
  %891 = extractelement <4 x float> %890, i64 0
  %892 = extractelement <4 x float> %890, i64 1
  %893 = extractelement <4 x float> %890, i64 2
  %894 = extractelement <4 x float> %890, i64 3
  %895 = insertelement <4 x float> zeroinitializer, float %312, i64 0
  %896 = insertelement <4 x float> %895, float %313, i64 1
  %897 = insertelement <4 x float> %896, float %314, i64 2
  %898 = insertelement <4 x float> %897, float %315, i64 3
  %899 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %481, <8 x half> %477, <4 x float> %898, i32 0, i32 0, i32 0)
  %900 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %899, i32 0, i32 0, i32 0)
  %901 = extractelement <4 x float> %900, i64 0
  %902 = extractelement <4 x float> %900, i64 1
  %903 = extractelement <4 x float> %900, i64 2
  %904 = extractelement <4 x float> %900, i64 3
  %905 = insertelement <4 x float> zeroinitializer, float %316, i64 0
  %906 = insertelement <4 x float> %905, float %317, i64 1
  %907 = insertelement <4 x float> %906, float %318, i64 2
  %908 = insertelement <4 x float> %907, float %319, i64 3
  %909 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %477, <4 x float> %908, i32 0, i32 0, i32 0)
  %910 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %909, i32 0, i32 0, i32 0)
  %911 = extractelement <4 x float> %910, i64 0
  %912 = extractelement <4 x float> %910, i64 1
  %913 = extractelement <4 x float> %910, i64 2
  %914 = extractelement <4 x float> %910, i64 3
  %915 = insertelement <4 x float> zeroinitializer, float %320, i64 0
  %916 = insertelement <4 x float> %915, float %321, i64 1
  %917 = insertelement <4 x float> %916, float %322, i64 2
  %918 = insertelement <4 x float> %917, float %323, i64 3
  %919 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %483, <8 x half> zeroinitializer, <4 x float> %918, i32 0, i32 0, i32 0)
  %920 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %919, i32 0, i32 0, i32 0)
  %921 = extractelement <4 x float> %920, i64 0
  %922 = extractelement <4 x float> %920, i64 1
  %923 = extractelement <4 x float> %920, i64 2
  %924 = extractelement <4 x float> %920, i64 3
  %925 = insertelement <4 x float> zeroinitializer, float %324, i64 0
  %926 = insertelement <4 x float> %925, float %325, i64 1
  %927 = insertelement <4 x float> %926, float %326, i64 2
  %928 = insertelement <4 x float> %927, float %327, i64 3
  %929 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %485, <8 x half> zeroinitializer, <4 x float> %928, i32 0, i32 0, i32 0)
  %930 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %929, i32 0, i32 0, i32 0)
  %931 = extractelement <4 x float> %930, i64 0
  %932 = extractelement <4 x float> %930, i64 1
  %933 = extractelement <4 x float> %930, i64 2
  %934 = extractelement <4 x float> %930, i64 3
  %935 = insertelement <4 x float> zeroinitializer, float %328, i64 0
  %936 = insertelement <4 x float> %935, float %329, i64 1
  %937 = insertelement <4 x float> %936, float %330, i64 2
  %938 = insertelement <4 x float> %937, float %331, i64 3
  %939 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %487, <8 x half> zeroinitializer, <4 x float> %938, i32 0, i32 0, i32 0)
  %940 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %939, i32 0, i32 0, i32 0)
  %941 = extractelement <4 x float> %940, i64 0
  %942 = extractelement <4 x float> %940, i64 1
  %943 = extractelement <4 x float> %940, i64 2
  %944 = extractelement <4 x float> %940, i64 3
  %945 = insertelement <4 x float> zeroinitializer, float %332, i64 0
  %946 = insertelement <4 x float> %945, float %333, i64 1
  %947 = insertelement <4 x float> %946, float %334, i64 2
  %948 = insertelement <4 x float> %947, float %335, i64 3
  %949 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %477, <4 x float> %948, i32 0, i32 0, i32 0)
  %950 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %949, i32 0, i32 0, i32 0)
  %951 = extractelement <4 x float> %950, i64 0
  %952 = extractelement <4 x float> %950, i64 1
  %953 = extractelement <4 x float> %950, i64 2
  %954 = extractelement <4 x float> %950, i64 3
  %955 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %495, <8 x half> zeroinitializer, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %956 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %496, <8 x half> zeroinitializer, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %957 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %956, i32 0, i32 0, i32 0)
  %958 = insertelement <4 x float> zeroinitializer, float %336, i64 2
  %959 = insertelement <4 x float> %958, float 0.000000e+00, i64 0
  %960 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %500, <8 x half> zeroinitializer, <4 x float> %959, i32 0, i32 0, i32 0)
  %961 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %960, i32 0, i32 0, i32 0)
  %962 = extractelement <4 x float> %961, i64 2
  %963 = insertelement <4 x float> zeroinitializer, float %337, i64 0
  %964 = insertelement <4 x float> %963, float %338, i64 1
  %965 = insertelement <4 x float> %964, float %339, i64 2
  %966 = insertelement <4 x float> %965, float %340, i64 3
  %967 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %502, <8 x half> zeroinitializer, <4 x float> %966, i32 0, i32 0, i32 0)
  %968 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %967, i32 0, i32 0, i32 0)
  %969 = extractelement <4 x float> %968, i64 0
  %970 = extractelement <4 x float> %968, i64 1
  %971 = extractelement <4 x float> %968, i64 2
  %972 = extractelement <4 x float> %968, i64 3
  %973 = insertelement <4 x float> zeroinitializer, float %341, i64 1
  %974 = insertelement <4 x float> %973, float %342, i64 2
  %975 = insertelement <4 x float> %974, float 0.000000e+00, i64 0
  %976 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %504, <8 x half> zeroinitializer, <4 x float> %975, i32 0, i32 0, i32 0)
  %977 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %976, i32 0, i32 0, i32 0)
  %978 = extractelement <4 x float> %977, i64 1
  %979 = extractelement <4 x float> %977, i64 2
  tail call void @llvm.amdgcn.s.waitcnt(i32 0)
  tail call void @llvm.amdgcn.s.barrier()
  store i16 0, ptr addrspace(3) %88, align 2
  store i16 %355, ptr addrspace(3) null, align 2
  store i16 0, ptr addrspace(3) %28, align 2
  store i16 0, ptr addrspace(3) null, align 2
  store i16 0, ptr addrspace(3) %29, align 2
  store i16 0, ptr addrspace(3) null, align 2
  store i16 0, ptr addrspace(3) %30, align 2
  store i16 0, ptr addrspace(3) null, align 2
  store i16 0, ptr addrspace(3) %31, align 2
  store i16 0, ptr addrspace(3) %32, align 2
  store i16 0, ptr addrspace(3) null, align 2
  store i16 0, ptr addrspace(3) %7, align 2
  store i16 0, ptr addrspace(3) %33, align 2
  store i16 0, ptr addrspace(3) %34, align 2
  store i16 0, ptr addrspace(3) %89, align 2
  store i16 0, ptr addrspace(3) %92, align 2
  store i16 %459, ptr addrspace(3) null, align 2
  store i16 %461, ptr addrspace(3) %93, align 2
  store i16 0, ptr addrspace(3) %94, align 2
  store i16 0, ptr addrspace(3) %35, align 2
  store i16 0, ptr addrspace(3) null, align 2
  store i16 %441, ptr addrspace(3) %36, align 2
  store i16 %443, ptr addrspace(3) null, align 2
  store i16 0, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i32 17152), align 2
  store i16 %457, ptr addrspace(3) null, align 2
  store i16 %445, ptr addrspace(3) %37, align 2
  store i16 %447, ptr addrspace(3) %97, align 2
  store i16 %449, ptr addrspace(3) null, align 2
  store i16 %451, ptr addrspace(3) %98, align 2
  store i16 0, ptr addrspace(3) %99, align 2
  store i16 %453, ptr addrspace(3) null, align 2
  store i16 %455, ptr addrspace(3) %38, align 2
  store i16 0, ptr addrspace(3) null, align 2
  store i16 0, ptr addrspace(3) inttoptr (i32 32768 to ptr addrspace(3)), align 2
  store i16 0, ptr addrspace(3) inttoptr (i32 33024 to ptr addrspace(3)), align 2
  store i16 0, ptr addrspace(3) inttoptr (i32 33280 to ptr addrspace(3)), align 2
  store i16 0, ptr addrspace(3) null, align 2
  store i16 0, ptr addrspace(3) inttoptr (i32 49664 to ptr addrspace(3)), align 2
  store i16 %462, ptr addrspace(3) inttoptr (i32 49920 to ptr addrspace(3)), align 2
  br i1 %exitcond.not, label %._crit_edge.loopexit, label %187

._crit_edge.loopexit:                             ; preds = %187
  %980 = shufflevector <4 x float> %507, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %981 = shufflevector <4 x float> %509, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %982 = shufflevector <4 x float> %509, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %983 = shufflevector <4 x float> %510, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %984 = shufflevector <4 x float> %512, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %985 = shufflevector <4 x float> %512, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %986 = shufflevector <4 x float> %514, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %987 = shufflevector <4 x float> %514, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %988 = shufflevector <4 x float> %516, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %989 = shufflevector <4 x float> %518, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %990 = shufflevector <4 x float> %518, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %991 = shufflevector <4 x float> %520, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %992 = shufflevector <4 x float> %522, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %993 = shufflevector <4 x float> %524, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %994 = shufflevector <4 x float> %526, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %995 = shufflevector <4 x float> %526, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %996 = shufflevector <4 x float> %528, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %997 = shufflevector <4 x float> %528, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %998 = shufflevector <4 x float> %530, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %999 = shufflevector <4 x float> %532, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1000 = shufflevector <4 x float> %534, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1001 = shufflevector <4 x float> %536, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1002 = shufflevector <4 x float> %538, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1003 = shufflevector <4 x float> %540, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1004 = shufflevector <4 x float> %542, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1005 = shufflevector <4 x float> %542, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1006 = shufflevector <4 x float> %544, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1007 = shufflevector <4 x float> %546, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1008 = shufflevector <4 x float> %548, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1009 = shufflevector <4 x float> %550, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1010 = shufflevector <4 x float> %550, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1011 = shufflevector <4 x float> %552, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1012 = shufflevector <4 x float> %552, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1013 = shufflevector <4 x float> %554, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1014 = shufflevector <4 x float> %559, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1015 = shufflevector <4 x float> %568, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1016 = shufflevector <4 x float> %578, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1017 = shufflevector <4 x float> %584, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1018 = shufflevector <4 x float> %589, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1019 = shufflevector <4 x float> %597, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1020 = shufflevector <4 x float> %613, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1021 = shufflevector <4 x float> %621, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1022 = shufflevector <4 x float> %629, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1023 = shufflevector <4 x float> %636, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1024 = shufflevector <4 x float> %636, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1025 = shufflevector <4 x float> %643, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1026 = shufflevector <4 x float> %653, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1027 = shufflevector <4 x float> %663, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1028 = shufflevector <4 x float> %673, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1029 = shufflevector <4 x float> %683, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1030 = shufflevector <4 x float> %690, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1031 = shufflevector <4 x float> %695, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1032 = shufflevector <4 x float> %702, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1033 = shufflevector <4 x float> %707, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1034 = shufflevector <4 x float> %713, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1035 = shufflevector <4 x float> %723, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1036 = shufflevector <4 x float> %733, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1037 = shufflevector <4 x float> %743, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1038 = shufflevector <4 x float> %743, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1039 = shufflevector <4 x float> %763, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1040 = shufflevector <4 x float> %773, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1041 = shufflevector <4 x float> %783, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1042 = shufflevector <4 x float> %793, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1043 = shufflevector <4 x float> %803, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1044 = shufflevector <4 x float> %813, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1045 = shufflevector <4 x float> %823, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1046 = shufflevector <4 x float> %833, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1047 = shufflevector <4 x float> %843, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1048 = shufflevector <4 x float> %846, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1049 = shufflevector <4 x float> %846, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1050 = shufflevector <4 x float> %848, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1051 = shufflevector <4 x float> %848, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1052 = shufflevector <4 x float> %860, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1053 = shufflevector <4 x float> %866, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1054 = shufflevector <4 x float> %866, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1055 = shufflevector <4 x float> %890, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1056 = shufflevector <4 x float> %890, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1057 = shufflevector <4 x float> %900, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1058 = shufflevector <4 x float> %930, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1059 = shufflevector <4 x float> %940, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1060 = shufflevector <4 x float> %955, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1061 = shufflevector <4 x float> %957, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1062 = shufflevector <4 x float> %957, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1063 = shufflevector <4 x float> %961, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge.loopexit, %.._crit_edge_crit_edge
  %.pre-phi2009 = phi i32 [ %.pre, %.._crit_edge_crit_edge ], [ %101, %._crit_edge.loopexit ]
  %1064 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1063, %._crit_edge.loopexit ]
  %1065 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1062, %._crit_edge.loopexit ]
  %1066 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1061, %._crit_edge.loopexit ]
  %1067 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1060, %._crit_edge.loopexit ]
  %1068 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1059, %._crit_edge.loopexit ]
  %1069 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1058, %._crit_edge.loopexit ]
  %1070 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1057, %._crit_edge.loopexit ]
  %1071 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1056, %._crit_edge.loopexit ]
  %1072 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1055, %._crit_edge.loopexit ]
  %1073 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1054, %._crit_edge.loopexit ]
  %1074 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1053, %._crit_edge.loopexit ]
  %1075 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1052, %._crit_edge.loopexit ]
  %1076 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1051, %._crit_edge.loopexit ]
  %1077 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1050, %._crit_edge.loopexit ]
  %1078 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1049, %._crit_edge.loopexit ]
  %1079 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1048, %._crit_edge.loopexit ]
  %1080 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1047, %._crit_edge.loopexit ]
  %1081 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1046, %._crit_edge.loopexit ]
  %1082 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1045, %._crit_edge.loopexit ]
  %1083 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1044, %._crit_edge.loopexit ]
  %1084 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1043, %._crit_edge.loopexit ]
  %1085 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1042, %._crit_edge.loopexit ]
  %1086 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1041, %._crit_edge.loopexit ]
  %1087 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1040, %._crit_edge.loopexit ]
  %1088 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1039, %._crit_edge.loopexit ]
  %1089 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1038, %._crit_edge.loopexit ]
  %1090 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1037, %._crit_edge.loopexit ]
  %1091 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1036, %._crit_edge.loopexit ]
  %1092 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1035, %._crit_edge.loopexit ]
  %1093 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1034, %._crit_edge.loopexit ]
  %1094 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1033, %._crit_edge.loopexit ]
  %1095 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1032, %._crit_edge.loopexit ]
  %1096 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1031, %._crit_edge.loopexit ]
  %1097 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1030, %._crit_edge.loopexit ]
  %1098 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1029, %._crit_edge.loopexit ]
  %1099 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1028, %._crit_edge.loopexit ]
  %1100 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1027, %._crit_edge.loopexit ]
  %1101 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1026, %._crit_edge.loopexit ]
  %1102 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1025, %._crit_edge.loopexit ]
  %1103 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1024, %._crit_edge.loopexit ]
  %1104 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1023, %._crit_edge.loopexit ]
  %1105 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1022, %._crit_edge.loopexit ]
  %1106 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1021, %._crit_edge.loopexit ]
  %1107 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1020, %._crit_edge.loopexit ]
  %1108 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1019, %._crit_edge.loopexit ]
  %1109 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1018, %._crit_edge.loopexit ]
  %1110 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1017, %._crit_edge.loopexit ]
  %1111 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1016, %._crit_edge.loopexit ]
  %1112 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1015, %._crit_edge.loopexit ]
  %1113 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1014, %._crit_edge.loopexit ]
  %1114 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %39, %._crit_edge.loopexit ]
  %1115 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1013, %._crit_edge.loopexit ]
  %1116 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1012, %._crit_edge.loopexit ]
  %1117 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1011, %._crit_edge.loopexit ]
  %1118 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1010, %._crit_edge.loopexit ]
  %1119 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1009, %._crit_edge.loopexit ]
  %1120 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1008, %._crit_edge.loopexit ]
  %1121 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1007, %._crit_edge.loopexit ]
  %1122 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1006, %._crit_edge.loopexit ]
  %1123 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1005, %._crit_edge.loopexit ]
  %1124 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1004, %._crit_edge.loopexit ]
  %1125 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1003, %._crit_edge.loopexit ]
  %1126 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1002, %._crit_edge.loopexit ]
  %1127 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1001, %._crit_edge.loopexit ]
  %1128 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1000, %._crit_edge.loopexit ]
  %1129 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %999, %._crit_edge.loopexit ]
  %1130 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %998, %._crit_edge.loopexit ]
  %1131 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %997, %._crit_edge.loopexit ]
  %1132 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %996, %._crit_edge.loopexit ]
  %1133 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %995, %._crit_edge.loopexit ]
  %1134 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %994, %._crit_edge.loopexit ]
  %1135 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %993, %._crit_edge.loopexit ]
  %1136 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %992, %._crit_edge.loopexit ]
  %1137 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %991, %._crit_edge.loopexit ]
  %1138 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %990, %._crit_edge.loopexit ]
  %1139 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %989, %._crit_edge.loopexit ]
  %1140 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %988, %._crit_edge.loopexit ]
  %1141 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %987, %._crit_edge.loopexit ]
  %1142 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %986, %._crit_edge.loopexit ]
  %1143 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %985, %._crit_edge.loopexit ]
  %1144 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %984, %._crit_edge.loopexit ]
  %1145 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %40, %._crit_edge.loopexit ]
  %1146 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %983, %._crit_edge.loopexit ]
  %1147 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %982, %._crit_edge.loopexit ]
  %1148 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %981, %._crit_edge.loopexit ]
  %1149 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %980, %._crit_edge.loopexit ]
  %1150 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %42)
  %1151 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %43)
  %1152 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull inttoptr (i32 64 to ptr addrspace(3)))
  %1153 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull inttoptr (i32 1024 to ptr addrspace(3)))
  %1154 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %44)
  %1155 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %45)
  %1156 = getelementptr inbounds nuw i8, ptr addrspace(3) %7, i32 %.pre-phi2009
  %1157 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) %1156)
  %1158 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %46)
  %1159 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %47)
  %1160 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %48)
  %1161 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %49)
  %1162 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %50)
  %1163 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %51)
  %1164 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %52)
  %1165 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull inttoptr (i32 4160 to ptr addrspace(3)))
  %1166 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull inttoptr (i32 32768 to ptr addrspace(3)))
  %1167 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull inttoptr (i32 36928 to ptr addrspace(3)))
  %1168 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull inttoptr (i32 4096 to ptr addrspace(3)))
  %1169 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull inttoptr (i32 32832 to ptr addrspace(3)))
  %1170 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %53)
  %1171 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %54)
  %1172 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %55)
  %1173 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %56)
  %1174 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull inttoptr (i32 832 to ptr addrspace(3)))
  %1175 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %57)
  %1176 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull inttoptr (i32 33600 to ptr addrspace(3)))
  %1177 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) @global_smem)
  %1178 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %58)
  %1179 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %59)
  %1180 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i32 36928))
  %1181 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i32 32832))
  %1182 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i32 36864))
  %1183 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i32 256))
  %1184 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i32 4416))
  %1185 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i32 33024))
  %1186 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i32 37184))
  %1187 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i32 320))
  %1188 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i32 37120))
  %1189 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i32 576))
  %1190 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull %60)
  %1191 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i32 37376))
  %1192 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull null)
  %1193 = tail call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) nonnull getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i32 37632))
  br i1 %61, label %1194, label %._crit_edge._crit_edge

1194:                                             ; preds = %._crit_edge
  %1195 = shufflevector <4 x half> %1150, <4 x half> %1151, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1196 = shufflevector <4 x half> %1157, <4 x half> %1158, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1197 = shufflevector <4 x half> %1159, <4 x half> %1160, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1198 = shufflevector <4 x half> %1152, <4 x half> %1153, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1199 = shufflevector <4 x half> %1154, <4 x half> %1155, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1200 = shufflevector <4 x half> %1161, <4 x half> %1162, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1201 = shufflevector <4 x half> %1163, <4 x half> %1164, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1202 = shufflevector <4 x half> splat (half 0xH3C00), <4 x half> %1165, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1203 = shufflevector <4 x half> %1166, <4 x half> %1167, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1204 = shufflevector <4 x half> splat (half 0xH3C00), <4 x half> %1168, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1205 = shufflevector <4 x half> %1169, <4 x half> %1170, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1206 = shufflevector <4 x half> %1177, <4 x half> %1178, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1207 = shufflevector <4 x half> %1179, <4 x half> %1180, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1208 = shufflevector <4 x half> %1181, <4 x half> %1182, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1209 = shufflevector <4 x half> splat (half 0xH3C00), <4 x half> %1171, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1210 = shufflevector <4 x half> %1172, <4 x half> %1173, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1211 = shufflevector <4 x half> %1183, <4 x half> %1184, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1212 = shufflevector <4 x half> %1185, <4 x half> %1186, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1213 = shufflevector <4 x half> %1187, <4 x half> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1214 = shufflevector <4 x half> zeroinitializer, <4 x half> %1188, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1215 = shufflevector <4 x half> %1189, <4 x half> %1190, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1216 = shufflevector <4 x half> zeroinitializer, <4 x half> %1191, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1217 = shufflevector <4 x half> %1174, <4 x half> %1175, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1218 = shufflevector <4 x half> %1176, <4 x half> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1219 = shufflevector <4 x half> %1192, <4 x half> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1220 = shufflevector <4 x half> zeroinitializer, <4 x half> %1193, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1221 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1219, <8 x half> %62, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %1222 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1220, <8 x half> %1195, <4 x float> %1221, i32 0, i32 0, i32 0)
  %1223 = shufflevector <2 x float> %1079, <2 x float> %1078, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1224 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1202, <8 x half> %1196, <4 x float> %1223, i32 0, i32 0, i32 0)
  %1225 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1203, <8 x half> %1197, <4 x float> %1224, i32 0, i32 0, i32 0)
  %1226 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1204, <8 x half> %1196, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %1227 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1205, <8 x half> %1197, <4 x float> %1226, i32 0, i32 0, i32 0)
  %1228 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1206, <8 x half> %1196, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %1229 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1207, <8 x half> %1197, <4 x float> %1228, i32 0, i32 0, i32 0)
  %1230 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1209, <8 x half> %1196, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %1231 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %1197, <4 x float> %1230, i32 0, i32 0, i32 0)
  %1232 = shufflevector <2 x float> %1077, <2 x float> %1076, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1233 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1210, <8 x half> %1196, <4 x float> %1232, i32 0, i32 0, i32 0)
  %1234 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %1197, <4 x float> %1233, i32 0, i32 0, i32 0)
  %1235 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1213, <8 x half> %1196, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %1236 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1214, <8 x half> %1197, <4 x float> %1235, i32 0, i32 0, i32 0)
  %1237 = shufflevector <2 x float> %1074, <2 x float> %1073, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1238 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %1196, <4 x float> %1237, i32 0, i32 0, i32 0)
  %1239 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %1197, <4 x float> %1238, i32 0, i32 0, i32 0)
  %1240 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %1196, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %1241 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %1197, <4 x float> %1240, i32 0, i32 0, i32 0)
  %1242 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1219, <8 x half> %1196, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %1243 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1220, <8 x half> %1197, <4 x float> %1242, i32 0, i32 0, i32 0)
  %1244 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1202, <8 x half> %1198, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %1245 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1203, <8 x half> %1199, <4 x float> %1244, i32 0, i32 0, i32 0)
  %1246 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1204, <8 x half> %1198, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %1247 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1205, <8 x half> %1199, <4 x float> %1246, i32 0, i32 0, i32 0)
  %1248 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1206, <8 x half> %1198, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %1249 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1207, <8 x half> %1199, <4 x float> %1248, i32 0, i32 0, i32 0)
  %1250 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %1198, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %1251 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1208, <8 x half> %1199, <4 x float> %1250, i32 0, i32 0, i32 0)
  %1252 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1209, <8 x half> %1198, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %1253 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %1199, <4 x float> %1252, i32 0, i32 0, i32 0)
  %1254 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1210, <8 x half> %1198, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %1255 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %1199, <4 x float> %1254, i32 0, i32 0, i32 0)
  %1256 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1211, <8 x half> %1198, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %1257 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1212, <8 x half> %1199, <4 x float> %1256, i32 0, i32 0, i32 0)
  %1258 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1219, <8 x half> %1198, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %1259 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %1199, <4 x float> %1258, i32 0, i32 0, i32 0)
  %1260 = shufflevector <2 x float> %1072, <2 x float> %1071, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1261 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1202, <8 x half> %1200, <4 x float> %1260, i32 0, i32 0, i32 0)
  %1262 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1203, <8 x half> %1201, <4 x float> %1261, i32 0, i32 0, i32 0)
  %1263 = shufflevector <2 x float> %1070, <2 x float> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1264 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1204, <8 x half> %1200, <4 x float> %1263, i32 0, i32 0, i32 0)
  %1265 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1205, <8 x half> %1201, <4 x float> %1264, i32 0, i32 0, i32 0)
  %1266 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %1200, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %1267 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1208, <8 x half> %1201, <4 x float> %1266, i32 0, i32 0, i32 0)
  %1268 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %1201, <4 x float> %64, i32 0, i32 0, i32 0)
  %1269 = shufflevector <2 x float> zeroinitializer, <2 x float> %1068, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1270 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1210, <8 x half> %1200, <4 x float> %1269, i32 0, i32 0, i32 0)
  %1271 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> %1201, <4 x float> %1270, i32 0, i32 0, i32 0)
  %1272 = shufflevector <2 x float> %1066, <2 x float> %1065, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1273 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1215, <8 x half> %1200, <4 x float> %1272, i32 0, i32 0, i32 0)
  %1274 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1216, <8 x half> %1201, <4 x float> %1273, i32 0, i32 0, i32 0)
  %1275 = shufflevector <2 x float> %1064, <2 x float> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1276 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1217, <8 x half> %1200, <4 x float> %1275, i32 0, i32 0, i32 0)
  %1277 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %1218, <8 x half> %1201, <4 x float> %1276, i32 0, i32 0, i32 0)
  %1278 = shufflevector <4 x float> %1277, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1279 = shufflevector <4 x float> %1274, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1280 = shufflevector <4 x float> %1271, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1281 = shufflevector <4 x float> %1268, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1282 = shufflevector <4 x float> %1267, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1283 = shufflevector <4 x float> %1265, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1284 = shufflevector <4 x float> %1262, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1285 = shufflevector <4 x float> %1259, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1286 = shufflevector <4 x float> %1257, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1287 = shufflevector <4 x float> %1255, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1288 = shufflevector <4 x float> %1253, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1289 = shufflevector <4 x float> %1251, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1290 = shufflevector <4 x float> %1249, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1291 = shufflevector <4 x float> %1247, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1292 = shufflevector <4 x float> %1245, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1293 = shufflevector <4 x float> %1243, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1294 = shufflevector <4 x float> %1241, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1295 = shufflevector <4 x float> %1239, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1296 = shufflevector <4 x float> %1236, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1297 = shufflevector <4 x float> %1234, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1298 = shufflevector <4 x float> %1231, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1299 = shufflevector <4 x float> %1229, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %1300 = shufflevector <4 x float> %1227, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1301 = shufflevector <4 x float> %1225, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %1302 = shufflevector <4 x float> %1222, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  br label %._crit_edge._crit_edge

._crit_edge._crit_edge:                           ; preds = %1194, %._crit_edge
  %1303 = phi <2 x float> [ %1278, %1194 ], [ zeroinitializer, %._crit_edge ]
  %1304 = phi <2 x float> [ %1279, %1194 ], [ %1066, %._crit_edge ]
  %1305 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1067, %._crit_edge ]
  %1306 = phi <2 x float> [ %1280, %1194 ], [ zeroinitializer, %._crit_edge ]
  %1307 = phi <2 x float> [ %1281, %1194 ], [ %1069, %._crit_edge ]
  %1308 = phi <2 x float> [ %1282, %1194 ], [ zeroinitializer, %._crit_edge ]
  %1309 = phi <2 x float> [ %1283, %1194 ], [ %1070, %._crit_edge ]
  %1310 = phi <2 x float> [ %1284, %1194 ], [ %1072, %._crit_edge ]
  %1311 = phi <2 x float> [ %1285, %1194 ], [ zeroinitializer, %._crit_edge ]
  %1312 = phi <2 x float> [ %1286, %1194 ], [ zeroinitializer, %._crit_edge ]
  %1313 = phi <2 x float> [ %1287, %1194 ], [ zeroinitializer, %._crit_edge ]
  %1314 = phi <2 x float> [ %1288, %1194 ], [ zeroinitializer, %._crit_edge ]
  %1315 = phi <2 x float> [ %1289, %1194 ], [ zeroinitializer, %._crit_edge ]
  %1316 = phi <2 x float> [ %1290, %1194 ], [ zeroinitializer, %._crit_edge ]
  %1317 = phi <2 x float> [ %1291, %1194 ], [ zeroinitializer, %._crit_edge ]
  %1318 = phi <2 x float> [ %1292, %1194 ], [ zeroinitializer, %._crit_edge ]
  %1319 = phi <2 x float> [ %1293, %1194 ], [ zeroinitializer, %._crit_edge ]
  %1320 = phi <2 x float> [ %1294, %1194 ], [ zeroinitializer, %._crit_edge ]
  %1321 = phi <2 x float> [ %1295, %1194 ], [ %1074, %._crit_edge ]
  %1322 = phi <2 x float> [ %1296, %1194 ], [ %1075, %._crit_edge ]
  %1323 = phi <2 x float> [ %1297, %1194 ], [ %1077, %._crit_edge ]
  %1324 = phi <2 x float> [ %1298, %1194 ], [ zeroinitializer, %._crit_edge ]
  %1325 = phi <2 x float> [ %1299, %1194 ], [ zeroinitializer, %._crit_edge ]
  %1326 = phi <2 x float> [ %1300, %1194 ], [ zeroinitializer, %._crit_edge ]
  %1327 = phi <2 x float> [ %1301, %1194 ], [ %1078, %._crit_edge ]
  %1328 = phi <2 x float> [ %1302, %1194 ], [ %1080, %._crit_edge ]
  %1329 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1081, %._crit_edge ]
  %1330 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1082, %._crit_edge ]
  %1331 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1083, %._crit_edge ]
  %1332 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1084, %._crit_edge ]
  %1333 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1085, %._crit_edge ]
  %1334 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1086, %._crit_edge ]
  %1335 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1087, %._crit_edge ]
  %1336 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1088, %._crit_edge ]
  %1337 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1089, %._crit_edge ]
  %1338 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1090, %._crit_edge ]
  %1339 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1091, %._crit_edge ]
  %1340 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1092, %._crit_edge ]
  %1341 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1093, %._crit_edge ]
  %1342 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1094, %._crit_edge ]
  %1343 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1095, %._crit_edge ]
  %1344 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1096, %._crit_edge ]
  %1345 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1097, %._crit_edge ]
  %1346 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1098, %._crit_edge ]
  %1347 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1099, %._crit_edge ]
  %1348 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1100, %._crit_edge ]
  %1349 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1101, %._crit_edge ]
  %1350 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1102, %._crit_edge ]
  %1351 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1103, %._crit_edge ]
  %1352 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1104, %._crit_edge ]
  %1353 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1105, %._crit_edge ]
  %1354 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1106, %._crit_edge ]
  %1355 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1107, %._crit_edge ]
  %1356 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1108, %._crit_edge ]
  %1357 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1109, %._crit_edge ]
  %1358 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1110, %._crit_edge ]
  %1359 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1111, %._crit_edge ]
  %1360 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1112, %._crit_edge ]
  %1361 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1113, %._crit_edge ]
  %1362 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1114, %._crit_edge ]
  %1363 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1115, %._crit_edge ]
  %1364 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1116, %._crit_edge ]
  %1365 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1117, %._crit_edge ]
  %1366 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1118, %._crit_edge ]
  %1367 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1119, %._crit_edge ]
  %1368 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1120, %._crit_edge ]
  %1369 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1121, %._crit_edge ]
  %1370 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1122, %._crit_edge ]
  %1371 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1123, %._crit_edge ]
  %1372 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1124, %._crit_edge ]
  %1373 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1125, %._crit_edge ]
  %1374 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1126, %._crit_edge ]
  %1375 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1127, %._crit_edge ]
  %1376 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1128, %._crit_edge ]
  %1377 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1129, %._crit_edge ]
  %1378 = phi <2 x float> [ zeroinitializer, %1194 ], [ %66, %._crit_edge ]
  %1379 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1130, %._crit_edge ]
  %1380 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1131, %._crit_edge ]
  %1381 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1132, %._crit_edge ]
  %1382 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1133, %._crit_edge ]
  %1383 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1134, %._crit_edge ]
  %1384 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1135, %._crit_edge ]
  %1385 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1136, %._crit_edge ]
  %1386 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1137, %._crit_edge ]
  %1387 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1138, %._crit_edge ]
  %1388 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1139, %._crit_edge ]
  %1389 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1140, %._crit_edge ]
  %1390 = phi <2 x float> [ zeroinitializer, %1194 ], [ %67, %._crit_edge ]
  %1391 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1141, %._crit_edge ]
  %1392 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1142, %._crit_edge ]
  %1393 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1143, %._crit_edge ]
  %1394 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1144, %._crit_edge ]
  %1395 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1145, %._crit_edge ]
  %1396 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1146, %._crit_edge ]
  %1397 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1147, %._crit_edge ]
  %1398 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1148, %._crit_edge ]
  %1399 = phi <2 x float> [ zeroinitializer, %1194 ], [ %1149, %._crit_edge ]
  %1400 = icmp slt i32 %77, 1
  br i1 %1400, label %1401, label %._crit_edge1985

1401:                                             ; preds = %._crit_edge._crit_edge
  %1402 = sext i32 %68 to i64
  %1403 = getelementptr float, ptr addrspace(1) %1, i64 %1402
  br label %._crit_edge1985

._crit_edge1985:                                  ; preds = %1401, %._crit_edge._crit_edge
  %1404 = load <4 x float>, ptr addrspace(3) inttoptr (i32 1568 to ptr addrspace(3)), align 16
  %1405 = load <4 x float>, ptr addrspace(3) null, align 16
  tail call void @llvm.amdgcn.s.waitcnt(i32 0)
  %1406 = load <4 x float>, ptr addrspace(3) null, align 16
  %1407 = load <4 x float>, ptr addrspace(3) inttoptr (i32 256 to ptr addrspace(3)), align 16
  %1408 = shufflevector <4 x float> %1406, <4 x float> zeroinitializer, <2 x i32> zeroinitializer
  %1409 = shufflevector <4 x float> %1404, <4 x float> %1405, <2 x i32> <i32 1, i32 5>
  %1410 = fadd <2 x float> %1399, %1409
  %1411 = fmul <2 x float> %1410, %1408
  %1412 = fadd <2 x float> %1398, zeroinitializer
  %1413 = fmul <2 x float> %1412, %1408
  %1414 = fadd <2 x float> %1397, zeroinitializer
  %1415 = fmul <2 x float> %1414, %1408
  %1416 = fadd <2 x float> %1396, zeroinitializer
  %1417 = fmul <2 x float> %1416, %1408
  %1418 = fadd <2 x float> %1395, zeroinitializer
  %1419 = fmul <2 x float> %1418, %1408
  %1420 = fadd <2 x float> %1394, zeroinitializer
  %1421 = fmul <2 x float> %1420, %1408
  %1422 = fadd <2 x float> %1393, zeroinitializer
  %1423 = fmul <2 x float> %1422, %1408
  %1424 = fadd <2 x float> %1392, zeroinitializer
  %1425 = fmul <2 x float> %1424, %1408
  %1426 = fadd <2 x float> %1391, zeroinitializer
  %1427 = fmul <2 x float> %1426, %1408
  %1428 = fadd <2 x float> %1390, zeroinitializer
  %1429 = fmul <2 x float> %1428, %1408
  %1430 = fadd <2 x float> %1389, zeroinitializer
  %1431 = fmul <2 x float> %1430, %1408
  %1432 = fadd <2 x float> %1388, zeroinitializer
  %1433 = fmul <2 x float> %1432, %1408
  %1434 = fadd <2 x float> %1387, zeroinitializer
  %1435 = fmul <2 x float> %1434, zeroinitializer
  %1436 = fadd <2 x float> %1386, zeroinitializer
  %1437 = fmul <2 x float> %1436, %69
  %1438 = fadd <2 x float> %1385, zeroinitializer
  %1439 = fmul <2 x float> %1438, %1408
  %1440 = shufflevector <4 x float> splat (float 1.000000e+00), <4 x float> %1405, <2 x i32> <i32 3, i32 7>
  %1441 = fadd <2 x float> %1384, zeroinitializer
  %1442 = fmul <2 x float> %1441, zeroinitializer
  %1443 = fadd <2 x float> %1383, zeroinitializer
  %1444 = fmul <2 x float> %1443, zeroinitializer
  %1445 = fadd <2 x float> %1382, zeroinitializer
  %1446 = fmul <2 x float> %1445, zeroinitializer
  %1447 = fadd <2 x float> %1381, zeroinitializer
  %1448 = fmul <2 x float> %1447, zeroinitializer
  %1449 = fadd <2 x float> %1380, zeroinitializer
  %1450 = fmul <2 x float> %1449, zeroinitializer
  %1451 = fadd <2 x float> %1379, zeroinitializer
  %1452 = fmul <2 x float> %1451, zeroinitializer
  %1453 = fadd <2 x float> %1378, zeroinitializer
  %1454 = fmul <2 x float> %1453, zeroinitializer
  %1455 = fadd <2 x float> %1377, zeroinitializer
  %1456 = fmul <2 x float> %1455, zeroinitializer
  %1457 = fadd <2 x float> %1376, zeroinitializer
  %1458 = fmul <2 x float> %1457, zeroinitializer
  %1459 = fadd <2 x float> %1375, %1440
  %1460 = fmul <2 x float> %1459, zeroinitializer
  %1461 = shufflevector <4 x float> %1406, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 2>
  %1462 = fadd <2 x float> %1374, zeroinitializer
  %1463 = fmul <2 x float> %1462, %1461
  %1464 = fmul <2 x float> zeroinitializer, %1461
  %1465 = fadd <2 x float> %1373, zeroinitializer
  %1466 = fmul <2 x float> %1465, %1461
  %1467 = fadd <2 x float> %1372, zeroinitializer
  %1468 = fmul <2 x float> %1467, %1461
  %1469 = fadd <2 x float> %1371, zeroinitializer
  %1470 = fmul <2 x float> %1469, %1461
  %1471 = fadd <2 x float> %1370, zeroinitializer
  %1472 = fmul <2 x float> %1471, %1461
  %1473 = fadd <2 x float> %1369, zeroinitializer
  %1474 = fmul <2 x float> %1473, %1461
  %1475 = fadd <2 x float> %1368, zeroinitializer
  %1476 = fmul <2 x float> %1475, %1461
  %1477 = fadd <2 x float> %1367, zeroinitializer
  %1478 = fmul <2 x float> %1477, %1461
  %1479 = fadd <2 x float> %1366, zeroinitializer
  %1480 = fmul <2 x float> %1479, %1461
  %1481 = fadd <2 x float> %1365, zeroinitializer
  %1482 = fmul <2 x float> %1481, %1461
  %1483 = fadd <2 x float> %1364, zeroinitializer
  %1484 = fmul <2 x float> %1483, zeroinitializer
  %1485 = fadd <2 x float> %1363, zeroinitializer
  %1486 = fmul <2 x float> %1485, %1461
  %1487 = fadd <2 x float> %1362, zeroinitializer
  %1488 = fmul <2 x float> %1487, %70
  %1489 = fadd <2 x float> %1361, zeroinitializer
  %1490 = fmul <2 x float> %1489, zeroinitializer
  %1491 = fadd <2 x float> %1360, zeroinitializer
  %1492 = fmul <2 x float> %1491, %1461
  %1493 = fadd <2 x float> %1359, zeroinitializer
  %1494 = fmul <2 x float> %1493, zeroinitializer
  %1495 = fadd <2 x float> %1358, zeroinitializer
  %1496 = fmul <2 x float> %1495, zeroinitializer
  %1497 = shufflevector <4 x float> %1406, <4 x float> zeroinitializer, <2 x i32> <i32 3, i32 3>
  %1498 = fmul <2 x float> %71, %1497
  %1499 = fadd <2 x float> %1357, zeroinitializer
  %1500 = fmul <2 x float> %1499, zeroinitializer
  %1501 = fadd <2 x float> %1356, zeroinitializer
  %1502 = fmul <2 x float> %1501, zeroinitializer
  %1503 = fadd <2 x float> %1355, zeroinitializer
  %1504 = fmul <2 x float> %1503, zeroinitializer
  %1505 = fadd <2 x float> %1354, zeroinitializer
  %1506 = fmul <2 x float> %1505, zeroinitializer
  %1507 = fadd <2 x float> %1353, zeroinitializer
  %1508 = fmul <2 x float> %1507, zeroinitializer
  %1509 = fmul <2 x float> zeroinitializer, %72
  %1510 = fadd <2 x float> %1352, zeroinitializer
  %1511 = fmul <2 x float> %1510, zeroinitializer
  %1512 = fadd <2 x float> %1351, zeroinitializer
  %1513 = fmul <2 x float> %1512, zeroinitializer
  %1514 = fadd <2 x float> %1350, zeroinitializer
  %1515 = fmul <2 x float> %1514, zeroinitializer
  %1516 = fadd <2 x float> %1349, zeroinitializer
  %1517 = fmul <2 x float> %1516, zeroinitializer
  %1518 = fadd <2 x float> %1348, zeroinitializer
  %1519 = fmul <2 x float> %1518, zeroinitializer
  %1520 = fadd <2 x float> %1347, zeroinitializer
  %1521 = fmul <2 x float> %1520, zeroinitializer
  %1522 = fadd <2 x float> %1346, zeroinitializer
  %1523 = fmul <2 x float> %1522, zeroinitializer
  %1524 = fadd <2 x float> %1345, zeroinitializer
  %1525 = fmul <2 x float> %1524, zeroinitializer
  %1526 = fadd <2 x float> %1344, zeroinitializer
  %1527 = fmul <2 x float> %1526, zeroinitializer
  %1528 = shufflevector <4 x float> %1407, <4 x float> zeroinitializer, <2 x i32> zeroinitializer
  %1529 = fadd <2 x float> %1343, zeroinitializer
  %1530 = fmul <2 x float> %1529, %1528
  %1531 = fadd <2 x float> %1342, zeroinitializer
  %1532 = fmul <2 x float> %1531, zeroinitializer
  %1533 = fadd <2 x float> %1341, zeroinitializer
  %1534 = fmul <2 x float> %1533, zeroinitializer
  %1535 = fadd <2 x float> %1340, zeroinitializer
  %1536 = fmul <2 x float> %1535, zeroinitializer
  %1537 = fadd <2 x float> %1339, zeroinitializer
  %1538 = fmul <2 x float> %1537, zeroinitializer
  %1539 = fadd <2 x float> %1338, zeroinitializer
  %1540 = fmul <2 x float> %1539, zeroinitializer
  %1541 = fadd <2 x float> %1337, zeroinitializer
  %1542 = fmul <2 x float> %1541, zeroinitializer
  %1543 = fadd <2 x float> %1336, zeroinitializer
  %1544 = fmul <2 x float> %1543, zeroinitializer
  %1545 = fadd <2 x float> %1335, zeroinitializer
  %1546 = fmul <2 x float> %1545, zeroinitializer
  %1547 = fadd <2 x float> %1334, zeroinitializer
  %1548 = fmul <2 x float> %1547, zeroinitializer
  %1549 = fadd <2 x float> %1333, zeroinitializer
  %1550 = fmul <2 x float> %1549, zeroinitializer
  %1551 = fadd <2 x float> %1332, zeroinitializer
  %1552 = fmul <2 x float> %1551, zeroinitializer
  %1553 = fadd <2 x float> %1331, zeroinitializer
  %1554 = fmul <2 x float> %1553, zeroinitializer
  %1555 = fadd <2 x float> %1330, zeroinitializer
  %1556 = fmul <2 x float> %1555, zeroinitializer
  %1557 = fadd <2 x float> %1329, zeroinitializer
  %1558 = fmul <2 x float> %1557, zeroinitializer
  %1559 = fadd <2 x float> %1328, %73
  %1560 = fmul <2 x float> %1559, zeroinitializer
  %1561 = fadd <2 x float> %1327, zeroinitializer
  %1562 = fmul <2 x float> %1561, zeroinitializer
  %1563 = fadd <2 x float> %1326, zeroinitializer
  %1564 = fmul <2 x float> %1563, zeroinitializer
  %1565 = fadd <2 x float> %1325, zeroinitializer
  %1566 = fmul <2 x float> %1565, zeroinitializer
  %1567 = fadd <2 x float> %1324, zeroinitializer
  %1568 = fmul <2 x float> %1567, zeroinitializer
  %1569 = fadd <2 x float> %1323, zeroinitializer
  %1570 = fmul <2 x float> %1569, zeroinitializer
  %1571 = fadd <2 x float> %1322, zeroinitializer
  %1572 = fmul <2 x float> %1571, zeroinitializer
  %1573 = fadd <2 x float> %74, %1409
  %1574 = fmul <2 x float> %1573, zeroinitializer
  %1575 = fadd <2 x float> %1321, zeroinitializer
  %1576 = fmul <2 x float> %1575, zeroinitializer
  %1577 = fadd <2 x float> %1320, zeroinitializer
  %1578 = fmul <2 x float> %1577, zeroinitializer
  %1579 = fadd <2 x float> %1319, zeroinitializer
  %1580 = fmul <2 x float> %1579, zeroinitializer
  %1581 = fadd <2 x float> %1318, zeroinitializer
  %1582 = shufflevector <4 x float> %1407, <4 x float> zeroinitializer, <2 x i32> <i32 2, i32 2>
  %1583 = fmul <2 x float> %1581, zeroinitializer
  %1584 = fadd <2 x float> %1317, zeroinitializer
  %1585 = fmul <2 x float> %1584, %1582
  %1586 = fadd <2 x float> %1316, zeroinitializer
  %1587 = fmul <2 x float> %1586, zeroinitializer
  %1588 = fadd <2 x float> %1315, zeroinitializer
  %1589 = fmul <2 x float> %1588, zeroinitializer
  %1590 = fadd <2 x float> %1314, zeroinitializer
  %1591 = fmul <2 x float> %1590, zeroinitializer
  %1592 = fadd <2 x float> %1313, zeroinitializer
  %1593 = fmul <2 x float> %1592, zeroinitializer
  %1594 = fadd <2 x float> %1312, zeroinitializer
  %1595 = fmul <2 x float> %1594, zeroinitializer
  %1596 = fadd <2 x float> %1311, zeroinitializer
  %1597 = fmul <2 x float> %1596, zeroinitializer
  %1598 = fadd <2 x float> %1310, zeroinitializer
  %1599 = fmul <2 x float> %1598, zeroinitializer
  %1600 = fadd <2 x float> %1309, zeroinitializer
  %1601 = fmul <2 x float> %1600, zeroinitializer
  %1602 = fadd <2 x float> %1308, zeroinitializer
  %1603 = fmul <2 x float> %1602, zeroinitializer
  %1604 = fadd <2 x float> %1307, zeroinitializer
  %1605 = fmul <2 x float> %1604, zeroinitializer
  %1606 = fadd <2 x float> %1306, zeroinitializer
  %1607 = fmul <2 x float> %1606, zeroinitializer
  %1608 = fadd <2 x float> %1305, zeroinitializer
  %1609 = fmul <2 x float> %1608, zeroinitializer
  %1610 = fadd <2 x float> %1304, zeroinitializer
  %1611 = fmul <2 x float> %1610, zeroinitializer
  %1612 = fadd <2 x float> %1303, zeroinitializer
  %1613 = fmul <2 x float> %1612, zeroinitializer
  %1614 = fptrunc <2 x float> %1411 to <2 x half>
  %1615 = fptrunc <2 x float> %1413 to <2 x half>
  %1616 = fptrunc <2 x float> %1415 to <2 x half>
  %1617 = fptrunc <2 x float> %1417 to <2 x half>
  %1618 = fptrunc <2 x float> %1419 to <2 x half>
  %1619 = fptrunc <2 x float> %1421 to <2 x half>
  %1620 = fptrunc <2 x float> %1423 to <2 x half>
  %1621 = fptrunc <2 x float> %1425 to <2 x half>
  %1622 = fptrunc <2 x float> %1427 to <2 x half>
  %1623 = fptrunc <2 x float> %1429 to <2 x half>
  %1624 = fptrunc <2 x float> %1431 to <2 x half>
  %1625 = fptrunc <2 x float> %1433 to <2 x half>
  %1626 = fptrunc <2 x float> %1435 to <2 x half>
  %1627 = fptrunc <2 x float> %1437 to <2 x half>
  %1628 = fptrunc <2 x float> %1439 to <2 x half>
  %1629 = fptrunc <2 x float> %1442 to <2 x half>
  %1630 = fptrunc <2 x float> %1444 to <2 x half>
  %1631 = fptrunc <2 x float> %1446 to <2 x half>
  %1632 = fptrunc <2 x float> %1448 to <2 x half>
  %1633 = fptrunc <2 x float> %1450 to <2 x half>
  %1634 = fptrunc <2 x float> %1452 to <2 x half>
  %1635 = fptrunc <2 x float> %1454 to <2 x half>
  %1636 = fptrunc <2 x float> %1456 to <2 x half>
  %1637 = fptrunc <2 x float> %1458 to <2 x half>
  %1638 = fptrunc <2 x float> %1460 to <2 x half>
  %1639 = fptrunc <2 x float> %1463 to <2 x half>
  %1640 = fptrunc <2 x float> %1464 to <2 x half>
  %1641 = fptrunc <2 x float> %1466 to <2 x half>
  %1642 = fptrunc <2 x float> %1468 to <2 x half>
  %1643 = fptrunc <2 x float> %1470 to <2 x half>
  %1644 = fptrunc <2 x float> %1472 to <2 x half>
  %1645 = fptrunc <2 x float> %1474 to <2 x half>
  %1646 = fptrunc <2 x float> %1476 to <2 x half>
  %1647 = fptrunc <2 x float> %1478 to <2 x half>
  %1648 = fptrunc <2 x float> %1480 to <2 x half>
  %1649 = fptrunc <2 x float> %1482 to <2 x half>
  %1650 = fptrunc <2 x float> %1484 to <2 x half>
  %1651 = fptrunc <2 x float> %1486 to <2 x half>
  %1652 = fptrunc <2 x float> %1488 to <2 x half>
  %1653 = fptrunc <2 x float> %1490 to <2 x half>
  %1654 = fptrunc <2 x float> %1492 to <2 x half>
  %1655 = fptrunc <2 x float> %1494 to <2 x half>
  %1656 = fptrunc <2 x float> %1496 to <2 x half>
  %1657 = fptrunc <2 x float> %1498 to <2 x half>
  %1658 = fptrunc <2 x float> %1500 to <2 x half>
  %1659 = fptrunc <2 x float> %1502 to <2 x half>
  %1660 = fptrunc <2 x float> %1504 to <2 x half>
  %1661 = fptrunc <2 x float> %1506 to <2 x half>
  %1662 = fptrunc <2 x float> %1508 to <2 x half>
  %1663 = fptrunc <2 x float> %1509 to <2 x half>
  %1664 = fptrunc <2 x float> %1511 to <2 x half>
  %1665 = fptrunc <2 x float> %1513 to <2 x half>
  %1666 = fptrunc <2 x float> %1515 to <2 x half>
  %1667 = fptrunc <2 x float> %1517 to <2 x half>
  %1668 = fptrunc <2 x float> %1519 to <2 x half>
  %1669 = fptrunc <2 x float> %1521 to <2 x half>
  %1670 = fptrunc <2 x float> %1523 to <2 x half>
  %1671 = fptrunc <2 x float> %1525 to <2 x half>
  %1672 = fptrunc <2 x float> %1527 to <2 x half>
  %1673 = fptrunc <2 x float> %1530 to <2 x half>
  %1674 = fptrunc <2 x float> %1532 to <2 x half>
  %1675 = fptrunc <2 x float> %1534 to <2 x half>
  %1676 = fptrunc <2 x float> %1536 to <2 x half>
  %1677 = fptrunc <2 x float> %1538 to <2 x half>
  %1678 = fptrunc <2 x float> %1540 to <2 x half>
  %1679 = fptrunc <2 x float> %1542 to <2 x half>
  %1680 = fptrunc <2 x float> %1544 to <2 x half>
  %1681 = fptrunc <2 x float> %1546 to <2 x half>
  %1682 = fptrunc <2 x float> %1548 to <2 x half>
  %1683 = fptrunc <2 x float> %1550 to <2 x half>
  %1684 = fptrunc <2 x float> %1552 to <2 x half>
  %1685 = fptrunc <2 x float> %1554 to <2 x half>
  %1686 = fptrunc <2 x float> %1556 to <2 x half>
  %1687 = fptrunc <2 x float> %1558 to <2 x half>
  %1688 = fptrunc <2 x float> %1560 to <2 x half>
  %1689 = fptrunc <2 x float> %1562 to <2 x half>
  %1690 = fptrunc <2 x float> %1564 to <2 x half>
  %1691 = fptrunc <2 x float> %1566 to <2 x half>
  %1692 = fptrunc <2 x float> %1568 to <2 x half>
  %1693 = fptrunc <2 x float> %1570 to <2 x half>
  %1694 = fptrunc <2 x float> %1572 to <2 x half>
  %1695 = fptrunc <2 x float> %1574 to <2 x half>
  %1696 = fptrunc <2 x float> %1576 to <2 x half>
  %1697 = fptrunc <2 x float> %1578 to <2 x half>
  %1698 = fptrunc <2 x float> %1580 to <2 x half>
  %1699 = fptrunc <2 x float> %1583 to <2 x half>
  %1700 = fptrunc <2 x float> %1585 to <2 x half>
  %1701 = fptrunc <2 x float> %1587 to <2 x half>
  %1702 = fptrunc <2 x float> %1589 to <2 x half>
  %1703 = fptrunc <2 x float> %1591 to <2 x half>
  %1704 = fptrunc <2 x float> %1593 to <2 x half>
  %1705 = fptrunc <2 x float> %1595 to <2 x half>
  %1706 = fptrunc <2 x float> %1597 to <2 x half>
  %1707 = fptrunc <2 x float> %1599 to <2 x half>
  %1708 = fptrunc <2 x float> %1601 to <2 x half>
  %1709 = fptrunc <2 x float> %1603 to <2 x half>
  %1710 = fptrunc <2 x float> %1605 to <2 x half>
  %1711 = fptrunc <2 x float> %1607 to <2 x half>
  %1712 = fptrunc <2 x float> %1609 to <2 x half>
  %1713 = fptrunc <2 x float> %1611 to <2 x half>
  %1714 = fptrunc <2 x float> %1613 to <2 x half>
  %.bc698 = bitcast <2 x half> %1614 to <2 x i16>
  %.extract701 = extractelement <2 x i16> %.bc698, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract701, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc702 = bitcast <2 x half> %1615 to <2 x i16>
  %.extract703 = extractelement <2 x i16> %.bc702, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract703, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc706 = bitcast <2 x half> %1616 to <2 x i16>
  %.extract709 = extractelement <2 x i16> %.bc706, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract709, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc710 = bitcast <2 x half> %1617 to <2 x i16>
  %.extract711 = extractelement <2 x i16> %.bc710, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract711, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc714 = bitcast <2 x half> %1618 to <2 x i16>
  %.extract715 = extractelement <2 x i16> %.bc714, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract715, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc718 = bitcast <2 x half> %1619 to <2 x i16>
  %.extract719 = extractelement <2 x i16> %.bc718, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract719, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc722 = bitcast <2 x half> %1620 to <2 x i16>
  %.extract723 = extractelement <2 x i16> %.bc722, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract723, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc726 = bitcast <2 x half> %1621 to <2 x i16>
  %.extract727 = extractelement <2 x i16> %.bc726, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract727, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc730 = bitcast <2 x half> %1622 to <2 x i16>
  %.extract731 = extractelement <2 x i16> %.bc730, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract731, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc734 = bitcast <2 x half> %1623 to <2 x i16>
  %.extract737 = extractelement <2 x i16> %.bc734, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract737, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc738 = bitcast <2 x half> %1624 to <2 x i16>
  %.extract741 = extractelement <2 x i16> %.bc738, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract741, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc742 = bitcast <2 x half> %1625 to <2 x i16>
  %.extract745 = extractelement <2 x i16> %.bc742, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract745, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc746 = bitcast <2 x half> %1626 to <2 x i16>
  %.extract747 = extractelement <2 x i16> %.bc746, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract747, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc754 = bitcast <2 x half> %1627 to <2 x i16>
  %.extract757 = extractelement <2 x i16> %.bc754, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract757, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc758 = bitcast <2 x half> %1628 to <2 x i16>
  %.extract759 = extractelement <2 x i16> %.bc758, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract759, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc810 = bitcast <2 x half> %1629 to <2 x i16>
  %.extract813 = extractelement <2 x i16> %.bc810, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract813, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc822 = bitcast <2 x half> %1630 to <2 x i16>
  %.extract825 = extractelement <2 x i16> %.bc822, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract825, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc826 = bitcast <2 x half> %1631 to <2 x i16>
  %.extract827 = extractelement <2 x i16> %.bc826, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract827, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc830 = bitcast <2 x half> %1632 to <2 x i16>
  %.extract831 = extractelement <2 x i16> %.bc830, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract831, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc834 = bitcast <2 x half> %1633 to <2 x i16>
  %.extract835 = extractelement <2 x i16> %.bc834, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract835, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc850 = bitcast <2 x half> %1634 to <2 x i16>
  %.extract853 = extractelement <2 x i16> %.bc850, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract853, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc854 = bitcast <2 x half> %1635 to <2 x i16>
  %.extract857 = extractelement <2 x i16> %.bc854, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract857, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc858 = bitcast <2 x half> %1636 to <2 x i16>
  %.extract859 = extractelement <2 x i16> %.bc858, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract859, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc882 = bitcast <2 x half> %1637 to <2 x i16>
  %.extract883 = extractelement <2 x i16> %.bc882, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract883, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc890 = bitcast <2 x half> %1638 to <2 x i16>
  %.extract891 = extractelement <2 x i16> %.bc890, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract891, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc910 = bitcast <2 x half> %1639 to <2 x i16>
  %.extract911 = extractelement <2 x i16> %.bc910, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract911, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc926 = bitcast <2 x half> %1640 to <2 x i16>
  %.extract927 = extractelement <2 x i16> %.bc926, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract927, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc930 = bitcast <2 x half> %1641 to <2 x i16>
  %.extract931 = extractelement <2 x i16> %.bc930, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract931, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc934 = bitcast <2 x half> %1642 to <2 x i16>
  %.extract935 = extractelement <2 x i16> %.bc934, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract935, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc938 = bitcast <2 x half> %1643 to <2 x i16>
  %.extract941 = extractelement <2 x i16> %.bc938, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract941, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc946 = bitcast <2 x half> %1644 to <2 x i16>
  %.extract949 = extractelement <2 x i16> %.bc946, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract949, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc954 = bitcast <2 x half> %1645 to <2 x i16>
  %.extract955 = extractelement <2 x i16> %.bc954, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract955, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc958 = bitcast <2 x half> %1646 to <2 x i16>
  %.extract959 = extractelement <2 x i16> %.bc958, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract959, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 0, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc966 = bitcast <2 x half> %1647 to <2 x i16>
  %.extract969 = extractelement <2 x i16> %.bc966, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract969, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc970 = bitcast <2 x half> %1648 to <2 x i16>
  %.extract971 = extractelement <2 x i16> %.bc970, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract971, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc974 = bitcast <2 x half> %1649 to <2 x i16>
  %.extract977 = extractelement <2 x i16> %.bc974, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract977, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc978 = bitcast <2 x half> %1650 to <2 x i16>
  %.extract979 = extractelement <2 x i16> %.bc978, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract979, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc982 = bitcast <2 x half> %1651 to <2 x i16>
  %.extract983 = extractelement <2 x i16> %.bc982, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract983, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc986 = bitcast <2 x half> %1652 to <2 x i16>
  %.extract987 = extractelement <2 x i16> %.bc986, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract987, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc990 = bitcast <2 x half> %1653 to <2 x i16>
  %.extract991 = extractelement <2 x i16> %.bc990, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract991, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1002 = bitcast <2 x half> %1654 to <2 x i16>
  %.extract1005 = extractelement <2 x i16> %.bc1002, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1005, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1010 = bitcast <2 x half> %1655 to <2 x i16>
  %.extract1013 = extractelement <2 x i16> %.bc1010, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1013, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1014 = bitcast <2 x half> %1656 to <2 x i16>
  %.extract1015 = extractelement <2 x i16> %.bc1014, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1015, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1022 = bitcast <2 x half> %1657 to <2 x i16>
  %.extract1023 = extractelement <2 x i16> %.bc1022, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1023, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1026 = bitcast <2 x half> %1658 to <2 x i16>
  %.extract1027 = extractelement <2 x i16> %.bc1026, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1027, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1030 = bitcast <2 x half> %1659 to <2 x i16>
  %.extract1031 = extractelement <2 x i16> %.bc1030, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1031, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1050 = bitcast <2 x half> %1660 to <2 x i16>
  %.extract1053 = extractelement <2 x i16> %.bc1050, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1053, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1054 = bitcast <2 x half> %1661 to <2 x i16>
  %.extract1057 = extractelement <2 x i16> %.bc1054, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1057, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1062 = bitcast <2 x half> %1662 to <2 x i16>
  %.extract1063 = extractelement <2 x i16> %.bc1062, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1063, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1066 = bitcast <2 x half> %1663 to <2 x i16>
  %.extract1067 = extractelement <2 x i16> %.bc1066, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1067, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1070 = bitcast <2 x half> %1664 to <2 x i16>
  %.extract1071 = extractelement <2 x i16> %.bc1070, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1071, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1074 = bitcast <2 x half> %1665 to <2 x i16>
  %.extract1077 = extractelement <2 x i16> %.bc1074, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1077, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1082 = bitcast <2 x half> %1666 to <2 x i16>
  %.extract1085 = extractelement <2 x i16> %.bc1082, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1085, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1086 = bitcast <2 x half> %1667 to <2 x i16>
  %.extract1089 = extractelement <2 x i16> %.bc1086, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1089, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1094 = bitcast <2 x half> %1668 to <2 x i16>
  %.extract1097 = extractelement <2 x i16> %.bc1094, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1097, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1102 = bitcast <2 x half> %1669 to <2 x i16>
  %.extract1103 = extractelement <2 x i16> %.bc1102, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1103, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1110 = bitcast <2 x half> %1670 to <2 x i16>
  %.extract1111 = extractelement <2 x i16> %.bc1110, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1111, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1134 = bitcast <2 x half> %1671 to <2 x i16>
  %.extract1135 = extractelement <2 x i16> %.bc1134, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1135, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1146 = bitcast <2 x half> %1672 to <2 x i16>
  %.extract1149 = extractelement <2 x i16> %.bc1146, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1149, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1154 = bitcast <2 x half> %1673 to <2 x i16>
  %.extract1157 = extractelement <2 x i16> %.bc1154, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1157, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1158 = bitcast <2 x half> %1674 to <2 x i16>
  %.extract1161 = extractelement <2 x i16> %.bc1158, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1161, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1170 = bitcast <2 x half> %1675 to <2 x i16>
  %.extract1171 = extractelement <2 x i16> %.bc1170, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1171, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1174 = bitcast <2 x half> %1676 to <2 x i16>
  %.extract1175 = extractelement <2 x i16> %.bc1174, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1175, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1182 = bitcast <2 x half> %1677 to <2 x i16>
  %.extract1183 = extractelement <2 x i16> %.bc1182, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1183, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1190 = bitcast <2 x half> %1678 to <2 x i16>
  %.extract1191 = extractelement <2 x i16> %.bc1190, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1191, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1194 = bitcast <2 x half> %1679 to <2 x i16>
  %.extract1195 = extractelement <2 x i16> %.bc1194, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1195, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1206 = bitcast <2 x half> %1680 to <2 x i16>
  %.extract1207 = extractelement <2 x i16> %.bc1206, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1207, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1218 = bitcast <2 x half> %1681 to <2 x i16>
  %.extract1219 = extractelement <2 x i16> %.bc1218, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1219, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1222 = bitcast <2 x half> %1682 to <2 x i16>
  %.extract1223 = extractelement <2 x i16> %.bc1222, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1223, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1230 = bitcast <2 x half> %1683 to <2 x i16>
  %.extract1231 = extractelement <2 x i16> %.bc1230, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1231, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1238 = bitcast <2 x half> %1684 to <2 x i16>
  %.extract1239 = extractelement <2 x i16> %.bc1238, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1239, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1246 = bitcast <2 x half> %1685 to <2 x i16>
  %.extract1247 = extractelement <2 x i16> %.bc1246, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1247, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1258 = bitcast <2 x half> %1686 to <2 x i16>
  %.extract1259 = extractelement <2 x i16> %.bc1258, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1259, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1262 = bitcast <2 x half> %1687 to <2 x i16>
  %.extract1263 = extractelement <2 x i16> %.bc1262, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1263, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1274 = bitcast <2 x half> %1688 to <2 x i16>
  %.extract1275 = extractelement <2 x i16> %.bc1274, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1275, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1282 = bitcast <2 x half> %1689 to <2 x i16>
  %.extract1285 = extractelement <2 x i16> %.bc1282, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1285, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1290 = bitcast <2 x half> %1690 to <2 x i16>
  %.extract1291 = extractelement <2 x i16> %.bc1290, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1291, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1294 = bitcast <2 x half> %1691 to <2 x i16>
  %.extract1295 = extractelement <2 x i16> %.bc1294, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1295, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1314 = bitcast <2 x half> %1692 to <2 x i16>
  %.extract1317 = extractelement <2 x i16> %.bc1314, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1317, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1318 = bitcast <2 x half> %1693 to <2 x i16>
  %.extract1319 = extractelement <2 x i16> %.bc1318, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1319, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1334 = bitcast <2 x half> %1694 to <2 x i16>
  %.extract1335 = extractelement <2 x i16> %.bc1334, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1335, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1338 = bitcast <2 x half> %1695 to <2 x i16>
  %.extract1339 = extractelement <2 x i16> %.bc1338, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1339, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1342 = bitcast <2 x half> %1696 to <2 x i16>
  %.extract1343 = extractelement <2 x i16> %.bc1342, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1343, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1350 = bitcast <2 x half> %1697 to <2 x i16>
  %.extract1351 = extractelement <2 x i16> %.bc1350, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1351, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1402 = bitcast <2 x half> %1698 to <2 x i16>
  %.extract1403 = extractelement <2 x i16> %.bc1402, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1403, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1406 = bitcast <2 x half> %1699 to <2 x i16>
  %.extract1407 = extractelement <2 x i16> %.bc1406, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1407, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1418 = bitcast <2 x half> %1700 to <2 x i16>
  %.extract1419 = extractelement <2 x i16> %.bc1418, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1419, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1426 = bitcast <2 x half> %1701 to <2 x i16>
  %.extract1427 = extractelement <2 x i16> %.bc1426, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1427, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1434 = bitcast <2 x half> %1702 to <2 x i16>
  %.extract1435 = extractelement <2 x i16> %.bc1434, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1435, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1438 = bitcast <2 x half> %1703 to <2 x i16>
  %.extract1441 = extractelement <2 x i16> %.bc1438, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1441, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1446 = bitcast <2 x half> %1704 to <2 x i16>
  %.extract1447 = extractelement <2 x i16> %.bc1446, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1447, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1454 = bitcast <2 x half> %1705 to <2 x i16>
  %.extract1455 = extractelement <2 x i16> %.bc1454, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1455, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1530 = bitcast <2 x half> %1706 to <2 x i16>
  %.extract1531 = extractelement <2 x i16> %.bc1530, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1531, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1534 = bitcast <2 x half> %1707 to <2 x i16>
  %.extract1535 = extractelement <2 x i16> %.bc1534, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1535, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1542 = bitcast <2 x half> %1708 to <2 x i16>
  %.extract1545 = extractelement <2 x i16> %.bc1542, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1545, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1562 = bitcast <2 x half> %1709 to <2 x i16>
  %.extract1563 = extractelement <2 x i16> %.bc1562, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1563, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1566 = bitcast <2 x half> %1710 to <2 x i16>
  %.extract1567 = extractelement <2 x i16> %.bc1566, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1567, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1574 = bitcast <2 x half> %1711 to <2 x i16>
  %.extract1575 = extractelement <2 x i16> %.bc1574, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1575, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1614 = bitcast <2 x half> %1712 to <2 x i16>
  %.extract1615 = extractelement <2 x i16> %.bc1614, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1615, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1622 = bitcast <2 x half> %1713 to <2 x i16>
  %.extract1625 = extractelement <2 x i16> %.bc1622, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1625, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %.bc1642 = bitcast <2 x half> %1714 to <2 x i16>
  %.extract1643 = extractelement <2 x i16> %.bc1642, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract1643, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.waitcnt(i32 immarg) #3

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #4

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(argmem: read)
declare <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) captures(none)) #5

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half>, <8 x half>, <4 x float>, i32 immarg, i32 immarg, i32 immarg) #6

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #7

; uselistorder directives
uselistorder ptr @llvm.amdgcn.raw.ptr.buffer.load.i16, { 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
uselistorder ptr @llvm.amdgcn.s.waitcnt, { 2, 1, 0 }
uselistorder ptr @llvm.amdgcn.s.barrier, { 1, 0 }
uselistorder ptr @llvm.amdgcn.ds.read.tr16.b64.v4f16, { 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
uselistorder ptr @llvm.amdgcn.mfma.f32.16x16x32.f16, { 201, 200, 199, 198, 197, 196, 195, 194, 193, 192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
uselistorder ptr @llvm.amdgcn.raw.ptr.buffer.store.i16, { 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }

attributes #0 = { alwaysinline nofree norecurse nounwind "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,128" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="0, 0" "denormal-fp-math-f32"="ieee" "uniform-work-group-size"="false" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #3 = { nocallback nofree nounwind willreturn }
attributes #4 = { convergent nocallback nofree nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #6 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #7 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
