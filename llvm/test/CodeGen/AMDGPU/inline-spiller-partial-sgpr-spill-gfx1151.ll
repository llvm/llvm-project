; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1151 -mattr=-real-true16 -O3 -stop-after=greedy %s -o - | FileCheck %s
;
; Reduced from a Triton 3.8.0 gfx1151 attention kernel (issue #225054). A
; partially-defined SGPR tuple shares a spill slot with its siblings; the
; spiller must store only the defined words (per-word SI_SPILL_S32_SAVE_PARTIAL
; with a 128-bit load of the same slot) instead of a full-width store that
; clobbers a sibling. Regression guard only: removing the fix drops the partial
; op and fails this test. Full bug->clean proof is in the #225054 reproducer.
;
; CHECK: SI_SPILL_S32_SAVE_PARTIAL {{%[0-9]+}}.sub{{[0-9]+}}, [[SLOT:%stack\.[0-9]+]], {{[0-9]+}}{{.*}}(load (s128) from [[SLOT]]
; ModuleID = 'seed.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: norecurse nounwind
define amdgpu_kernel void @attention_kernel(ptr addrspace(1) inreg readonly captures(none) %0, ptr addrspace(1) inreg readonly captures(none) %1, ptr addrspace(1) inreg readonly captures(none) %2, <1 x i32> %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10, i32 %11, i32 %12, i32 %13, i32 %14, i32 %15, i32 %16, i32 %17, i32 %18, i32 %19, i32 %20, i32 %21, i32 %22, i32 %23, i32 %24, i32 %25, i32 %26, i32 %27, i32 %28, i32 %29, i32 %30, i32 %31, i32 %32, i32 %33, i32 %34, i32 %35, i32 %36, i32 %37, i32 %38, i32 %39, i32 %40, i32 %41, i32 %42, i32 %43, i32 %44, i32 %45, i32 %46, i32 %47, i32 %48, i32 %49, i32 %50, i32 %51, i32 %52, i32 %53, i32 %54, ptr addrspace(8) %55, i32 %56, i32 %57, i32 %58, i32 %59, i32 %60, i32 %61, i32 %62, i32 %63, i32 %64, i32 %65, i32 %66, i32 %67, i32 %68, i32 %69, i32 %70, i32 %71, i32 %72, i32 %73, i32 %74, i32 %75, i32 %76, i32 %77, i32 %78, i32 %79, i32 %80, i32 %81, i32 %82, i32 %83, i32 %84, i32 %85, i32 %86, i32 %87, i32 %88, i32 %89, i32 %90, i32 %91, i32 %92, i32 %93, i32 %94, i32 %95, i32 %96, i32 %97, i32 %98, i32 %99, i32 %100, i32 %101, i16 %102, <8 x i16> %103, i16 %104, <8 x i16> %105, i16 %106, <8 x i16> %107, i16 %108, <8 x i16> %109, i16 %110, <8 x i16> %111, i16 %112, <8 x i16> %113, ptr addrspace(3) %114, i16 %115, i16 %116, i16 %117, i16 %118, i16 %119, i16 %120, i16 %121, i16 %122, i16 %123, i32 %124, i32 %125, i32 %126, i32 %127, i32 %128, i32 %129, i32 %130, i32 %131, i32 %132, i32 %133, i32 %134, i32 %135) #0 {
.lr.ph:
  %136 = load <1 x i32>, ptr addrspace(1) null, align 4
  %137 = extractelement <1 x i32> %3, i64 0
  %138 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1) %0, i16 0, i64 2147483646, i32 822243328)
  %139 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1) %1, i16 0, i64 0, i32 0)
  br label %140

140:                                              ; preds = %140, %.lr.ph
  %141 = phi i32 [ 0, %.lr.ph ], [ 0, %140 ]
  %142 = phi i32 [ 0, %.lr.ph ], [ 0, %140 ]
  %143 = phi i32 [ 14, %.lr.ph ], [ %4, %140 ]
  %144 = phi i32 [ 1, %.lr.ph ], [ 1, %140 ]
  %145 = phi i32 [ 16, %.lr.ph ], [ %5, %140 ]
  %146 = phi i32 [ 17, %.lr.ph ], [ %6, %140 ]
  %147 = phi i32 [ 18, %.lr.ph ], [ 0, %140 ]
  %148 = phi i32 [ 19, %.lr.ph ], [ %7, %140 ]
  %149 = phi i32 [ 20, %.lr.ph ], [ %8, %140 ]
  %150 = phi i32 [ 21, %.lr.ph ], [ %9, %140 ]
  %151 = phi i32 [ 22, %.lr.ph ], [ %10, %140 ]
  %152 = phi i32 [ 23, %.lr.ph ], [ %11, %140 ]
  %153 = phi i32 [ 24, %.lr.ph ], [ %12, %140 ]
  %154 = phi i32 [ 25, %.lr.ph ], [ %13, %140 ]
  %155 = phi i32 [ 26, %.lr.ph ], [ %14, %140 ]
  %156 = phi i32 [ 27, %.lr.ph ], [ %15, %140 ]
  %157 = phi i32 [ 28, %.lr.ph ], [ %16, %140 ]
  %158 = phi i32 [ 29, %.lr.ph ], [ %17, %140 ]
  %159 = phi i32 [ 1, %.lr.ph ], [ %18, %140 ]
  %160 = phi i32 [ 31, %.lr.ph ], [ %19, %140 ]
  %161 = phi i32 [ 32, %.lr.ph ], [ %20, %140 ]
  %162 = phi i32 [ 33, %.lr.ph ], [ %21, %140 ]
  %163 = phi i32 [ 34, %.lr.ph ], [ %22, %140 ]
  %164 = phi i32 [ 35, %.lr.ph ], [ %23, %140 ]
  %165 = phi i32 [ 36, %.lr.ph ], [ %24, %140 ]
  %166 = phi i32 [ 37, %.lr.ph ], [ %25, %140 ]
  %167 = phi i32 [ 38, %.lr.ph ], [ %26, %140 ]
  %168 = phi i32 [ 39, %.lr.ph ], [ %27, %140 ]
  %169 = phi i32 [ 40, %.lr.ph ], [ %28, %140 ]
  %170 = phi i32 [ 41, %.lr.ph ], [ %29, %140 ]
  %171 = phi i32 [ 42, %.lr.ph ], [ %30, %140 ]
  %172 = phi i32 [ 43, %.lr.ph ], [ %31, %140 ]
  %173 = phi i32 [ 44, %.lr.ph ], [ %32, %140 ]
  %174 = phi i32 [ 45, %.lr.ph ], [ %33, %140 ]
  %175 = phi i32 [ 46, %.lr.ph ], [ %34, %140 ]
  %176 = phi i32 [ 47, %.lr.ph ], [ %35, %140 ]
  %177 = phi i32 [ 48, %.lr.ph ], [ %36, %140 ]
  %178 = phi i32 [ 49, %.lr.ph ], [ %37, %140 ]
  %179 = phi i32 [ 50, %.lr.ph ], [ %38, %140 ]
  %180 = phi i32 [ 51, %.lr.ph ], [ %39, %140 ]
  %181 = phi i32 [ 52, %.lr.ph ], [ %40, %140 ]
  %182 = phi i32 [ 53, %.lr.ph ], [ %41, %140 ]
  %183 = phi i32 [ 54, %.lr.ph ], [ %42, %140 ]
  %184 = phi i32 [ 1, %.lr.ph ], [ %43, %140 ]
  %185 = phi i32 [ 56, %.lr.ph ], [ %44, %140 ]
  %186 = phi i32 [ 57, %.lr.ph ], [ %45, %140 ]
  %187 = phi i32 [ 58, %.lr.ph ], [ %46, %140 ]
  %188 = phi i32 [ 59, %.lr.ph ], [ %47, %140 ]
  %189 = phi i32 [ 60, %.lr.ph ], [ 0, %140 ]
  %190 = phi i32 [ 1, %.lr.ph ], [ %48, %140 ]
  %191 = phi i32 [ 0, %.lr.ph ], [ %49, %140 ]
  %192 = phi i32 [ 1, %.lr.ph ], [ %50, %140 ]
  %193 = phi i32 [ 0, %.lr.ph ], [ %51, %140 ]
  %194 = icmp slt i32 %193, 0
  %195 = shl i32 %52, 1
  %196 = shl i32 %53, 0
  %197 = select i1 %194, i32 %54, i32 1
  %198 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %55, i32 %197, i32 0, i32 0)
  store <4 x i32> %198, ptr addrspace(3) null, align 16
  fence syncscope("workgroup") release
  %199 = srem i32 %56, 96
  %200 = srem i32 1, 1
  %201 = srem i32 %57, 96
  %202 = srem i32 %58, 96
  %203 = srem i32 %59, 96
  %204 = srem i32 %60, 96
  %205 = srem i32 %149, 96
  %206 = srem i32 %61, 96
  %207 = srem i32 %151, 96
  %208 = srem i32 %152, 96
  %209 = srem i32 %153, 96
  %210 = srem i32 %154, 96
  %211 = srem i32 %155, 96
  %212 = srem i32 %156, 96
  %213 = srem i32 %157, 96
  %214 = srem i32 %158, 96
  %215 = srem i32 %159, 96
  %216 = srem i32 %160, 96
  %217 = srem i32 %161, 96
  %218 = srem i32 %162, 96
  %219 = srem i32 %163, 96
  %220 = srem i32 %164, 96
  %221 = srem i32 %165, 96
  %222 = srem i32 %166, 96
  %223 = srem i32 %167, 96
  %224 = srem i32 %168, 96
  %225 = srem i32 %169, 96
  %226 = srem i32 %170, 96
  %227 = srem i32 %171, 96
  %228 = srem i32 %172, 96
  %229 = srem i32 %173, 96
  %230 = srem i32 %174, 96
  %231 = srem i32 %175, 96
  %232 = srem i32 %176, 96
  %233 = srem i32 %177, 96
  %234 = srem i32 %178, 96
  %235 = srem i32 %179, 96
  %236 = srem i32 %180, 96
  %237 = srem i32 %181, 96
  %238 = srem i32 %182, 96
  %239 = srem i32 %62, 96
  %240 = srem i32 %63, 96
  %241 = srem i32 %64, 96
  %242 = srem i32 %65, 96
  %243 = srem i32 %66, 96
  %244 = srem i32 %67, 96
  %245 = srem i32 %68, 96
  %246 = srem i32 %69, 96
  %247 = srem i32 %70, 96
  %248 = srem i32 %71, 96
  %249 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %250 = shl i32 0, 0
  %251 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %252 = shl i32 %72, 1
  %253 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %73, i32 0, i32 0)
  %254 = shl i32 %74, 1
  %255 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %75, i32 0, i32 0)
  %256 = shl i32 %76, 1
  %257 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %77, i32 0, i32 0)
  %258 = shl i32 %78, 0
  %259 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %79, i32 0, i32 0)
  %260 = shl i32 %80, 1
  %261 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %81, i32 0, i32 0)
  %262 = shl i32 %205, 1
  %263 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %262, i32 0, i32 0)
  %264 = shl i32 %82, 1
  %265 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %83, i32 0, i32 0)
  %266 = shl i32 %207, 1
  %267 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %266, i32 0, i32 0)
  %268 = shl i32 %208, 1
  %269 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %268, i32 0, i32 0)
  %270 = shl i32 %209, 1
  %271 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %270, i32 0, i32 0)
  %272 = shl i32 %210, 1
  %273 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %272, i32 0, i32 0)
  %274 = shl i32 %211, 1
  %275 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %274, i32 0, i32 0)
  %276 = shl i32 %212, 1
  %277 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %276, i32 0, i32 0)
  %278 = shl i32 %213, 1
  %279 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %278, i32 0, i32 0)
  %280 = shl i32 %214, 1
  %281 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %280, i32 0, i32 0)
  %282 = shl i32 %215, 1
  %283 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %282, i32 0, i32 0)
  %284 = shl i32 %216, 1
  %285 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %284, i32 0, i32 0)
  %286 = shl i32 %217, 1
  %287 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %286, i32 0, i32 0)
  %288 = shl i32 %218, 1
  %289 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %288, i32 0, i32 0)
  %290 = shl i32 %219, 1
  %291 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %290, i32 0, i32 0)
  %292 = shl i32 %220, 1
  %293 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %292, i32 0, i32 0)
  %294 = shl i32 %221, 1
  %295 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %294, i32 0, i32 0)
  %296 = shl i32 %222, 1
  %297 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %296, i32 0, i32 0)
  %298 = shl i32 %223, 1
  %299 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %298, i32 0, i32 0)
  %300 = shl i32 %224, 1
  %301 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %300, i32 0, i32 0)
  %302 = shl i32 %225, 1
  %303 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %302, i32 0, i32 0)
  %304 = shl i32 %226, 1
  %305 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %304, i32 0, i32 0)
  %306 = shl i32 %227, 1
  %307 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %306, i32 0, i32 0)
  %308 = shl i32 %228, 1
  %309 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %308, i32 0, i32 0)
  %310 = shl i32 %229, 1
  %311 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %310, i32 0, i32 0)
  %312 = shl i32 %230, 1
  %313 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %312, i32 0, i32 0)
  %314 = shl i32 %231, 1
  %315 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %314, i32 0, i32 0)
  %316 = shl i32 %232, 1
  %317 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %316, i32 0, i32 0)
  %318 = shl i32 %233, 1
  %319 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %318, i32 0, i32 0)
  %320 = shl i32 %234, 1
  %321 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %320, i32 0, i32 0)
  %322 = shl i32 %235, 1
  %323 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %322, i32 0, i32 0)
  %324 = shl i32 %236, 1
  %325 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %324, i32 0, i32 0)
  %326 = shl i32 %237, 1
  %327 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %326, i32 0, i32 0)
  %328 = shl i32 %238, 1
  %329 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %328, i32 0, i32 0)
  %330 = shl i32 %84, 1
  %331 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %85, i32 0, i32 0)
  %332 = shl i32 1, 1
  %333 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 1, i32 0, i32 0)
  %334 = shl i32 %86, 1
  %335 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %87, i32 0, i32 0)
  %336 = shl i32 %88, 1
  %337 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %89, i32 0, i32 0)
  %338 = shl i32 %90, 1
  %339 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %91, i32 0, i32 0)
  %340 = shl i32 %92, 1
  %341 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %93, i32 0, i32 0)
  %342 = shl i32 %94, 1
  %343 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %95, i32 0, i32 0)
  %344 = shl i32 %96, 0
  %345 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %97, i32 0, i32 0)
  %346 = shl i32 %98, 0
  %347 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %99, i32 0, i32 0)
  %348 = shl i32 %100, 1
  %349 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %101, i32 0, i32 0)
  %350 = insertelement <8 x i16> zeroinitializer, i16 0, i64 0
  %351 = insertelement <8 x i16> zeroinitializer, i16 0, i64 0
  %352 = insertelement <8 x i16> zeroinitializer, i16 0, i64 0
  %353 = insertelement <8 x i16> zeroinitializer, i16 0, i64 0
  %354 = insertelement <8 x i16> zeroinitializer, i16 0, i64 0
  %355 = insertelement <8 x i16> zeroinitializer, i16 0, i64 0
  %356 = insertelement <8 x i16> zeroinitializer, i16 1, i64 1
  %357 = insertelement <8 x i16> splat (i16 1), i16 %102, i64 1
  store <8 x i16> %103, ptr addrspace(3) null, align 16
  tail call void @llvm.amdgcn.s.barrier()
  %358 = insertelement <8 x i16> zeroinitializer, i16 %104, i64 0
  %359 = insertelement <8 x i16> %105, i16 %106, i64 1
  %360 = insertelement <8 x i16> %107, i16 %263, i64 2
  %361 = insertelement <8 x i16> %360, i16 %267, i64 3
  %362 = insertelement <8 x i16> %361, i16 %271, i64 4
  %363 = insertelement <8 x i16> %362, i16 %275, i64 5
  %364 = insertelement <8 x i16> %363, i16 %279, i64 6
  %365 = insertelement <8 x i16> %364, i16 %283, i64 7
  store <8 x i16> %365, ptr addrspace(3) null, align 16
  %366 = insertelement <8 x i16> zeroinitializer, i16 %108, i64 0
  %367 = insertelement <8 x i16> %109, i16 %110, i64 1
  %368 = insertelement <8 x i16> %111, i16 %112, i64 2
  %369 = insertelement <8 x i16> %113, i16 %269, i64 3
  %370 = insertelement <8 x i16> %369, i16 %273, i64 4
  %371 = insertelement <8 x i16> %370, i16 %277, i64 5
  %372 = insertelement <8 x i16> %371, i16 %281, i64 6
  %373 = insertelement <8 x i16> %372, i16 %285, i64 7
  store <8 x i16> %373, ptr addrspace(3) %114, align 16
  %374 = insertelement <8 x i16> zeroinitializer, i16 %287, i64 0
  %375 = insertelement <8 x i16> %374, i16 %291, i64 1
  %376 = insertelement <8 x i16> %375, i16 %295, i64 2
  %377 = insertelement <8 x i16> %376, i16 %299, i64 3
  %378 = insertelement <8 x i16> %377, i16 %303, i64 4
  %379 = insertelement <8 x i16> %378, i16 %307, i64 5
  %380 = insertelement <8 x i16> %379, i16 %311, i64 6
  %381 = insertelement <8 x i16> %380, i16 %315, i64 7
  store <8 x i16> %381, ptr addrspace(3) null, align 16
  %382 = insertelement <8 x i16> zeroinitializer, i16 %289, i64 0
  %383 = insertelement <8 x i16> %382, i16 %293, i64 1
  %384 = insertelement <8 x i16> %383, i16 %297, i64 2
  %385 = insertelement <8 x i16> %384, i16 %301, i64 3
  %386 = insertelement <8 x i16> %385, i16 %305, i64 4
  %387 = insertelement <8 x i16> %386, i16 %309, i64 5
  %388 = insertelement <8 x i16> %387, i16 %313, i64 6
  %389 = insertelement <8 x i16> %388, i16 %317, i64 7
  store <8 x i16> %389, ptr addrspace(3) %114, align 16
  fence syncscope("workgroup") release
  %390 = insertelement <8 x i16> zeroinitializer, i16 %319, i64 0
  %391 = insertelement <8 x i16> %390, i16 %323, i64 1
  %392 = insertelement <8 x i16> %391, i16 %327, i64 2
  %393 = insertelement <8 x i16> %392, i16 %115, i64 3
  %394 = insertelement <8 x i16> %393, i16 %335, i64 4
  %395 = insertelement <8 x i16> %394, i16 %116, i64 5
  %396 = insertelement <8 x i16> %395, i16 %117, i64 6
  %397 = insertelement <8 x i16> %396, i16 %118, i64 7
  store <8 x i16> %397, ptr addrspace(3) null, align 16
  %398 = insertelement <8 x i16> zeroinitializer, i16 %321, i64 0
  %399 = insertelement <8 x i16> %398, i16 %325, i64 1
  %400 = insertelement <8 x i16> %399, i16 %329, i64 2
  %401 = insertelement <8 x i16> %400, i16 %119, i64 3
  %402 = insertelement <8 x i16> %401, i16 %120, i64 4
  %403 = insertelement <8 x i16> %402, i16 %121, i64 5
  %404 = insertelement <8 x i16> %403, i16 %122, i64 6
  %405 = insertelement <8 x i16> %404, i16 %123, i64 7
  store <8 x i16> %405, ptr addrspace(3) %114, align 16
  %406 = select i1 %194, i32 0, i32 -2147483648
  %407 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %139, i32 %406, i32 0, i32 0)
  %408 = bitcast <4 x i32> %407 to <8 x bfloat>
  %409 = extractelement <8 x bfloat> %408, i64 0
  %410 = insertelement <4 x bfloat> zeroinitializer, bfloat %409, i64 0
  %411 = insertelement <4 x bfloat> %410, bfloat 0.000000e+00, i64 1
  %412 = insertelement <4 x bfloat> %411, bfloat 0.000000e+00, i64 1
  %413 = insertelement <4 x bfloat> %412, bfloat 0.000000e+00, i64 1
  store <4 x bfloat> %413, ptr addrspace(3) null, align 8
  %414 = add i32 0, 0
  %415 = add i32 1, %124
  %416 = add i32 1, 1
  %417 = add i32 %57, 1
  %418 = add i32 %58, 1
  %419 = add i32 0, 0
  %420 = add i32 1, %124
  %421 = add i32 %125, %124
  %422 = add i32 %61, 1
  %423 = add i32 %126, 1
  %424 = add i32 %127, 1
  %425 = add i32 1, %124
  %426 = add i32 1, %124
  %427 = add i32 1, %124
  %428 = add i32 1, %124
  %429 = add i32 1, %124
  %430 = add i32 %128, 1
  %431 = add i32 %129, 1
  %432 = add i32 1, %124
  %433 = add i32 1, %124
  %434 = add i32 %130, %124
  %435 = add i32 1, %124
  %436 = add i32 1, %124
  %437 = add i32 1, %124
  %438 = add i32 1, %124
  %439 = add i32 %131, %124
  %440 = add i32 1, %124
  %441 = add i32 %132, %124
  %442 = add i32 %133, %124
  %443 = add i32 %134, %124
  %444 = add i32 1, %124
  %445 = add i32 1, %124
  %446 = add i32 1, %124
  %447 = add i32 %135, %124
  %448 = add i32 1, %124
  %449 = add i32 1, %124
  %450 = add i32 1, %124
  %451 = add i32 1, %124
  %452 = add i32 1, %124
  %453 = add i32 1, %124
  %454 = add i32 1, %124
  %455 = add i32 1, %124
  %456 = add i32 %63, %124
  %457 = add i32 1, %124
  %458 = add i32 1, %124
  %459 = add i32 1, %124
  %460 = add i32 1, %124
  %461 = add i32 0, 0
  %462 = add i32 0, %124
  %463 = add i32 %70, 1
  %464 = add i32 1, %124
  %465 = add i32 0, %124
  br label %140
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #2

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.f32(float, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.v4f32(<4 x float>, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #4

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16>, <16 x i16>, <8 x float>) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.permlanex16.i32(i32, i32, i32, i32, i1 immarg, i1 immarg) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #6

; Function Attrs: nocallback nofree nounwind speculatable willreturn memory(none)
declare i32 @llvm.amdgcn.perm(i32, i32, i32) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1) readnone, i16, i64, i32) #6

; uselistorder directives
uselistorder ptr @llvm.amdgcn.raw.ptr.buffer.load.v4i32, { 1, 0 }
uselistorder ptr @llvm.amdgcn.raw.ptr.buffer.load.i16, { 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
uselistorder ptr @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64, { 1, 0 }

attributes #0 = { norecurse nounwind "amdgpu-cluster-dims"="1,1,1" "amdgpu-flat-work-group-size"="1,128" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "amdgpu-waves-per-eu"="0, 0" "denormal-fp-math-f32"="ieee" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #3 = { convergent nocallback nofree nounwind willreturn }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
attributes #5 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { nocallback nofree nounwind speculatable willreturn memory(none) }
