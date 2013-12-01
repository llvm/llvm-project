; RUN: llc -march=r600 -mcpu=SI < %s | FileCheck %s

; XXX: Enable when spilling is supported
; XFAIL: *

; These tests check that the compiler won't crash when it needs to spill
; SGPRs.

; CHECK-LABEL: @main
; Writing to M0 from an SMRD instruction will hang the GPU.
; CHECK-NOT: S_BUFFER_LOAD_DWORD m0
; CHECK: S_ENDPGM
@ddxy_lds = external addrspace(3) global [64 x i32]

define void @main([17 x <16 x i8>] addrspace(2)* byval, [32 x <16 x i8>] addrspace(2)* byval, [16 x <32 x i8>] addrspace(2)* byval, float inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, float, float, float) #0 {
main_body:
  %21 = getelementptr [17 x <16 x i8>] addrspace(2)* %0, i64 0, i32 0
  %22 = load <16 x i8> addrspace(2)* %21, !tbaa !0
  %23 = call float @llvm.SI.load.const(<16 x i8> %22, i32 96)
  %24 = call float @llvm.SI.load.const(<16 x i8> %22, i32 100)
  %25 = call float @llvm.SI.load.const(<16 x i8> %22, i32 104)
  %26 = call float @llvm.SI.load.const(<16 x i8> %22, i32 112)
  %27 = call float @llvm.SI.load.const(<16 x i8> %22, i32 116)
  %28 = call float @llvm.SI.load.const(<16 x i8> %22, i32 120)
  %29 = call float @llvm.SI.load.const(<16 x i8> %22, i32 128)
  %30 = call float @llvm.SI.load.const(<16 x i8> %22, i32 132)
  %31 = call float @llvm.SI.load.const(<16 x i8> %22, i32 140)
  %32 = call float @llvm.SI.load.const(<16 x i8> %22, i32 144)
  %33 = call float @llvm.SI.load.const(<16 x i8> %22, i32 160)
  %34 = call float @llvm.SI.load.const(<16 x i8> %22, i32 176)
  %35 = call float @llvm.SI.load.const(<16 x i8> %22, i32 180)
  %36 = call float @llvm.SI.load.const(<16 x i8> %22, i32 184)
  %37 = call float @llvm.SI.load.const(<16 x i8> %22, i32 192)
  %38 = call float @llvm.SI.load.const(<16 x i8> %22, i32 196)
  %39 = call float @llvm.SI.load.const(<16 x i8> %22, i32 200)
  %40 = call float @llvm.SI.load.const(<16 x i8> %22, i32 208)
  %41 = call float @llvm.SI.load.const(<16 x i8> %22, i32 212)
  %42 = call float @llvm.SI.load.const(<16 x i8> %22, i32 216)
  %43 = call float @llvm.SI.load.const(<16 x i8> %22, i32 224)
  %44 = call float @llvm.SI.load.const(<16 x i8> %22, i32 240)
  %45 = call float @llvm.SI.load.const(<16 x i8> %22, i32 244)
  %46 = call float @llvm.SI.load.const(<16 x i8> %22, i32 248)
  %47 = call float @llvm.SI.load.const(<16 x i8> %22, i32 256)
  %48 = call float @llvm.SI.load.const(<16 x i8> %22, i32 272)
  %49 = call float @llvm.SI.load.const(<16 x i8> %22, i32 276)
  %50 = call float @llvm.SI.load.const(<16 x i8> %22, i32 280)
  %51 = call float @llvm.SI.load.const(<16 x i8> %22, i32 288)
  %52 = call float @llvm.SI.load.const(<16 x i8> %22, i32 292)
  %53 = call float @llvm.SI.load.const(<16 x i8> %22, i32 296)
  %54 = call float @llvm.SI.load.const(<16 x i8> %22, i32 304)
  %55 = call float @llvm.SI.load.const(<16 x i8> %22, i32 308)
  %56 = call float @llvm.SI.load.const(<16 x i8> %22, i32 312)
  %57 = call float @llvm.SI.load.const(<16 x i8> %22, i32 368)
  %58 = call float @llvm.SI.load.const(<16 x i8> %22, i32 372)
  %59 = call float @llvm.SI.load.const(<16 x i8> %22, i32 376)
  %60 = call float @llvm.SI.load.const(<16 x i8> %22, i32 384)
  %61 = getelementptr [16 x <32 x i8>] addrspace(2)* %2, i64 0, i32 0
  %62 = load <32 x i8> addrspace(2)* %61, !tbaa !0
  %63 = getelementptr [32 x <16 x i8>] addrspace(2)* %1, i64 0, i32 0
  %64 = load <16 x i8> addrspace(2)* %63, !tbaa !0
  %65 = getelementptr [16 x <32 x i8>] addrspace(2)* %2, i64 0, i32 1
  %66 = load <32 x i8> addrspace(2)* %65, !tbaa !0
  %67 = getelementptr [32 x <16 x i8>] addrspace(2)* %1, i64 0, i32 1
  %68 = load <16 x i8> addrspace(2)* %67, !tbaa !0
  %69 = getelementptr [16 x <32 x i8>] addrspace(2)* %2, i64 0, i32 2
  %70 = load <32 x i8> addrspace(2)* %69, !tbaa !0
  %71 = getelementptr [32 x <16 x i8>] addrspace(2)* %1, i64 0, i32 2
  %72 = load <16 x i8> addrspace(2)* %71, !tbaa !0
  %73 = getelementptr [16 x <32 x i8>] addrspace(2)* %2, i64 0, i32 3
  %74 = load <32 x i8> addrspace(2)* %73, !tbaa !0
  %75 = getelementptr [32 x <16 x i8>] addrspace(2)* %1, i64 0, i32 3
  %76 = load <16 x i8> addrspace(2)* %75, !tbaa !0
  %77 = getelementptr [16 x <32 x i8>] addrspace(2)* %2, i64 0, i32 4
  %78 = load <32 x i8> addrspace(2)* %77, !tbaa !0
  %79 = getelementptr [32 x <16 x i8>] addrspace(2)* %1, i64 0, i32 4
  %80 = load <16 x i8> addrspace(2)* %79, !tbaa !0
  %81 = getelementptr [16 x <32 x i8>] addrspace(2)* %2, i64 0, i32 5
  %82 = load <32 x i8> addrspace(2)* %81, !tbaa !0
  %83 = getelementptr [32 x <16 x i8>] addrspace(2)* %1, i64 0, i32 5
  %84 = load <16 x i8> addrspace(2)* %83, !tbaa !0
  %85 = getelementptr [16 x <32 x i8>] addrspace(2)* %2, i64 0, i32 6
  %86 = load <32 x i8> addrspace(2)* %85, !tbaa !0
  %87 = getelementptr [32 x <16 x i8>] addrspace(2)* %1, i64 0, i32 6
  %88 = load <16 x i8> addrspace(2)* %87, !tbaa !0
  %89 = getelementptr [16 x <32 x i8>] addrspace(2)* %2, i64 0, i32 7
  %90 = load <32 x i8> addrspace(2)* %89, !tbaa !0
  %91 = getelementptr [32 x <16 x i8>] addrspace(2)* %1, i64 0, i32 7
  %92 = load <16 x i8> addrspace(2)* %91, !tbaa !0
  %93 = call float @llvm.SI.fs.interp(i32 0, i32 0, i32 %4, <2 x i32> %6)
  %94 = call float @llvm.SI.fs.interp(i32 1, i32 0, i32 %4, <2 x i32> %6)
  %95 = call float @llvm.SI.fs.interp(i32 0, i32 1, i32 %4, <2 x i32> %6)
  %96 = call float @llvm.SI.fs.interp(i32 1, i32 1, i32 %4, <2 x i32> %6)
  %97 = call float @llvm.SI.fs.interp(i32 2, i32 1, i32 %4, <2 x i32> %6)
  %98 = call float @llvm.SI.fs.interp(i32 0, i32 2, i32 %4, <2 x i32> %6)
  %99 = call float @llvm.SI.fs.interp(i32 1, i32 2, i32 %4, <2 x i32> %6)
  %100 = call float @llvm.SI.fs.interp(i32 2, i32 2, i32 %4, <2 x i32> %6)
  %101 = call float @llvm.SI.fs.interp(i32 0, i32 3, i32 %4, <2 x i32> %6)
  %102 = call float @llvm.SI.fs.interp(i32 1, i32 3, i32 %4, <2 x i32> %6)
  %103 = call float @llvm.SI.fs.interp(i32 2, i32 3, i32 %4, <2 x i32> %6)
  %104 = call float @llvm.SI.fs.interp(i32 0, i32 4, i32 %4, <2 x i32> %6)
  %105 = call float @llvm.SI.fs.interp(i32 1, i32 4, i32 %4, <2 x i32> %6)
  %106 = call float @llvm.SI.fs.interp(i32 2, i32 4, i32 %4, <2 x i32> %6)
  %107 = call float @llvm.SI.fs.interp(i32 0, i32 5, i32 %4, <2 x i32> %6)
  %108 = call float @llvm.SI.fs.interp(i32 1, i32 5, i32 %4, <2 x i32> %6)
  %109 = call float @llvm.SI.fs.interp(i32 2, i32 5, i32 %4, <2 x i32> %6)
  %110 = call i32 @llvm.SI.tid()
  %111 = getelementptr [64 x i32] addrspace(3)* @ddxy_lds, i32 0, i32 %110
  %112 = bitcast float %93 to i32
  store i32 %112, i32 addrspace(3)* %111
  %113 = bitcast float %94 to i32
  store i32 %113, i32 addrspace(3)* %111
  %114 = call i32 @llvm.SI.tid()
  %115 = getelementptr [64 x i32] addrspace(3)* @ddxy_lds, i32 0, i32 %114
  %116 = and i32 %114, -4
  %117 = getelementptr [64 x i32] addrspace(3)* @ddxy_lds, i32 0, i32 %116
  %118 = add i32 %116, 1
  %119 = getelementptr [64 x i32] addrspace(3)* @ddxy_lds, i32 0, i32 %118
  %120 = bitcast float %93 to i32
  store i32 %120, i32 addrspace(3)* %115
  %121 = load i32 addrspace(3)* %117
  %122 = bitcast i32 %121 to float
  %123 = load i32 addrspace(3)* %119
  %124 = bitcast i32 %123 to float
  %125 = fsub float %124, %122
  %126 = bitcast float %94 to i32
  store i32 %126, i32 addrspace(3)* %115
  %127 = load i32 addrspace(3)* %117
  %128 = bitcast i32 %127 to float
  %129 = load i32 addrspace(3)* %119
  %130 = bitcast i32 %129 to float
  %131 = fsub float %130, %128
  %132 = insertelement <4 x float> undef, float %125, i32 0
  %133 = insertelement <4 x float> %132, float %131, i32 1
  %134 = insertelement <4 x float> %133, float %131, i32 2
  %135 = insertelement <4 x float> %134, float %131, i32 3
  %136 = extractelement <4 x float> %135, i32 0
  %137 = extractelement <4 x float> %135, i32 1
  %138 = fmul float %60, %93
  %139 = fmul float %60, %94
  %140 = fmul float %60, %94
  %141 = fmul float %60, %94
  %142 = call i32 @llvm.SI.tid()
  %143 = getelementptr [64 x i32] addrspace(3)* @ddxy_lds, i32 0, i32 %142
  %144 = bitcast float %138 to i32
  store i32 %144, i32 addrspace(3)* %143
  %145 = bitcast float %139 to i32
  store i32 %145, i32 addrspace(3)* %143
  %146 = bitcast float %140 to i32
  store i32 %146, i32 addrspace(3)* %143
  %147 = bitcast float %141 to i32
  store i32 %147, i32 addrspace(3)* %143
  %148 = call i32 @llvm.SI.tid()
  %149 = getelementptr [64 x i32] addrspace(3)* @ddxy_lds, i32 0, i32 %148
  %150 = and i32 %148, -4
  %151 = getelementptr [64 x i32] addrspace(3)* @ddxy_lds, i32 0, i32 %150
  %152 = add i32 %150, 2
  %153 = getelementptr [64 x i32] addrspace(3)* @ddxy_lds, i32 0, i32 %152
  %154 = bitcast float %138 to i32
  store i32 %154, i32 addrspace(3)* %149
  %155 = load i32 addrspace(3)* %151
  %156 = bitcast i32 %155 to float
  %157 = load i32 addrspace(3)* %153
  %158 = bitcast i32 %157 to float
  %159 = fsub float %158, %156
  %160 = bitcast float %139 to i32
  store i32 %160, i32 addrspace(3)* %149
  %161 = load i32 addrspace(3)* %151
  %162 = bitcast i32 %161 to float
  %163 = load i32 addrspace(3)* %153
  %164 = bitcast i32 %163 to float
  %165 = fsub float %164, %162
  %166 = bitcast float %140 to i32
  store i32 %166, i32 addrspace(3)* %149
  %167 = load i32 addrspace(3)* %151
  %168 = bitcast i32 %167 to float
  %169 = load i32 addrspace(3)* %153
  %170 = bitcast i32 %169 to float
  %171 = fsub float %170, %168
  %172 = bitcast float %141 to i32
  store i32 %172, i32 addrspace(3)* %149
  %173 = load i32 addrspace(3)* %151
  %174 = bitcast i32 %173 to float
  %175 = load i32 addrspace(3)* %153
  %176 = bitcast i32 %175 to float
  %177 = fsub float %176, %174
  %178 = insertelement <4 x float> undef, float %159, i32 0
  %179 = insertelement <4 x float> %178, float %165, i32 1
  %180 = insertelement <4 x float> %179, float %171, i32 2
  %181 = insertelement <4 x float> %180, float %177, i32 3
  %182 = extractelement <4 x float> %181, i32 0
  %183 = extractelement <4 x float> %181, i32 1
  %184 = fdiv float 1.000000e+00, %97
  %185 = fmul float %33, %184
  %186 = fcmp uge float 1.000000e+00, %185
  %187 = select i1 %186, float %185, float 1.000000e+00
  %188 = fmul float %187, %30
  %189 = call float @ceil(float %188)
  %190 = fcmp uge float 3.000000e+00, %189
  %191 = select i1 %190, float 3.000000e+00, float %189
  %192 = fdiv float 1.000000e+00, %191
  %193 = fdiv float 1.000000e+00, %30
  %194 = fmul float %191, %193
  %195 = fmul float %31, %194
  %196 = fmul float %95, %95
  %197 = fmul float %96, %96
  %198 = fadd float %197, %196
  %199 = fmul float %97, %97
  %200 = fadd float %198, %199
  %201 = call float @llvm.AMDGPU.rsq(float %200)
  %202 = fmul float %95, %201
  %203 = fmul float %96, %201
  %204 = fmul float %202, %29
  %205 = fmul float %203, %29
  %206 = fmul float %204, -1.000000e+00
  %207 = fmul float %205, 1.000000e+00
  %208 = fmul float %206, %32
  %209 = fmul float %207, %32
  %210 = fsub float -0.000000e+00, %208
  %211 = fadd float %93, %210
  %212 = fsub float -0.000000e+00, %209
  %213 = fadd float %94, %212
  %214 = fmul float %206, %192
  %215 = fmul float %207, %192
  %216 = fmul float -1.000000e+00, %192
  %217 = bitcast float %136 to i32
  %218 = bitcast float %182 to i32
  %219 = bitcast float %137 to i32
  %220 = bitcast float %183 to i32
  %221 = insertelement <8 x i32> undef, i32 %217, i32 0
  %222 = insertelement <8 x i32> %221, i32 %218, i32 1
  %223 = insertelement <8 x i32> %222, i32 %219, i32 2
  %224 = insertelement <8 x i32> %223, i32 %220, i32 3
  br label %LOOP

LOOP:                                             ; preds = %ENDIF, %main_body
  %temp24.0 = phi float [ 1.000000e+00, %main_body ], [ %258, %ENDIF ]
  %temp28.0 = phi float [ %211, %main_body ], [ %253, %ENDIF ]
  %temp29.0 = phi float [ %213, %main_body ], [ %255, %ENDIF ]
  %temp30.0 = phi float [ 1.000000e+00, %main_body ], [ %257, %ENDIF ]
  %225 = fcmp oge float %temp24.0, %191
  %226 = sext i1 %225 to i32
  %227 = bitcast i32 %226 to float
  %228 = bitcast float %227 to i32
  %229 = icmp ne i32 %228, 0
  br i1 %229, label %IF, label %ENDIF

IF:                                               ; preds = %LOOP
  %230 = bitcast float %136 to i32
  %231 = bitcast float %182 to i32
  %232 = bitcast float %137 to i32
  %233 = bitcast float %183 to i32
  %234 = insertelement <8 x i32> undef, i32 %230, i32 0
  %235 = insertelement <8 x i32> %234, i32 %231, i32 1
  %236 = insertelement <8 x i32> %235, i32 %232, i32 2
  %237 = insertelement <8 x i32> %236, i32 %233, i32 3
  br label %LOOP65

ENDIF:                                            ; preds = %LOOP
  %238 = bitcast float %temp28.0 to i32
  %239 = bitcast float %temp29.0 to i32
  %240 = insertelement <8 x i32> %224, i32 %238, i32 4
  %241 = insertelement <8 x i32> %240, i32 %239, i32 5
  %242 = insertelement <8 x i32> %241, i32 undef, i32 6
  %243 = insertelement <8 x i32> %242, i32 undef, i32 7
  %244 = call <4 x float> @llvm.SI.sampled.v8i32(<8 x i32> %243, <32 x i8> %62, <16 x i8> %64, i32 2)
  %245 = extractelement <4 x float> %244, i32 3
  %246 = fcmp oge float %temp30.0, %245
  %247 = sext i1 %246 to i32
  %248 = bitcast i32 %247 to float
  %249 = bitcast float %248 to i32
  %250 = and i32 %249, 1065353216
  %251 = bitcast i32 %250 to float
  %252 = fmul float %214, %251
  %253 = fadd float %252, %temp28.0
  %254 = fmul float %215, %251
  %255 = fadd float %254, %temp29.0
  %256 = fmul float %216, %251
  %257 = fadd float %256, %temp30.0
  %258 = fadd float %temp24.0, 1.000000e+00
  br label %LOOP

LOOP65:                                           ; preds = %ENDIF66, %IF
  %temp24.1 = phi float [ 0.000000e+00, %IF ], [ %610, %ENDIF66 ]
  %temp28.1 = phi float [ %temp28.0, %IF ], [ %605, %ENDIF66 ]
  %temp29.1 = phi float [ %temp29.0, %IF ], [ %607, %ENDIF66 ]
  %temp30.1 = phi float [ %temp30.0, %IF ], [ %609, %ENDIF66 ]
  %temp32.0 = phi float [ 1.000000e+00, %IF ], [ %611, %ENDIF66 ]
  %259 = fcmp oge float %temp24.1, %195
  %260 = sext i1 %259 to i32
  %261 = bitcast i32 %260 to float
  %262 = bitcast float %261 to i32
  %263 = icmp ne i32 %262, 0
  br i1 %263, label %IF67, label %ENDIF66

IF67:                                             ; preds = %LOOP65
  %264 = bitcast float %136 to i32
  %265 = bitcast float %182 to i32
  %266 = bitcast float %137 to i32
  %267 = bitcast float %183 to i32
  %268 = bitcast float %temp28.1 to i32
  %269 = bitcast float %temp29.1 to i32
  %270 = insertelement <8 x i32> undef, i32 %264, i32 0
  %271 = insertelement <8 x i32> %270, i32 %265, i32 1
  %272 = insertelement <8 x i32> %271, i32 %266, i32 2
  %273 = insertelement <8 x i32> %272, i32 %267, i32 3
  %274 = insertelement <8 x i32> %273, i32 %268, i32 4
  %275 = insertelement <8 x i32> %274, i32 %269, i32 5
  %276 = insertelement <8 x i32> %275, i32 undef, i32 6
  %277 = insertelement <8 x i32> %276, i32 undef, i32 7
  %278 = call <4 x float> @llvm.SI.sampled.v8i32(<8 x i32> %277, <32 x i8> %66, <16 x i8> %68, i32 2)
  %279 = extractelement <4 x float> %278, i32 0
  %280 = extractelement <4 x float> %278, i32 1
  %281 = extractelement <4 x float> %278, i32 2
  %282 = extractelement <4 x float> %278, i32 3
  %283 = fmul float %282, %47
  %284 = bitcast float %136 to i32
  %285 = bitcast float %182 to i32
  %286 = bitcast float %137 to i32
  %287 = bitcast float %183 to i32
  %288 = bitcast float %temp28.1 to i32
  %289 = bitcast float %temp29.1 to i32
  %290 = insertelement <8 x i32> undef, i32 %284, i32 0
  %291 = insertelement <8 x i32> %290, i32 %285, i32 1
  %292 = insertelement <8 x i32> %291, i32 %286, i32 2
  %293 = insertelement <8 x i32> %292, i32 %287, i32 3
  %294 = insertelement <8 x i32> %293, i32 %288, i32 4
  %295 = insertelement <8 x i32> %294, i32 %289, i32 5
  %296 = insertelement <8 x i32> %295, i32 undef, i32 6
  %297 = insertelement <8 x i32> %296, i32 undef, i32 7
  %298 = call <4 x float> @llvm.SI.sampled.v8i32(<8 x i32> %297, <32 x i8> %82, <16 x i8> %84, i32 2)
  %299 = extractelement <4 x float> %298, i32 0
  %300 = extractelement <4 x float> %298, i32 1
  %301 = extractelement <4 x float> %298, i32 2
  %302 = bitcast float %136 to i32
  %303 = bitcast float %182 to i32
  %304 = bitcast float %137 to i32
  %305 = bitcast float %183 to i32
  %306 = bitcast float %temp28.1 to i32
  %307 = bitcast float %temp29.1 to i32
  %308 = insertelement <8 x i32> undef, i32 %302, i32 0
  %309 = insertelement <8 x i32> %308, i32 %303, i32 1
  %310 = insertelement <8 x i32> %309, i32 %304, i32 2
  %311 = insertelement <8 x i32> %310, i32 %305, i32 3
  %312 = insertelement <8 x i32> %311, i32 %306, i32 4
  %313 = insertelement <8 x i32> %312, i32 %307, i32 5
  %314 = insertelement <8 x i32> %313, i32 undef, i32 6
  %315 = insertelement <8 x i32> %314, i32 undef, i32 7
  %316 = call <4 x float> @llvm.SI.sampled.v8i32(<8 x i32> %315, <32 x i8> %78, <16 x i8> %80, i32 2)
  %317 = extractelement <4 x float> %316, i32 0
  %318 = extractelement <4 x float> %316, i32 1
  %319 = extractelement <4 x float> %316, i32 2
  %320 = fmul float %317, %23
  %321 = fmul float %318, %24
  %322 = fmul float %319, %25
  %323 = fmul float %299, %26
  %324 = fadd float %323, %320
  %325 = fmul float %300, %27
  %326 = fadd float %325, %321
  %327 = fmul float %301, %28
  %328 = fadd float %327, %322
  %329 = fadd float %279, %324
  %330 = fadd float %280, %326
  %331 = fadd float %281, %328
  %332 = bitcast float %136 to i32
  %333 = bitcast float %182 to i32
  %334 = bitcast float %137 to i32
  %335 = bitcast float %183 to i32
  %336 = bitcast float %temp28.1 to i32
  %337 = bitcast float %temp29.1 to i32
  %338 = insertelement <8 x i32> undef, i32 %332, i32 0
  %339 = insertelement <8 x i32> %338, i32 %333, i32 1
  %340 = insertelement <8 x i32> %339, i32 %334, i32 2
  %341 = insertelement <8 x i32> %340, i32 %335, i32 3
  %342 = insertelement <8 x i32> %341, i32 %336, i32 4
  %343 = insertelement <8 x i32> %342, i32 %337, i32 5
  %344 = insertelement <8 x i32> %343, i32 undef, i32 6
  %345 = insertelement <8 x i32> %344, i32 undef, i32 7
  %346 = call <4 x float> @llvm.SI.sampled.v8i32(<8 x i32> %345, <32 x i8> %62, <16 x i8> %64, i32 2)
  %347 = extractelement <4 x float> %346, i32 0
  %348 = extractelement <4 x float> %346, i32 1
  %349 = extractelement <4 x float> %346, i32 2
  %350 = fadd float %347, -5.000000e-01
  %351 = fadd float %348, -5.000000e-01
  %352 = fadd float %349, -5.000000e-01
  %353 = fmul float %350, %350
  %354 = fmul float %351, %351
  %355 = fadd float %354, %353
  %356 = fmul float %352, %352
  %357 = fadd float %355, %356
  %358 = call float @llvm.AMDGPU.rsq(float %357)
  %359 = fmul float %350, %358
  %360 = fmul float %351, %358
  %361 = fmul float %352, %358
  %362 = bitcast float %136 to i32
  %363 = bitcast float %182 to i32
  %364 = bitcast float %137 to i32
  %365 = bitcast float %183 to i32
  %366 = bitcast float %temp28.1 to i32
  %367 = bitcast float %temp29.1 to i32
  %368 = insertelement <8 x i32> undef, i32 %362, i32 0
  %369 = insertelement <8 x i32> %368, i32 %363, i32 1
  %370 = insertelement <8 x i32> %369, i32 %364, i32 2
  %371 = insertelement <8 x i32> %370, i32 %365, i32 3
  %372 = insertelement <8 x i32> %371, i32 %366, i32 4
  %373 = insertelement <8 x i32> %372, i32 %367, i32 5
  %374 = insertelement <8 x i32> %373, i32 undef, i32 6
  %375 = insertelement <8 x i32> %374, i32 undef, i32 7
  %376 = call <4 x float> @llvm.SI.sampled.v8i32(<8 x i32> %375, <32 x i8> %70, <16 x i8> %72, i32 2)
  %377 = extractelement <4 x float> %376, i32 0
  %378 = extractelement <4 x float> %376, i32 1
  %379 = extractelement <4 x float> %376, i32 2
  %380 = extractelement <4 x float> %376, i32 3
  %381 = fsub float -0.000000e+00, %95
  %382 = fsub float -0.000000e+00, %96
  %383 = fsub float -0.000000e+00, %97
  %384 = fmul float %359, %381
  %385 = fmul float %360, %382
  %386 = fadd float %385, %384
  %387 = fmul float %361, %383
  %388 = fadd float %386, %387
  %389 = fmul float %388, %359
  %390 = fmul float %388, %360
  %391 = fmul float %388, %361
  %392 = fmul float 2.000000e+00, %389
  %393 = fmul float 2.000000e+00, %390
  %394 = fmul float 2.000000e+00, %391
  %395 = fsub float -0.000000e+00, %392
  %396 = fadd float %381, %395
  %397 = fsub float -0.000000e+00, %393
  %398 = fadd float %382, %397
  %399 = fsub float -0.000000e+00, %394
  %400 = fadd float %383, %399
  %401 = fmul float %396, %98
  %402 = fmul float %396, %99
  %403 = fmul float %396, %100
  %404 = fmul float %398, %101
  %405 = fadd float %404, %401
  %406 = fmul float %398, %102
  %407 = fadd float %406, %402
  %408 = fmul float %398, %103
  %409 = fadd float %408, %403
  %410 = fmul float %400, %104
  %411 = fadd float %410, %405
  %412 = fmul float %400, %105
  %413 = fadd float %412, %407
  %414 = fmul float %400, %106
  %415 = fadd float %414, %409
  %416 = bitcast float %136 to i32
  %417 = bitcast float %182 to i32
  %418 = bitcast float %137 to i32
  %419 = bitcast float %183 to i32
  %420 = bitcast float %temp28.1 to i32
  %421 = bitcast float %temp29.1 to i32
  %422 = insertelement <8 x i32> undef, i32 %416, i32 0
  %423 = insertelement <8 x i32> %422, i32 %417, i32 1
  %424 = insertelement <8 x i32> %423, i32 %418, i32 2
  %425 = insertelement <8 x i32> %424, i32 %419, i32 3
  %426 = insertelement <8 x i32> %425, i32 %420, i32 4
  %427 = insertelement <8 x i32> %426, i32 %421, i32 5
  %428 = insertelement <8 x i32> %427, i32 undef, i32 6
  %429 = insertelement <8 x i32> %428, i32 undef, i32 7
  %430 = call <4 x float> @llvm.SI.sampled.v8i32(<8 x i32> %429, <32 x i8> %86, <16 x i8> %88, i32 2)
  %431 = extractelement <4 x float> %430, i32 0
  %432 = extractelement <4 x float> %430, i32 1
  %433 = extractelement <4 x float> %430, i32 2
  %434 = fmul float %48, %411
  %435 = fmul float %49, %411
  %436 = fmul float %50, %411
  %437 = fmul float %51, %413
  %438 = fadd float %437, %434
  %439 = fmul float %52, %413
  %440 = fadd float %439, %435
  %441 = fmul float %53, %413
  %442 = fadd float %441, %436
  %443 = fmul float %54, %415
  %444 = fadd float %443, %438
  %445 = fmul float %55, %415
  %446 = fadd float %445, %440
  %447 = fmul float %56, %415
  %448 = fadd float %447, %442
  %449 = insertelement <4 x float> undef, float %444, i32 0
  %450 = insertelement <4 x float> %449, float %446, i32 1
  %451 = insertelement <4 x float> %450, float %448, i32 2
  %452 = insertelement <4 x float> %451, float %195, i32 3
  %453 = call <4 x float> @llvm.AMDGPU.cube(<4 x float> %452)
  %454 = extractelement <4 x float> %453, i32 0
  %455 = extractelement <4 x float> %453, i32 1
  %456 = extractelement <4 x float> %453, i32 2
  %457 = extractelement <4 x float> %453, i32 3
  %458 = call float @fabs(float %456)
  %459 = fdiv float 1.000000e+00, %458
  %460 = fmul float %454, %459
  %461 = fadd float %460, 1.500000e+00
  %462 = fmul float %455, %459
  %463 = fadd float %462, 1.500000e+00
  %464 = bitcast float %463 to i32
  %465 = bitcast float %461 to i32
  %466 = bitcast float %457 to i32
  %467 = insertelement <4 x i32> undef, i32 %464, i32 0
  %468 = insertelement <4 x i32> %467, i32 %465, i32 1
  %469 = insertelement <4 x i32> %468, i32 %466, i32 2
  %470 = insertelement <4 x i32> %469, i32 undef, i32 3
  %471 = call <4 x float> @llvm.SI.sample.v4i32(<4 x i32> %470, <32 x i8> %90, <16 x i8> %92, i32 4)
  %472 = extractelement <4 x float> %471, i32 0
  %473 = extractelement <4 x float> %471, i32 1
  %474 = extractelement <4 x float> %471, i32 2
  %475 = fmul float %431, %472
  %476 = fadd float %475, %329
  %477 = fmul float %432, %473
  %478 = fadd float %477, %330
  %479 = fmul float %433, %474
  %480 = fadd float %479, %331
  %481 = fmul float %107, %107
  %482 = fmul float %108, %108
  %483 = fadd float %482, %481
  %484 = fmul float %109, %109
  %485 = fadd float %483, %484
  %486 = call float @llvm.AMDGPU.rsq(float %485)
  %487 = fmul float %107, %486
  %488 = fmul float %108, %486
  %489 = fmul float %109, %486
  %490 = fmul float %377, %40
  %491 = fmul float %378, %41
  %492 = fmul float %379, %42
  %493 = fmul float %359, %487
  %494 = fmul float %360, %488
  %495 = fadd float %494, %493
  %496 = fmul float %361, %489
  %497 = fadd float %495, %496
  %498 = fmul float %497, %359
  %499 = fmul float %497, %360
  %500 = fmul float %497, %361
  %501 = fmul float 2.000000e+00, %498
  %502 = fmul float 2.000000e+00, %499
  %503 = fmul float 2.000000e+00, %500
  %504 = fsub float -0.000000e+00, %501
  %505 = fadd float %487, %504
  %506 = fsub float -0.000000e+00, %502
  %507 = fadd float %488, %506
  %508 = fsub float -0.000000e+00, %503
  %509 = fadd float %489, %508
  %510 = fmul float %95, %95
  %511 = fmul float %96, %96
  %512 = fadd float %511, %510
  %513 = fmul float %97, %97
  %514 = fadd float %512, %513
  %515 = call float @llvm.AMDGPU.rsq(float %514)
  %516 = fmul float %95, %515
  %517 = fmul float %96, %515
  %518 = fmul float %97, %515
  %519 = fmul float %505, %516
  %520 = fmul float %507, %517
  %521 = fadd float %520, %519
  %522 = fmul float %509, %518
  %523 = fadd float %521, %522
  %524 = fsub float -0.000000e+00, %523
  %525 = fcmp uge float %524, 0.000000e+00
  %526 = select i1 %525, float %524, float 0.000000e+00
  %527 = fmul float %43, %380
  %528 = fadd float %527, 1.000000e+00
  %529 = call float @llvm.pow.f32(float %526, float %528)
  %530 = fmul float %476, %37
  %531 = fmul float %478, %38
  %532 = fmul float %480, %39
  %533 = fmul float %359, %487
  %534 = fmul float %360, %488
  %535 = fadd float %534, %533
  %536 = fmul float %361, %489
  %537 = fadd float %535, %536
  %538 = fcmp uge float %537, 0.000000e+00
  %539 = select i1 %538, float %537, float 0.000000e+00
  %540 = fmul float %530, %539
  %541 = fmul float %531, %539
  %542 = fmul float %532, %539
  %543 = fmul float %490, %529
  %544 = fadd float %543, %540
  %545 = fmul float %491, %529
  %546 = fadd float %545, %541
  %547 = fmul float %492, %529
  %548 = fadd float %547, %542
  %549 = fmul float %476, %34
  %550 = fmul float %478, %35
  %551 = fmul float %480, %36
  %552 = fmul float %544, %57
  %553 = fadd float %552, %549
  %554 = fmul float %546, %58
  %555 = fadd float %554, %550
  %556 = fmul float %548, %59
  %557 = fadd float %556, %551
  %558 = bitcast float %136 to i32
  %559 = bitcast float %182 to i32
  %560 = bitcast float %137 to i32
  %561 = bitcast float %183 to i32
  %562 = bitcast float %temp28.1 to i32
  %563 = bitcast float %temp29.1 to i32
  %564 = insertelement <8 x i32> undef, i32 %558, i32 0
  %565 = insertelement <8 x i32> %564, i32 %559, i32 1
  %566 = insertelement <8 x i32> %565, i32 %560, i32 2
  %567 = insertelement <8 x i32> %566, i32 %561, i32 3
  %568 = insertelement <8 x i32> %567, i32 %562, i32 4
  %569 = insertelement <8 x i32> %568, i32 %563, i32 5
  %570 = insertelement <8 x i32> %569, i32 undef, i32 6
  %571 = insertelement <8 x i32> %570, i32 undef, i32 7
  %572 = call <4 x float> @llvm.SI.sampled.v8i32(<8 x i32> %571, <32 x i8> %74, <16 x i8> %76, i32 2)
  %573 = extractelement <4 x float> %572, i32 0
  %574 = extractelement <4 x float> %572, i32 1
  %575 = extractelement <4 x float> %572, i32 2
  %576 = fmul float %573, %44
  %577 = fadd float %576, %553
  %578 = fmul float %574, %45
  %579 = fadd float %578, %555
  %580 = fmul float %575, %46
  %581 = fadd float %580, %557
  %582 = call i32 @llvm.SI.packf16(float %577, float %579)
  %583 = bitcast i32 %582 to float
  %584 = call i32 @llvm.SI.packf16(float %581, float %283)
  %585 = bitcast i32 %584 to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %583, float %585, float %583, float %585)
  ret void

ENDIF66:                                          ; preds = %LOOP65
  %586 = bitcast float %temp28.1 to i32
  %587 = bitcast float %temp29.1 to i32
  %588 = insertelement <8 x i32> %237, i32 %586, i32 4
  %589 = insertelement <8 x i32> %588, i32 %587, i32 5
  %590 = insertelement <8 x i32> %589, i32 undef, i32 6
  %591 = insertelement <8 x i32> %590, i32 undef, i32 7
  %592 = call <4 x float> @llvm.SI.sampled.v8i32(<8 x i32> %591, <32 x i8> %62, <16 x i8> %64, i32 2)
  %593 = extractelement <4 x float> %592, i32 3
  %594 = fcmp oge float %temp30.1, %593
  %595 = sext i1 %594 to i32
  %596 = bitcast i32 %595 to float
  %597 = bitcast float %596 to i32
  %598 = and i32 %597, 1065353216
  %599 = bitcast i32 %598 to float
  %600 = fmul float 5.000000e-01, %temp32.0
  %601 = fsub float -0.000000e+00, %600
  %602 = fmul float %599, %temp32.0
  %603 = fadd float %602, %601
  %604 = fmul float %214, %603
  %605 = fadd float %604, %temp28.1
  %606 = fmul float %215, %603
  %607 = fadd float %606, %temp29.1
  %608 = fmul float %216, %603
  %609 = fadd float %608, %temp30.1
  %610 = fadd float %temp24.1, 1.000000e+00
  %611 = fmul float %temp32.0, 5.000000e-01
  br label %LOOP65
}

; Function Attrs: nounwind readnone
declare float @llvm.SI.load.const(<16 x i8>, i32) #1

; Function Attrs: nounwind readnone
declare float @llvm.SI.fs.interp(i32, i32, i32, <2 x i32>) #1

; Function Attrs: readnone
declare i32 @llvm.SI.tid() #2

; Function Attrs: readonly
declare float @ceil(float) #3

; Function Attrs: readnone
declare float @llvm.AMDGPU.rsq(float) #2

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.SI.sampled.v8i32(<8 x i32>, <32 x i8>, <16 x i8>, i32) #1

; Function Attrs: readnone
declare <4 x float> @llvm.AMDGPU.cube(<4 x float>) #2

; Function Attrs: readnone
declare float @fabs(float) #2

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.SI.sample.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.pow.f32(float, float) #4

; Function Attrs: nounwind readnone
declare i32 @llvm.SI.packf16(float, float) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" }
attributes #1 = { nounwind readnone }
attributes #2 = { readnone }
attributes #3 = { readonly }
attributes #4 = { nounwind readonly }

!0 = metadata !{metadata !"const", null, i32 1}
