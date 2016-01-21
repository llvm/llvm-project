; RUN: llc -march=amdgcn -mcpu=SI < %s -verify-machineinstrs | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s -verify-machineinstrs | FileCheck -check-prefix=SI %s

; If this occurs it is likely due to reordering and the restore was
; originally supposed to happen before SI_END_CF.
; SI: s_or_b64 exec, exec, [[SAVED:s\[[0-9]+:[0-9]+\]|[a-z]+]]
; SI-NOT: v_readlane_b32 [[SAVED]]

define void @main() #0 {
main_body:
  %0 = call float @llvm.SI.load.const(<16 x i8> undef, i32 16)
  %1 = call float @llvm.SI.load.const(<16 x i8> undef, i32 32)
  %2 = call float @llvm.SI.load.const(<16 x i8> undef, i32 80)
  %3 = call float @llvm.SI.load.const(<16 x i8> undef, i32 84)
  %4 = call float @llvm.SI.load.const(<16 x i8> undef, i32 88)
  %5 = call float @llvm.SI.load.const(<16 x i8> undef, i32 96)
  %6 = call float @llvm.SI.load.const(<16 x i8> undef, i32 100)
  %7 = call float @llvm.SI.load.const(<16 x i8> undef, i32 104)
  %8 = call float @llvm.SI.load.const(<16 x i8> undef, i32 112)
  %9 = call float @llvm.SI.load.const(<16 x i8> undef, i32 116)
  %10 = call float @llvm.SI.load.const(<16 x i8> undef, i32 120)
  %11 = call float @llvm.SI.load.const(<16 x i8> undef, i32 128)
  %12 = call float @llvm.SI.load.const(<16 x i8> undef, i32 132)
  %13 = call float @llvm.SI.load.const(<16 x i8> undef, i32 136)
  %14 = call float @llvm.SI.load.const(<16 x i8> undef, i32 144)
  %15 = call float @llvm.SI.load.const(<16 x i8> undef, i32 148)
  %16 = call float @llvm.SI.load.const(<16 x i8> undef, i32 152)
  %17 = call float @llvm.SI.load.const(<16 x i8> undef, i32 160)
  %18 = call float @llvm.SI.load.const(<16 x i8> undef, i32 164)
  %19 = call float @llvm.SI.load.const(<16 x i8> undef, i32 168)
  %20 = call float @llvm.SI.load.const(<16 x i8> undef, i32 176)
  %21 = call float @llvm.SI.load.const(<16 x i8> undef, i32 180)
  %22 = call float @llvm.SI.load.const(<16 x i8> undef, i32 184)
  %23 = call float @llvm.SI.load.const(<16 x i8> undef, i32 192)
  %24 = call float @llvm.SI.load.const(<16 x i8> undef, i32 196)
  %25 = call float @llvm.SI.load.const(<16 x i8> undef, i32 200)
  %26 = call float @llvm.SI.load.const(<16 x i8> undef, i32 208)
  %27 = call float @llvm.SI.load.const(<16 x i8> undef, i32 212)
  %28 = call float @llvm.SI.load.const(<16 x i8> undef, i32 216)
  %29 = call float @llvm.SI.load.const(<16 x i8> undef, i32 224)
  %30 = call float @llvm.SI.load.const(<16 x i8> undef, i32 228)
  %31 = call float @llvm.SI.load.const(<16 x i8> undef, i32 232)
  %32 = call float @llvm.SI.load.const(<16 x i8> undef, i32 240)
  %33 = call float @llvm.SI.load.const(<16 x i8> undef, i32 244)
  %34 = call float @llvm.SI.load.const(<16 x i8> undef, i32 248)
  %35 = call float @llvm.SI.load.const(<16 x i8> undef, i32 256)
  %36 = call float @llvm.SI.load.const(<16 x i8> undef, i32 260)
  %37 = call float @llvm.SI.load.const(<16 x i8> undef, i32 264)
  %38 = call float @llvm.SI.load.const(<16 x i8> undef, i32 272)
  %39 = call float @llvm.SI.load.const(<16 x i8> undef, i32 276)
  %40 = call float @llvm.SI.load.const(<16 x i8> undef, i32 280)
  %41 = call float @llvm.SI.load.const(<16 x i8> undef, i32 288)
  %42 = call float @llvm.SI.load.const(<16 x i8> undef, i32 292)
  %43 = call float @llvm.SI.load.const(<16 x i8> undef, i32 296)
  %44 = call float @llvm.SI.load.const(<16 x i8> undef, i32 304)
  %45 = call float @llvm.SI.load.const(<16 x i8> undef, i32 308)
  %46 = call float @llvm.SI.load.const(<16 x i8> undef, i32 312)
  %47 = call float @llvm.SI.load.const(<16 x i8> undef, i32 320)
  %48 = call float @llvm.SI.load.const(<16 x i8> undef, i32 324)
  %49 = call float @llvm.SI.load.const(<16 x i8> undef, i32 328)
  %50 = call float @llvm.SI.load.const(<16 x i8> undef, i32 336)
  %51 = call float @llvm.SI.load.const(<16 x i8> undef, i32 340)
  %52 = call float @llvm.SI.load.const(<16 x i8> undef, i32 344)
  %53 = call float @llvm.SI.load.const(<16 x i8> undef, i32 352)
  %54 = call float @llvm.SI.load.const(<16 x i8> undef, i32 356)
  %55 = call float @llvm.SI.load.const(<16 x i8> undef, i32 360)
  %56 = call float @llvm.SI.load.const(<16 x i8> undef, i32 368)
  %57 = call float @llvm.SI.load.const(<16 x i8> undef, i32 372)
  %58 = call float @llvm.SI.load.const(<16 x i8> undef, i32 376)
  %59 = call float @llvm.SI.load.const(<16 x i8> undef, i32 384)
  %60 = call float @llvm.SI.load.const(<16 x i8> undef, i32 388)
  %61 = call float @llvm.SI.load.const(<16 x i8> undef, i32 392)
  %62 = call float @llvm.SI.load.const(<16 x i8> undef, i32 400)
  %63 = call float @llvm.SI.load.const(<16 x i8> undef, i32 404)
  %64 = call float @llvm.SI.load.const(<16 x i8> undef, i32 408)
  %65 = call float @llvm.SI.load.const(<16 x i8> undef, i32 416)
  %66 = call float @llvm.SI.load.const(<16 x i8> undef, i32 420)
  br label %LOOP

LOOP:                                             ; preds = %ENDIF2795, %main_body
  %temp894.0 = phi float [ 0.000000e+00, %main_body ], [ %temp894.1, %ENDIF2795 ]
  %temp18.0 = phi float [ undef, %main_body ], [ %temp18.1, %ENDIF2795 ]
  %67 = icmp sgt i32 undef, 4
  br i1 %67, label %ENDLOOP, label %ENDIF

ENDLOOP:                                          ; preds = %ELSE2566, %LOOP
  %68 = call float @llvm.AMDGPU.lrp(float %0, float undef, float undef)
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float undef, float %68, float undef, float 1.000000e+00)
  ret void

ENDIF:                                            ; preds = %LOOP
  %69 = fsub float %2, undef
  %70 = fsub float %3, undef
  %71 = fsub float %4, undef
  %72 = fmul float %69, 0.000000e+00
  %73 = fmul float %70, undef
  %74 = fmul float %71, undef
  %75 = fsub float %6, undef
  %76 = fsub float %7, undef
  %77 = fmul float %75, undef
  %78 = fmul float %76, 0.000000e+00
  %79 = call float @llvm.minnum.f32(float %74, float %78)
  %80 = call float @llvm.maxnum.f32(float %72, float 0.000000e+00)
  %81 = call float @llvm.maxnum.f32(float %73, float %77)
  %82 = call float @llvm.maxnum.f32(float undef, float %79)
  %83 = call float @llvm.minnum.f32(float %80, float %81)
  %84 = call float @llvm.minnum.f32(float %83, float undef)
  %85 = fsub float %14, undef
  %86 = fsub float %15, undef
  %87 = fsub float %16, undef
  %88 = fmul float %85, undef
  %89 = fmul float %86, undef
  %90 = fmul float %87, undef
  %91 = fsub float %17, undef
  %92 = fsub float %18, undef
  %93 = fsub float %19, undef
  %94 = fmul float %91, 0.000000e+00
  %95 = fmul float %92, undef
  %96 = fmul float %93, undef
  %97 = call float @llvm.minnum.f32(float %89, float %95)
  %98 = call float @llvm.maxnum.f32(float %88, float %94)
  %99 = call float @llvm.maxnum.f32(float %90, float %96)
  %100 = call float @llvm.maxnum.f32(float undef, float %97)
  %101 = call float @llvm.maxnum.f32(float %100, float undef)
  %102 = call float @llvm.minnum.f32(float %98, float undef)
  %103 = call float @llvm.minnum.f32(float %102, float %99)
  %104 = fsub float %30, undef
  %105 = fsub float %31, undef
  %106 = fmul float %104, 0.000000e+00
  %107 = fmul float %105, 0.000000e+00
  %108 = call float @llvm.minnum.f32(float undef, float %106)
  %109 = call float @llvm.maxnum.f32(float undef, float %107)
  %110 = call float @llvm.maxnum.f32(float undef, float %108)
  %111 = call float @llvm.maxnum.f32(float %110, float undef)
  %112 = call float @llvm.minnum.f32(float undef, float %109)
  %113 = fsub float %32, undef
  %114 = fsub float %33, undef
  %115 = fsub float %34, undef
  %116 = fmul float %113, 0.000000e+00
  %117 = fmul float %114, undef
  %118 = fmul float %115, undef
  %119 = fsub float %35, undef
  %120 = fsub float %36, undef
  %121 = fsub float %37, undef
  %122 = fmul float %119, undef
  %123 = fmul float %120, undef
  %124 = fmul float %121, undef
  %125 = call float @llvm.minnum.f32(float %116, float %122)
  %126 = call float @llvm.minnum.f32(float %117, float %123)
  %127 = call float @llvm.minnum.f32(float %118, float %124)
  %128 = call float @llvm.maxnum.f32(float %125, float %126)
  %129 = call float @llvm.maxnum.f32(float %128, float %127)
  %130 = fsub float %38, undef
  %131 = fsub float %39, undef
  %132 = fsub float %40, undef
  %133 = fmul float %130, 0.000000e+00
  %134 = fmul float %131, undef
  %135 = fmul float %132, undef
  %136 = fsub float %41, undef
  %137 = fsub float %42, undef
  %138 = fsub float %43, undef
  %139 = fmul float %136, undef
  %140 = fmul float %137, undef
  %141 = fmul float %138, undef
  %142 = call float @llvm.minnum.f32(float %133, float %139)
  %143 = call float @llvm.minnum.f32(float %134, float %140)
  %144 = call float @llvm.minnum.f32(float %135, float %141)
  %145 = call float @llvm.maxnum.f32(float %142, float %143)
  %146 = call float @llvm.maxnum.f32(float %145, float %144)
  %147 = fsub float %44, undef
  %148 = fsub float %45, undef
  %149 = fsub float %46, undef
  %150 = fmul float %147, 0.000000e+00
  %151 = fmul float %148, 0.000000e+00
  %152 = fmul float %149, undef
  %153 = fsub float %47, undef
  %154 = fsub float %48, undef
  %155 = fsub float %49, undef
  %156 = fmul float %153, undef
  %157 = fmul float %154, 0.000000e+00
  %158 = fmul float %155, undef
  %159 = call float @llvm.minnum.f32(float %150, float %156)
  %160 = call float @llvm.minnum.f32(float %151, float %157)
  %161 = call float @llvm.minnum.f32(float %152, float %158)
  %162 = call float @llvm.maxnum.f32(float %159, float %160)
  %163 = call float @llvm.maxnum.f32(float %162, float %161)
  %164 = fsub float %50, undef
  %165 = fsub float %51, undef
  %166 = fsub float %52, undef
  %167 = fmul float %164, undef
  %168 = fmul float %165, 0.000000e+00
  %169 = fmul float %166, 0.000000e+00
  %170 = fsub float %53, undef
  %171 = fsub float %54, undef
  %172 = fsub float %55, undef
  %173 = fdiv float 1.000000e+00, %temp18.0
  %174 = fmul float %170, undef
  %175 = fmul float %171, undef
  %176 = fmul float %172, %173
  %177 = call float @llvm.minnum.f32(float %167, float %174)
  %178 = call float @llvm.minnum.f32(float %168, float %175)
  %179 = call float @llvm.minnum.f32(float %169, float %176)
  %180 = call float @llvm.maxnum.f32(float %177, float %178)
  %181 = call float @llvm.maxnum.f32(float %180, float %179)
  %182 = fsub float %62, undef
  %183 = fsub float %63, undef
  %184 = fsub float %64, undef
  %185 = fmul float %182, 0.000000e+00
  %186 = fmul float %183, undef
  %187 = fmul float %184, undef
  %188 = fsub float %65, undef
  %189 = fsub float %66, undef
  %190 = fmul float %188, undef
  %191 = fmul float %189, undef
  %192 = call float @llvm.maxnum.f32(float %185, float %190)
  %193 = call float @llvm.maxnum.f32(float %186, float %191)
  %194 = call float @llvm.maxnum.f32(float %187, float undef)
  %195 = call float @llvm.minnum.f32(float %192, float %193)
  %196 = call float @llvm.minnum.f32(float %195, float %194)
  %.temp292.7 = select i1 undef, float %163, float undef
  %temp292.9 = select i1 false, float %181, float %.temp292.7
  %.temp292.9 = select i1 undef, float undef, float %temp292.9
  %197 = fcmp ogt float undef, 0.000000e+00
  %198 = fcmp olt float undef, %196
  %199 = and i1 %197, %198
  %200 = fcmp olt float undef, %.temp292.9
  %201 = and i1 %199, %200
  %temp292.11 = select i1 %201, float undef, float %.temp292.9
  br i1 undef, label %IF2565, label %ELSE2566

IF2565:                                           ; preds = %ENDIF
  br i1 false, label %ENDIF2582, label %ELSE2584

ELSE2566:                                         ; preds = %ENDIF
  %202 = fcmp oeq float %temp292.11, 1.000000e+04
  br i1 %202, label %ENDLOOP, label %ELSE2593

ENDIF2564:                                        ; preds = %ENDIF2594, %ENDIF2588
  %temp894.1 = phi float [ undef, %ENDIF2588 ], [ %temp894.2, %ENDIF2594 ]
  %temp18.1 = phi float [ %219, %ENDIF2588 ], [ undef, %ENDIF2594 ]
  %203 = fsub float %5, undef
  %204 = fmul float %203, undef
  %205 = call float @llvm.maxnum.f32(float undef, float %204)
  %206 = call float @llvm.minnum.f32(float %205, float undef)
  %207 = call float @llvm.minnum.f32(float %206, float undef)
  %208 = fcmp ogt float undef, 0.000000e+00
  %209 = fcmp olt float undef, 1.000000e+00
  %210 = and i1 %208, %209
  %211 = fcmp olt float undef, %207
  %212 = and i1 %210, %211
  br i1 %212, label %ENDIF2795, label %ELSE2797

ELSE2584:                                         ; preds = %IF2565
  br label %ENDIF2582

ENDIF2582:                                        ; preds = %ELSE2584, %IF2565
  %213 = fadd float %1, undef
  %214 = fadd float 0.000000e+00, %213
  %215 = call float @llvm.AMDGPU.fract.f32(float %214)
  br i1 undef, label %IF2589, label %ELSE2590

IF2589:                                           ; preds = %ENDIF2582
  br label %ENDIF2588

ELSE2590:                                         ; preds = %ENDIF2582
  br label %ENDIF2588

ENDIF2588:                                        ; preds = %ELSE2590, %IF2589
  %216 = fsub float 1.000000e+00, %215
  %217 = call float @llvm.sqrt.f32(float %216)
  %218 = fmul float %217, undef
  %219 = fadd float %218, undef
  br label %ENDIF2564

ELSE2593:                                         ; preds = %ELSE2566
  %220 = fcmp oeq float %temp292.11, %82
  %221 = fcmp olt float %82, %84
  %222 = and i1 %220, %221
  br i1 %222, label %ENDIF2594, label %ELSE2596

ELSE2596:                                         ; preds = %ELSE2593
  %223 = fcmp oeq float %temp292.11, %101
  %224 = fcmp olt float %101, %103
  %225 = and i1 %223, %224
  br i1 %225, label %ENDIF2594, label %ELSE2632

ENDIF2594:                                        ; preds = %ELSE2788, %ELSE2785, %ELSE2782, %ELSE2779, %IF2775, %ELSE2761, %ELSE2758, %IF2757, %ELSE2704, %ELSE2686, %ELSE2671, %ELSE2668, %IF2667, %ELSE2632, %ELSE2596, %ELSE2593
  %temp894.2 = phi float [ 0.000000e+00, %IF2667 ], [ 0.000000e+00, %ELSE2671 ], [ 0.000000e+00, %IF2757 ], [ 0.000000e+00, %ELSE2761 ], [ %temp894.0, %ELSE2758 ], [ 0.000000e+00, %IF2775 ], [ 0.000000e+00, %ELSE2779 ], [ 0.000000e+00, %ELSE2782 ], [ %.2848, %ELSE2788 ], [ 0.000000e+00, %ELSE2785 ], [ 0.000000e+00, %ELSE2593 ], [ 0.000000e+00, %ELSE2632 ], [ 0.000000e+00, %ELSE2704 ], [ 0.000000e+00, %ELSE2686 ], [ 0.000000e+00, %ELSE2668 ], [ 0.000000e+00, %ELSE2596 ]
  %226 = fmul float %temp894.2, undef
  br label %ENDIF2564

ELSE2632:                                         ; preds = %ELSE2596
  br i1 undef, label %ENDIF2594, label %ELSE2650

ELSE2650:                                         ; preds = %ELSE2632
  %227 = fcmp oeq float %temp292.11, %111
  %228 = fcmp olt float %111, %112
  %229 = and i1 %227, %228
  br i1 %229, label %IF2667, label %ELSE2668

IF2667:                                           ; preds = %ELSE2650
  br i1 undef, label %ENDIF2594, label %ELSE2671

ELSE2668:                                         ; preds = %ELSE2650
  %230 = fcmp oeq float %temp292.11, %129
  %231 = fcmp olt float %129, undef
  %232 = and i1 %230, %231
  br i1 %232, label %ENDIF2594, label %ELSE2686

ELSE2671:                                         ; preds = %IF2667
  br label %ENDIF2594

ELSE2686:                                         ; preds = %ELSE2668
  %233 = fcmp oeq float %temp292.11, %146
  %234 = fcmp olt float %146, undef
  %235 = and i1 %233, %234
  br i1 %235, label %ENDIF2594, label %ELSE2704

ELSE2704:                                         ; preds = %ELSE2686
  %236 = fcmp oeq float %temp292.11, %181
  %237 = fcmp olt float %181, undef
  %238 = and i1 %236, %237
  br i1 %238, label %ENDIF2594, label %ELSE2740

ELSE2740:                                         ; preds = %ELSE2704
  br i1 undef, label %IF2757, label %ELSE2758

IF2757:                                           ; preds = %ELSE2740
  br i1 undef, label %ENDIF2594, label %ELSE2761

ELSE2758:                                         ; preds = %ELSE2740
  br i1 undef, label %IF2775, label %ENDIF2594

ELSE2761:                                         ; preds = %IF2757
  br label %ENDIF2594

IF2775:                                           ; preds = %ELSE2758
  %239 = fcmp olt float undef, undef
  br i1 %239, label %ENDIF2594, label %ELSE2779

ELSE2779:                                         ; preds = %IF2775
  br i1 undef, label %ENDIF2594, label %ELSE2782

ELSE2782:                                         ; preds = %ELSE2779
  br i1 undef, label %ENDIF2594, label %ELSE2785

ELSE2785:                                         ; preds = %ELSE2782
  %240 = fcmp olt float undef, 0.000000e+00
  br i1 %240, label %ENDIF2594, label %ELSE2788

ELSE2788:                                         ; preds = %ELSE2785
  %241 = fcmp olt float 0.000000e+00, undef
  %.2848 = select i1 %241, float -1.000000e+00, float 1.000000e+00
  br label %ENDIF2594

ELSE2797:                                         ; preds = %ENDIF2564
  %242 = fsub float %8, undef
  %243 = fsub float %9, undef
  %244 = fsub float %10, undef
  %245 = fmul float %242, undef
  %246 = fmul float %243, undef
  %247 = fmul float %244, undef
  %248 = fsub float %11, undef
  %249 = fsub float %12, undef
  %250 = fsub float %13, undef
  %251 = fmul float %248, undef
  %252 = fmul float %249, undef
  %253 = fmul float %250, undef
  %254 = call float @llvm.minnum.f32(float %245, float %251)
  %255 = call float @llvm.minnum.f32(float %246, float %252)
  %256 = call float @llvm.maxnum.f32(float %247, float %253)
  %257 = call float @llvm.maxnum.f32(float %254, float %255)
  %258 = call float @llvm.maxnum.f32(float %257, float undef)
  %259 = call float @llvm.minnum.f32(float undef, float %256)
  %260 = fcmp ogt float %258, 0.000000e+00
  %261 = fcmp olt float %258, 1.000000e+00
  %262 = and i1 %260, %261
  %263 = fcmp olt float %258, %259
  %264 = and i1 %262, %263
  br i1 %264, label %ENDIF2795, label %ELSE2800

ENDIF2795:                                        ; preds = %ELSE2824, %ELSE2821, %ELSE2818, %ELSE2815, %ELSE2812, %ELSE2809, %ELSE2806, %ELSE2803, %ELSE2800, %ELSE2797, %ENDIF2564
  br label %LOOP

ELSE2800:                                         ; preds = %ELSE2797
  br i1 undef, label %ENDIF2795, label %ELSE2803

ELSE2803:                                         ; preds = %ELSE2800
  %265 = fsub float %20, undef
  %266 = fsub float %21, undef
  %267 = fsub float %22, undef
  %268 = fmul float %265, undef
  %269 = fmul float %266, undef
  %270 = fmul float %267, 0.000000e+00
  %271 = fsub float %23, undef
  %272 = fsub float %24, undef
  %273 = fsub float %25, undef
  %274 = fmul float %271, undef
  %275 = fmul float %272, undef
  %276 = fmul float %273, undef
  %277 = call float @llvm.minnum.f32(float %268, float %274)
  %278 = call float @llvm.maxnum.f32(float %269, float %275)
  %279 = call float @llvm.maxnum.f32(float %270, float %276)
  %280 = call float @llvm.maxnum.f32(float %277, float undef)
  %281 = call float @llvm.maxnum.f32(float %280, float undef)
  %282 = call float @llvm.minnum.f32(float undef, float %278)
  %283 = call float @llvm.minnum.f32(float %282, float %279)
  %284 = fcmp ogt float %281, 0.000000e+00
  %285 = fcmp olt float %281, 1.000000e+00
  %286 = and i1 %284, %285
  %287 = fcmp olt float %281, %283
  %288 = and i1 %286, %287
  br i1 %288, label %ENDIF2795, label %ELSE2806

ELSE2806:                                         ; preds = %ELSE2803
  %289 = fsub float %26, undef
  %290 = fsub float %27, undef
  %291 = fsub float %28, undef
  %292 = fmul float %289, undef
  %293 = fmul float %290, 0.000000e+00
  %294 = fmul float %291, undef
  %295 = fsub float %29, undef
  %296 = fmul float %295, undef
  %297 = call float @llvm.minnum.f32(float %292, float %296)
  %298 = call float @llvm.minnum.f32(float %293, float undef)
  %299 = call float @llvm.maxnum.f32(float %294, float undef)
  %300 = call float @llvm.maxnum.f32(float %297, float %298)
  %301 = call float @llvm.maxnum.f32(float %300, float undef)
  %302 = call float @llvm.minnum.f32(float undef, float %299)
  %303 = fcmp ogt float %301, 0.000000e+00
  %304 = fcmp olt float %301, 1.000000e+00
  %305 = and i1 %303, %304
  %306 = fcmp olt float %301, %302
  %307 = and i1 %305, %306
  br i1 %307, label %ENDIF2795, label %ELSE2809

ELSE2809:                                         ; preds = %ELSE2806
  br i1 undef, label %ENDIF2795, label %ELSE2812

ELSE2812:                                         ; preds = %ELSE2809
  br i1 undef, label %ENDIF2795, label %ELSE2815

ELSE2815:                                         ; preds = %ELSE2812
  br i1 undef, label %ENDIF2795, label %ELSE2818

ELSE2818:                                         ; preds = %ELSE2815
  br i1 undef, label %ENDIF2795, label %ELSE2821

ELSE2821:                                         ; preds = %ELSE2818
  %308 = fsub float %56, undef
  %309 = fsub float %57, undef
  %310 = fsub float %58, undef
  %311 = fmul float %308, undef
  %312 = fmul float %309, 0.000000e+00
  %313 = fmul float %310, undef
  %314 = fsub float %59, undef
  %315 = fsub float %60, undef
  %316 = fsub float %61, undef
  %317 = fmul float %314, undef
  %318 = fmul float %315, undef
  %319 = fmul float %316, undef
  %320 = call float @llvm.maxnum.f32(float %311, float %317)
  %321 = call float @llvm.maxnum.f32(float %312, float %318)
  %322 = call float @llvm.maxnum.f32(float %313, float %319)
  %323 = call float @llvm.minnum.f32(float %320, float %321)
  %324 = call float @llvm.minnum.f32(float %323, float %322)
  %325 = fcmp ogt float undef, 0.000000e+00
  %326 = fcmp olt float undef, 1.000000e+00
  %327 = and i1 %325, %326
  %328 = fcmp olt float undef, %324
  %329 = and i1 %327, %328
  br i1 %329, label %ENDIF2795, label %ELSE2824

ELSE2824:                                         ; preds = %ELSE2821
  %.2849 = select i1 undef, float 0.000000e+00, float 1.000000e+00
  br label %ENDIF2795
}

; Function Attrs: nounwind readnone
declare float @llvm.SI.load.const(<16 x i8>, i32) #1

; Function Attrs: nounwind readnone
declare float @llvm.AMDGPU.fract.f32(float) #1

; Function Attrs: nounwind readnone
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: nounwind readnone
declare float @llvm.minnum.f32(float, float) #1

; Function Attrs: nounwind readnone
declare float @llvm.maxnum.f32(float, float) #1

; Function Attrs: readnone
declare float @llvm.AMDGPU.lrp(float, float, float) #2

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" "enable-no-nans-fp-math"="true" }
attributes #1 = { nounwind readnone }
attributes #2 = { readnone }
