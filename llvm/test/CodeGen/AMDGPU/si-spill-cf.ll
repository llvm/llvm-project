; RUN: llc -march=amdgcn < %s -verify-machineinstrs | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s -verify-machineinstrs | FileCheck -check-prefix=SI %s

; If this occurs it is likely due to reordering and the restore was
; originally supposed to happen before SI_END_CF.

; SI: s_or_b64 exec, exec, [[SAVED:s\[[0-9]+:[0-9]+\]|[a-z]+]]
; SI-NOT: v_readlane_b32 [[SAVED]]
define amdgpu_ps void @main() #0 {
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
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %67 = icmp sgt i32 %tid, 4
  br i1 %67, label %ENDLOOP, label %ENDIF

ENDLOOP:                                          ; preds = %ELSE2566, %LOOP
  %one.sub.a.i = fsub float 1.000000e+00, %0
  %one.sub.ac.i = fmul float %one.sub.a.i, undef
  %result.i = fadd float fmul (float undef, float undef), %one.sub.ac.i
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float undef, float %result.i, float undef, float 1.000000e+00)
  ret void

ENDIF:                                            ; preds = %LOOP
  %68 = fsub float %2, undef
  %69 = fsub float %3, undef
  %70 = fsub float %4, undef
  %71 = fmul float %68, 0.000000e+00
  %72 = fmul float %69, undef
  %73 = fmul float %70, undef
  %74 = fsub float %6, undef
  %75 = fsub float %7, undef
  %76 = fmul float %74, undef
  %77 = fmul float %75, 0.000000e+00
  %78 = call float @llvm.minnum.f32(float %73, float %77)
  %79 = call float @llvm.maxnum.f32(float %71, float 0.000000e+00)
  %80 = call float @llvm.maxnum.f32(float %72, float %76)
  %81 = call float @llvm.maxnum.f32(float undef, float %78)
  %82 = call float @llvm.minnum.f32(float %79, float %80)
  %83 = call float @llvm.minnum.f32(float %82, float undef)
  %84 = fsub float %14, undef
  %85 = fsub float %15, undef
  %86 = fsub float %16, undef
  %87 = fmul float %84, undef
  %88 = fmul float %85, undef
  %89 = fmul float %86, undef
  %90 = fsub float %17, undef
  %91 = fsub float %18, undef
  %92 = fsub float %19, undef
  %93 = fmul float %90, 0.000000e+00
  %94 = fmul float %91, undef
  %95 = fmul float %92, undef
  %96 = call float @llvm.minnum.f32(float %88, float %94)
  %97 = call float @llvm.maxnum.f32(float %87, float %93)
  %98 = call float @llvm.maxnum.f32(float %89, float %95)
  %99 = call float @llvm.maxnum.f32(float undef, float %96)
  %100 = call float @llvm.maxnum.f32(float %99, float undef)
  %101 = call float @llvm.minnum.f32(float %97, float undef)
  %102 = call float @llvm.minnum.f32(float %101, float %98)
  %103 = fsub float %30, undef
  %104 = fsub float %31, undef
  %105 = fmul float %103, 0.000000e+00
  %106 = fmul float %104, 0.000000e+00
  %107 = call float @llvm.minnum.f32(float undef, float %105)
  %108 = call float @llvm.maxnum.f32(float undef, float %106)
  %109 = call float @llvm.maxnum.f32(float undef, float %107)
  %110 = call float @llvm.maxnum.f32(float %109, float undef)
  %111 = call float @llvm.minnum.f32(float undef, float %108)
  %112 = fsub float %32, undef
  %113 = fsub float %33, undef
  %114 = fsub float %34, undef
  %115 = fmul float %112, 0.000000e+00
  %116 = fmul float %113, undef
  %117 = fmul float %114, undef
  %118 = fsub float %35, undef
  %119 = fsub float %36, undef
  %120 = fsub float %37, undef
  %121 = fmul float %118, undef
  %122 = fmul float %119, undef
  %123 = fmul float %120, undef
  %124 = call float @llvm.minnum.f32(float %115, float %121)
  %125 = call float @llvm.minnum.f32(float %116, float %122)
  %126 = call float @llvm.minnum.f32(float %117, float %123)
  %127 = call float @llvm.maxnum.f32(float %124, float %125)
  %128 = call float @llvm.maxnum.f32(float %127, float %126)
  %129 = fsub float %38, undef
  %130 = fsub float %39, undef
  %131 = fsub float %40, undef
  %132 = fmul float %129, 0.000000e+00
  %133 = fmul float %130, undef
  %134 = fmul float %131, undef
  %135 = fsub float %41, undef
  %136 = fsub float %42, undef
  %137 = fsub float %43, undef
  %138 = fmul float %135, undef
  %139 = fmul float %136, undef
  %140 = fmul float %137, undef
  %141 = call float @llvm.minnum.f32(float %132, float %138)
  %142 = call float @llvm.minnum.f32(float %133, float %139)
  %143 = call float @llvm.minnum.f32(float %134, float %140)
  %144 = call float @llvm.maxnum.f32(float %141, float %142)
  %145 = call float @llvm.maxnum.f32(float %144, float %143)
  %146 = fsub float %44, undef
  %147 = fsub float %45, undef
  %148 = fsub float %46, undef
  %149 = fmul float %146, 0.000000e+00
  %150 = fmul float %147, 0.000000e+00
  %151 = fmul float %148, undef
  %152 = fsub float %47, undef
  %153 = fsub float %48, undef
  %154 = fsub float %49, undef
  %155 = fmul float %152, undef
  %156 = fmul float %153, 0.000000e+00
  %157 = fmul float %154, undef
  %158 = call float @llvm.minnum.f32(float %149, float %155)
  %159 = call float @llvm.minnum.f32(float %150, float %156)
  %160 = call float @llvm.minnum.f32(float %151, float %157)
  %161 = call float @llvm.maxnum.f32(float %158, float %159)
  %162 = call float @llvm.maxnum.f32(float %161, float %160)
  %163 = fsub float %50, undef
  %164 = fsub float %51, undef
  %165 = fsub float %52, undef
  %166 = fmul float %163, undef
  %167 = fmul float %164, 0.000000e+00
  %168 = fmul float %165, 0.000000e+00
  %169 = fsub float %53, undef
  %170 = fsub float %54, undef
  %171 = fsub float %55, undef
  %172 = fdiv float 1.000000e+00, %temp18.0
  %173 = fmul float %169, undef
  %174 = fmul float %170, undef
  %175 = fmul float %171, %172
  %176 = call float @llvm.minnum.f32(float %166, float %173)
  %177 = call float @llvm.minnum.f32(float %167, float %174)
  %178 = call float @llvm.minnum.f32(float %168, float %175)
  %179 = call float @llvm.maxnum.f32(float %176, float %177)
  %180 = call float @llvm.maxnum.f32(float %179, float %178)
  %181 = fsub float %62, undef
  %182 = fsub float %63, undef
  %183 = fsub float %64, undef
  %184 = fmul float %181, 0.000000e+00
  %185 = fmul float %182, undef
  %186 = fmul float %183, undef
  %187 = fsub float %65, undef
  %188 = fsub float %66, undef
  %189 = fmul float %187, undef
  %190 = fmul float %188, undef
  %191 = call float @llvm.maxnum.f32(float %184, float %189)
  %192 = call float @llvm.maxnum.f32(float %185, float %190)
  %193 = call float @llvm.maxnum.f32(float %186, float undef)
  %194 = call float @llvm.minnum.f32(float %191, float %192)
  %195 = call float @llvm.minnum.f32(float %194, float %193)
  %.temp292.7 = select i1 undef, float %162, float undef
  %temp292.9 = select i1 false, float %180, float %.temp292.7
  %.temp292.9 = select i1 undef, float undef, float %temp292.9
  %196 = fcmp ogt float undef, 0.000000e+00
  %197 = fcmp olt float undef, %195
  %198 = and i1 %196, %197
  %199 = fcmp olt float undef, %.temp292.9
  %200 = and i1 %198, %199
  %temp292.11 = select i1 %200, float undef, float %.temp292.9
  %tid0 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #2
  %cmp0 = icmp eq i32 %tid0, 0
  br i1 %cmp0, label %IF2565, label %ELSE2566

IF2565:                                           ; preds = %ENDIF
  %tid1 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #2
  %cmp1 = icmp eq i32 %tid1, 0
  br i1 %cmp1, label %ENDIF2582, label %ELSE2584

ELSE2566:                                         ; preds = %ENDIF
  %tid2 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #2
  %tidf = bitcast i32 %tid2 to float
  %201 = fcmp oeq float %temp292.11, %tidf
  br i1 %201, label %ENDLOOP, label %ELSE2593

ENDIF2564:                                        ; preds = %ENDIF2594, %ENDIF2588
  %temp894.1 = phi float [ undef, %ENDIF2588 ], [ %temp894.2, %ENDIF2594 ]
  %temp18.1 = phi float [ %218, %ENDIF2588 ], [ undef, %ENDIF2594 ]
  %202 = fsub float %5, undef
  %203 = fmul float %202, undef
  %204 = call float @llvm.maxnum.f32(float undef, float %203)
  %205 = call float @llvm.minnum.f32(float %204, float undef)
  %206 = call float @llvm.minnum.f32(float %205, float undef)
  %207 = fcmp ogt float undef, 0.000000e+00
  %208 = fcmp olt float undef, 1.000000e+00
  %209 = and i1 %207, %208
  %tid3 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #2
  %tidf3 = bitcast i32 %tid3 to float
  %210 = fcmp olt float %tidf3, %206
  %211 = and i1 %209, %210
  br i1 %211, label %ENDIF2795, label %ELSE2797

ELSE2584:                                         ; preds = %IF2565
  br label %ENDIF2582

ENDIF2582:                                        ; preds = %ELSE2584, %IF2565
  %212 = fadd float %1, undef
  %213 = fadd float 0.000000e+00, %212
  %floor = call float @llvm.floor.f32(float %213)
  %214 = fsub float %213, %floor
  %tid4 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #2
  %cmp4 = icmp eq i32 %tid4, 0
  br i1 %cmp4, label %IF2589, label %ELSE2590

IF2589:                                           ; preds = %ENDIF2582
  br label %ENDIF2588

ELSE2590:                                         ; preds = %ENDIF2582
  br label %ENDIF2588

ENDIF2588:                                        ; preds = %ELSE2590, %IF2589
  %215 = fsub float 1.000000e+00, %214
  %216 = call float @llvm.sqrt.f32(float %215)
  %217 = fmul float %216, undef
  %218 = fadd float %217, undef
  br label %ENDIF2564

ELSE2593:                                         ; preds = %ELSE2566
  %219 = fcmp oeq float %temp292.11, %81
  %220 = fcmp olt float %81, %83
  %221 = and i1 %219, %220
  br i1 %221, label %ENDIF2594, label %ELSE2596

ELSE2596:                                         ; preds = %ELSE2593
  %222 = fcmp oeq float %temp292.11, %100
  %223 = fcmp olt float %100, %102
  %224 = and i1 %222, %223
  br i1 %224, label %ENDIF2594, label %ELSE2632

ENDIF2594:                                        ; preds = %ELSE2788, %ELSE2785, %ELSE2782, %ELSE2779, %IF2775, %ELSE2761, %ELSE2758, %IF2757, %ELSE2704, %ELSE2686, %ELSE2671, %ELSE2668, %IF2667, %ELSE2632, %ELSE2596, %ELSE2593
  %temp894.2 = phi float [ 0.000000e+00, %IF2667 ], [ 0.000000e+00, %ELSE2671 ], [ 0.000000e+00, %IF2757 ], [ 0.000000e+00, %ELSE2761 ], [ %temp894.0, %ELSE2758 ], [ 0.000000e+00, %IF2775 ], [ 0.000000e+00, %ELSE2779 ], [ 0.000000e+00, %ELSE2782 ], [ %.2848, %ELSE2788 ], [ 0.000000e+00, %ELSE2785 ], [ 0.000000e+00, %ELSE2593 ], [ 0.000000e+00, %ELSE2632 ], [ 0.000000e+00, %ELSE2704 ], [ 0.000000e+00, %ELSE2686 ], [ 0.000000e+00, %ELSE2668 ], [ 0.000000e+00, %ELSE2596 ]
  %225 = fmul float %temp894.2, undef
  br label %ENDIF2564

ELSE2632:                                         ; preds = %ELSE2596
  br i1 undef, label %ENDIF2594, label %ELSE2650

ELSE2650:                                         ; preds = %ELSE2632
  %226 = fcmp oeq float %temp292.11, %110
  %227 = fcmp olt float %110, %111
  %228 = and i1 %226, %227
  br i1 %228, label %IF2667, label %ELSE2668

IF2667:                                           ; preds = %ELSE2650
  br i1 undef, label %ENDIF2594, label %ELSE2671

ELSE2668:                                         ; preds = %ELSE2650
  %229 = fcmp oeq float %temp292.11, %128
  %230 = fcmp olt float %128, undef
  %231 = and i1 %229, %230
  br i1 %231, label %ENDIF2594, label %ELSE2686

ELSE2671:                                         ; preds = %IF2667
  br label %ENDIF2594

ELSE2686:                                         ; preds = %ELSE2668
  %232 = fcmp oeq float %temp292.11, %145
  %233 = fcmp olt float %145, undef
  %234 = and i1 %232, %233
  br i1 %234, label %ENDIF2594, label %ELSE2704

ELSE2704:                                         ; preds = %ELSE2686
  %235 = fcmp oeq float %temp292.11, %180
  %236 = fcmp olt float %180, undef
  %237 = and i1 %235, %236
  br i1 %237, label %ENDIF2594, label %ELSE2740

ELSE2740:                                         ; preds = %ELSE2704
  br i1 undef, label %IF2757, label %ELSE2758

IF2757:                                           ; preds = %ELSE2740
  br i1 undef, label %ENDIF2594, label %ELSE2761

ELSE2758:                                         ; preds = %ELSE2740
  br i1 undef, label %IF2775, label %ENDIF2594

ELSE2761:                                         ; preds = %IF2757
  br label %ENDIF2594

IF2775:                                           ; preds = %ELSE2758
  %238 = fcmp olt float undef, undef
  br i1 %238, label %ENDIF2594, label %ELSE2779

ELSE2779:                                         ; preds = %IF2775
  br i1 undef, label %ENDIF2594, label %ELSE2782

ELSE2782:                                         ; preds = %ELSE2779
  br i1 undef, label %ENDIF2594, label %ELSE2785

ELSE2785:                                         ; preds = %ELSE2782
  %239 = fcmp olt float undef, 0.000000e+00
  br i1 %239, label %ENDIF2594, label %ELSE2788

ELSE2788:                                         ; preds = %ELSE2785
  %240 = fcmp olt float 0.000000e+00, undef
  %.2848 = select i1 %240, float -1.000000e+00, float 1.000000e+00
  br label %ENDIF2594

ELSE2797:                                         ; preds = %ENDIF2564
  %241 = fsub float %8, undef
  %242 = fsub float %9, undef
  %243 = fsub float %10, undef
  %244 = fmul float %241, undef
  %245 = fmul float %242, undef
  %246 = fmul float %243, undef
  %247 = fsub float %11, undef
  %248 = fsub float %12, undef
  %249 = fsub float %13, undef
  %250 = fmul float %247, undef
  %251 = fmul float %248, undef
  %252 = fmul float %249, undef
  %253 = call float @llvm.minnum.f32(float %244, float %250)
  %254 = call float @llvm.minnum.f32(float %245, float %251)
  %255 = call float @llvm.maxnum.f32(float %246, float %252)
  %256 = call float @llvm.maxnum.f32(float %253, float %254)
  %257 = call float @llvm.maxnum.f32(float %256, float undef)
  %258 = call float @llvm.minnum.f32(float undef, float %255)
  %259 = fcmp ogt float %257, 0.000000e+00
  %260 = fcmp olt float %257, 1.000000e+00
  %261 = and i1 %259, %260
  %262 = fcmp olt float %257, %258
  %263 = and i1 %261, %262
  br i1 %263, label %ENDIF2795, label %ELSE2800

ENDIF2795:                                        ; preds = %ELSE2824, %ELSE2821, %ELSE2818, %ELSE2815, %ELSE2812, %ELSE2809, %ELSE2806, %ELSE2803, %ELSE2800, %ELSE2797, %ENDIF2564
  br label %LOOP

ELSE2800:                                         ; preds = %ELSE2797
  br i1 undef, label %ENDIF2795, label %ELSE2803

ELSE2803:                                         ; preds = %ELSE2800
  %264 = fsub float %20, undef
  %265 = fsub float %21, undef
  %266 = fsub float %22, undef
  %267 = fmul float %264, undef
  %268 = fmul float %265, undef
  %269 = fmul float %266, 0.000000e+00
  %270 = fsub float %23, undef
  %271 = fsub float %24, undef
  %272 = fsub float %25, undef
  %273 = fmul float %270, undef
  %274 = fmul float %271, undef
  %275 = fmul float %272, undef
  %276 = call float @llvm.minnum.f32(float %267, float %273)
  %277 = call float @llvm.maxnum.f32(float %268, float %274)
  %278 = call float @llvm.maxnum.f32(float %269, float %275)
  %279 = call float @llvm.maxnum.f32(float %276, float undef)
  %280 = call float @llvm.maxnum.f32(float %279, float undef)
  %281 = call float @llvm.minnum.f32(float undef, float %277)
  %282 = call float @llvm.minnum.f32(float %281, float %278)
  %283 = fcmp ogt float %280, 0.000000e+00
  %284 = fcmp olt float %280, 1.000000e+00
  %285 = and i1 %283, %284
  %286 = fcmp olt float %280, %282
  %287 = and i1 %285, %286
  br i1 %287, label %ENDIF2795, label %ELSE2806

ELSE2806:                                         ; preds = %ELSE2803
  %288 = fsub float %26, undef
  %289 = fsub float %27, undef
  %290 = fsub float %28, undef
  %291 = fmul float %288, undef
  %292 = fmul float %289, 0.000000e+00
  %293 = fmul float %290, undef
  %294 = fsub float %29, undef
  %295 = fmul float %294, undef
  %296 = call float @llvm.minnum.f32(float %291, float %295)
  %297 = call float @llvm.minnum.f32(float %292, float undef)
  %298 = call float @llvm.maxnum.f32(float %293, float undef)
  %299 = call float @llvm.maxnum.f32(float %296, float %297)
  %300 = call float @llvm.maxnum.f32(float %299, float undef)
  %301 = call float @llvm.minnum.f32(float undef, float %298)
  %302 = fcmp ogt float %300, 0.000000e+00
  %303 = fcmp olt float %300, 1.000000e+00
  %304 = and i1 %302, %303
  %305 = fcmp olt float %300, %301
  %306 = and i1 %304, %305
  br i1 %306, label %ENDIF2795, label %ELSE2809

ELSE2809:                                         ; preds = %ELSE2806
  br i1 undef, label %ENDIF2795, label %ELSE2812

ELSE2812:                                         ; preds = %ELSE2809
  br i1 undef, label %ENDIF2795, label %ELSE2815

ELSE2815:                                         ; preds = %ELSE2812
  br i1 undef, label %ENDIF2795, label %ELSE2818

ELSE2818:                                         ; preds = %ELSE2815
  br i1 undef, label %ENDIF2795, label %ELSE2821

ELSE2821:                                         ; preds = %ELSE2818
  %307 = fsub float %56, undef
  %308 = fsub float %57, undef
  %309 = fsub float %58, undef
  %310 = fmul float %307, undef
  %311 = fmul float %308, 0.000000e+00
  %312 = fmul float %309, undef
  %313 = fsub float %59, undef
  %314 = fsub float %60, undef
  %315 = fsub float %61, undef
  %316 = fmul float %313, undef
  %317 = fmul float %314, undef
  %318 = fmul float %315, undef
  %319 = call float @llvm.maxnum.f32(float %310, float %316)
  %320 = call float @llvm.maxnum.f32(float %311, float %317)
  %321 = call float @llvm.maxnum.f32(float %312, float %318)
  %322 = call float @llvm.minnum.f32(float %319, float %320)
  %323 = call float @llvm.minnum.f32(float %322, float %321)
  %324 = fcmp ogt float undef, 0.000000e+00
  %325 = fcmp olt float undef, 1.000000e+00
  %326 = and i1 %324, %325
  %327 = fcmp olt float undef, %323
  %328 = and i1 %326, %327
  br i1 %328, label %ENDIF2795, label %ELSE2824

ELSE2824:                                         ; preds = %ELSE2821
  %.2849 = select i1 undef, float 0.000000e+00, float 1.000000e+00
  br label %ENDIF2795
}

declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #1

; Function Attrs: nounwind readnone
declare float @llvm.SI.load.const(<16 x i8>, i32) #1

; Function Attrs: nounwind readnone
declare float @llvm.floor.f32(float) #1

; Function Attrs: nounwind readnone
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: nounwind readnone
declare float @llvm.minnum.f32(float, float) #1

; Function Attrs: nounwind readnone
declare float @llvm.maxnum.f32(float, float) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
