; RUN: llc -march=amdgcn -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
;
; This test was causing a "Use not jointly dominated by defs" when
; removePartialRedundancy in RegisterCoalescing.cpp was calling extendToIndices
; without passing in an Undefs vector.

; GCN-LABEL: _amdgpu_ps_main

define dllexport amdgpu_ps void @_amdgpu_ps_main() local_unnamed_addr {
.entry:
  br i1 undef, label %bb77, label %bb

bb:                                               ; preds = %.entry
  %tmp = call <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32 15, i32 undef, i32 undef, <8 x i32> undef, i32 0, i32 0)
  %tmp1 = extractelement <4 x float> %tmp, i32 2
  br i1 undef, label %bb3, label %bb2

bb2:                                              ; preds = %bb
  br label %bb78

bb3:                                              ; preds = %bb
  %tmp4 = tail call float @llvm.floor.f32(float undef)
  %tmp5 = bitcast float %tmp4 to i32
  %tmp6 = fsub reassoc nnan arcp contract float 1.000000e+00, %tmp1
  %tmp7 = bitcast float %tmp6 to i32
  %tmp8 = insertelement <3 x i32> undef, i32 %tmp7, i32 0
  %tmp9 = shufflevector <3 x i32> %tmp8, <3 x i32> undef, <3 x i32> zeroinitializer
  %tmp10 = bitcast <3 x i32> %tmp9 to <3 x float>
  %tmp11 = fmul reassoc nnan arcp contract <3 x float> zeroinitializer, %tmp10
  %tmp12 = bitcast <3 x float> %tmp11 to <3 x i32>
  %tmp13 = shufflevector <3 x i32> %tmp12, <3 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  br i1 undef, label %bb15, label %bb14

bb14:                                             ; preds = %bb3
  br label %bb19

bb15:                                             ; preds = %bb3
  %tmp16 = insertelement <4 x i32> undef, i32 %tmp5, i32 1
  %tmp17 = insertelement <4 x i32> %tmp16, i32 undef, i32 2
  %tmp18 = insertelement <4 x i32> %tmp17, i32 undef, i32 3
  br label %bb19

bb19:                                             ; preds = %bb15, %bb14
  %__llpc_global_proxy_r3.0 = phi <4 x i32> [ undef, %bb14 ], [ %tmp18, %bb15 ]
  br i1 undef, label %bb41, label %bb20

bb20:                                             ; preds = %bb19
  br i1 undef, label %bb21, label %bb22

bb21:                                             ; preds = %bb20
  br label %bb23

bb22:                                             ; preds = %bb20
  br label %bb23

bb23:                                             ; preds = %bb22, %bb21
  br i1 undef, label %.lr.ph2363, label %._crit_edge2364

.lr.ph2363:                                       ; preds = %bb23
  %tmp24 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> undef, i32 2708, i32 0)
  br label %bb25

bb25:                                             ; preds = %bb25, %.lr.ph2363
  %__llpc_global_proxy_r9.02353 = phi <4 x i32> [ %tmp13, %.lr.ph2363 ], [ %__llpc_global_proxy_r9.12.vec.insert1367, %bb25 ]
  %__llpc_global_proxy_r5.12351 = phi <4 x i32> [ undef, %.lr.ph2363 ], [ %__llpc_global_proxy_r5.12.vec.insert1086, %bb25 ]
  %tmp26 = fadd reassoc nnan arcp contract float 0.000000e+00, 1.000000e+00
  %tmp27 = fmul reassoc nnan arcp contract float %tmp26, %tmp26
  %tmp28 = fmul reassoc nnan arcp contract float %tmp27, 0x400921FB60000000
  %bc2203 = bitcast <4 x i32> %__llpc_global_proxy_r5.12351 to <4 x float>
  %tmp29 = extractelement <4 x float> %bc2203, i32 0
  %tmp30 = fmul reassoc nnan arcp contract float %tmp29, 0.000000e+00
  %tmp31 = extractelement <4 x float> %bc2203, i32 2
  %tmp32 = fadd reassoc nnan arcp contract float %tmp31, %tmp30
  %tmp33 = fdiv float 1.000000e+00, %tmp32
  %tmp34 = fmul reassoc nnan arcp contract float %tmp33, 0.000000e+00
  %tmp35 = fmul reassoc nnan arcp contract float %tmp34, 2.500000e-01
  %tmp36 = fmul reassoc nnan arcp contract float %tmp35, 0.000000e+00
  %tmp37 = bitcast float %tmp36 to i32
  %__llpc_global_proxy_r5.12.vec.insert1086 = insertelement <4 x i32> %__llpc_global_proxy_r5.12351, i32 undef, i32 3
  %__llpc_global_proxy_r9.12.vec.insert1367 = insertelement <4 x i32> %__llpc_global_proxy_r9.02353, i32 %tmp37, i32 3
  %tmp38 = bitcast float %tmp33 to i32
  %__llpc_global_proxy_r11.12.vec.insert1740 = insertelement <4 x i32> undef, i32 %tmp38, i32 3
  %tmp39 = icmp ult i32 0, %tmp24
  br i1 %tmp39, label %bb25, label %._crit_edge2364

._crit_edge2364:                                  ; preds = %bb25, %bb23
  %__llpc_global_proxy_r9.0.lcssa = phi <4 x i32> [ %tmp13, %bb23 ], [ %__llpc_global_proxy_r9.12.vec.insert1367, %bb25 ]
  %__llpc_global_proxy_r11.0.lcssa = phi <4 x i32> [ undef, %bb23 ], [ %__llpc_global_proxy_r11.12.vec.insert1740, %bb25 ]
  %tmp40 = shufflevector <4 x i32> %__llpc_global_proxy_r3.0, <4 x i32> undef, <4 x i32> <i32 4, i32 1, i32 5, i32 6>
  br label %bb41

bb41:                                             ; preds = %._crit_edge2364, %bb19
  %__llpc_global_proxy_r3.1 = phi <4 x i32> [ %tmp40, %._crit_edge2364 ], [ undef, %bb19 ]
  %__llpc_global_proxy_r9.1 = phi <4 x i32> [ %__llpc_global_proxy_r9.0.lcssa, %._crit_edge2364 ], [ %tmp13, %bb19 ]
  %__llpc_global_proxy_r11.1 = phi <4 x i32> [ %__llpc_global_proxy_r11.0.lcssa, %._crit_edge2364 ], [ undef, %bb19 ]
  br i1 undef, label %bb52, label %bb42

bb42:                                             ; preds = %bb41
  %bc2199 = bitcast <4 x i32> %__llpc_global_proxy_r3.1 to <4 x float>
  %tmp43 = extractelement <4 x float> %bc2199, i32 1
  %tmp44 = fmul reassoc nnan arcp contract float %tmp43, 1.250000e-01
  br i1 undef, label %.lr.ph2337.preheader, label %._crit_edge2338

.lr.ph2337.preheader:                             ; preds = %bb42
  br label %.lr.ph2337

.lr.ph2337:                                       ; preds = %._crit_edge2318, %.lr.ph2337.preheader
  br i1 undef, label %.lr.ph2317, label %._crit_edge2318

.lr.ph2317:                                       ; preds = %.lr.ph2337
  br label %bb45

bb45:                                             ; preds = %bb51, %.lr.ph2317
  br i1 undef, label %bb51, label %bb46

bb46:                                             ; preds = %bb45
  br i1 undef, label %bb51, label %bb47

bb47:                                             ; preds = %bb46
  br i1 undef, label %bb49, label %bb48

bb48:                                             ; preds = %bb47
  br label %bb50

bb49:                                             ; preds = %bb47
  br label %bb50

bb50:                                             ; preds = %bb49, %bb48
  br label %bb51

bb51:                                             ; preds = %bb50, %bb46, %bb45
  br i1 undef, label %bb45, label %._crit_edge2318

._crit_edge2318:                                  ; preds = %bb51, %.lr.ph2337
  br i1 undef, label %.lr.ph2337, label %._crit_edge2338

._crit_edge2338:                                  ; preds = %._crit_edge2318, %bb42
  br label %bb52

bb52:                                             ; preds = %._crit_edge2338, %bb41
  %__llpc_global_proxy_r9.3 = phi <4 x i32> [ undef, %._crit_edge2338 ], [ %__llpc_global_proxy_r9.1, %bb41 ]
  %__llpc_global_proxy_r11.8 = phi <4 x i32> [ undef, %._crit_edge2338 ], [ %__llpc_global_proxy_r11.1, %bb41 ]
  br i1 undef, label %bb60, label %bb53

bb53:                                             ; preds = %bb52
  br i1 undef, label %.lr.ph2297.preheader, label %._crit_edge2298

.lr.ph2297.preheader:                             ; preds = %bb53
  br label %.lr.ph2297

.lr.ph2297:                                       ; preds = %._crit_edge2270, %.lr.ph2297.preheader
  %__llpc_global_proxy_r9.42285 = phi <4 x i32> [ %__llpc_global_proxy_r9.12.vec.insert1355, %._crit_edge2270 ], [ %__llpc_global_proxy_r9.3, %.lr.ph2297.preheader ]
  %__llpc_global_proxy_r9.12.vec.insert1355 = insertelement <4 x i32> %__llpc_global_proxy_r9.42285, i32 0, i32 3
  br i1 undef, label %._crit_edge2270, label %.lr.ph2269

.lr.ph2269:                                       ; preds = %.lr.ph2297
  br label %bb54

bb54:                                             ; preds = %bb59, %.lr.ph2269
  br i1 undef, label %bb56, label %bb55

bb55:                                             ; preds = %bb54
  br label %bb59

bb56:                                             ; preds = %bb54
  br i1 undef, label %bb58, label %bb57

bb57:                                             ; preds = %bb56
  br label %bb59

bb58:                                             ; preds = %bb56
  br label %bb59

bb59:                                             ; preds = %bb58, %bb57, %bb55
  br i1 undef, label %._crit_edge2270, label %bb54

._crit_edge2270:                                  ; preds = %bb59, %.lr.ph2297
  br i1 undef, label %.lr.ph2297, label %._crit_edge2298

._crit_edge2298:                                  ; preds = %._crit_edge2270, %bb53
  %__llpc_global_proxy_r9.4.lcssa = phi <4 x i32> [ %__llpc_global_proxy_r9.3, %bb53 ], [ %__llpc_global_proxy_r9.12.vec.insert1355, %._crit_edge2270 ]
  br label %bb60

bb60:                                             ; preds = %._crit_edge2298, %bb52
  %__llpc_global_proxy_r9.5 = phi <4 x i32> [ %__llpc_global_proxy_r9.4.lcssa, %._crit_edge2298 ], [ %__llpc_global_proxy_r9.3, %bb52 ]
  br i1 undef, label %bb76, label %bb61

bb61:                                             ; preds = %bb60
  %tmp62 = select i1 false, i32 undef, i32 128
  %tmp63 = add nuw nsw i32 %tmp62, 31
  %tmp64 = lshr i32 %tmp63, 5
  br i1 undef, label %._crit_edge2258, label %.lr.ph2257.preheader

.lr.ph2257.preheader:                             ; preds = %bb61
  %tmp65 = shufflevector <4 x i32> %__llpc_global_proxy_r9.5, <4 x i32> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %tmp66 = bitcast <3 x i32> %tmp65 to <3 x float>
  %tmp67 = fmul reassoc nnan arcp contract <3 x float> %tmp66, <float 0x3FD45F3060000000, float 0x3FD45F3060000000, float 0x3FD45F3060000000>
  %tmp68 = bitcast <3 x float> %tmp67 to <3 x i32>
  %tmp69 = shufflevector <3 x i32> %tmp68, <3 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  %tmp70 = shufflevector <4 x i32> %tmp69, <4 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  br label %.lr.ph2257

.lr.ph2257:                                       ; preds = %._crit_edge, %.lr.ph2257.preheader
  %__llpc_global_proxy_r8.62249 = phi <4 x i32> [ %__llpc_global_proxy_r8.7.lcssa, %._crit_edge ], [ %tmp70, %.lr.ph2257.preheader ]
  br i1 undef, label %._crit_edge, label %.lr.ph

.lr.ph:                                           ; preds = %.lr.ph2257
  br label %bb71

bb71:                                             ; preds = %bb74, %.lr.ph
  %__llpc_global_proxy_r8.72232 = phi <4 x i32> [ %__llpc_global_proxy_r8.62249, %.lr.ph ], [ %__llpc_global_proxy_r8.12.vec.insert1266, %bb74 ]
  %__llpc_global_proxy_r8.12.vec.insert1266 = insertelement <4 x i32> %__llpc_global_proxy_r8.72232, i32 0, i32 3
  br i1 undef, label %bb73, label %bb72

bb72:                                             ; preds = %bb71
  br label %bb74

bb73:                                             ; preds = %bb71
  br label %bb74

bb74:                                             ; preds = %bb73, %bb72
  br i1 undef, label %._crit_edge, label %bb71

._crit_edge:                                      ; preds = %bb74, %.lr.ph2257
  %__llpc_global_proxy_r8.7.lcssa = phi <4 x i32> [ %__llpc_global_proxy_r8.62249, %.lr.ph2257 ], [ %__llpc_global_proxy_r8.12.vec.insert1266, %bb74 ]
  %tmp75 = icmp ult i32 0, %tmp64
  br i1 %tmp75, label %.lr.ph2257, label %._crit_edge2258

._crit_edge2258:                                  ; preds = %._crit_edge, %bb61
  br label %bb76

bb76:                                             ; preds = %._crit_edge2258, %bb60
  br label %bb78

bb77:                                             ; preds = %.entry
  br label %bb78

bb78:                                             ; preds = %bb77, %bb76, %bb2
  ret void
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32)

; Function Attrs: nounwind readnone speculatable
declare float @llvm.floor.f32(float)

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32>, i32, i32)

