; RUN: llc -asm-verbose=false -disable-branch-fold -disable-block-placement -disable-tail-duplicate -march=x86-64 -mcpu=nehalem -no-integrated-as < %s | FileCheck %s
; rdar://7236213
;
; The scheduler's 2-address hack has been disabled, so there is
; currently no good guarantee that this test will pass until the
; machine scheduler develops an equivalent heuristic.

; CodeGen shouldn't require any lea instructions inside the marked loop.
; It should properly set up post-increment uses and do coalescing for
; the induction variables.

; CHECK: # Start
; CHECK-NOT: lea
; CHECK: # Stop

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @foo(float* %I, i64 %IS, float* nocapture %Start, float* nocapture %Step, float* %O, i64 %OS, i64 %N) nounwind {
entry:
  %times4 = alloca float, align 4                 ; <float*> [#uses=3]
  %timesN = alloca float, align 4                 ; <float*> [#uses=2]
  %0 = load float, float* %Step, align 4                 ; <float> [#uses=8]
  %1 = ptrtoint float* %I to i64                  ; <i64> [#uses=1]
  %2 = ptrtoint float* %O to i64                  ; <i64> [#uses=1]
  %tmp = xor i64 %2, %1                           ; <i64> [#uses=1]
  %tmp16 = and i64 %tmp, 15                       ; <i64> [#uses=1]
  %3 = icmp eq i64 %tmp16, 0                      ; <i1> [#uses=1]
  %4 = trunc i64 %IS to i32                       ; <i32> [#uses=1]
  %5 = xor i32 %4, 1                              ; <i32> [#uses=1]
  %6 = trunc i64 %OS to i32                       ; <i32> [#uses=1]
  %7 = xor i32 %6, 1                              ; <i32> [#uses=1]
  %8 = or i32 %7, %5                              ; <i32> [#uses=1]
  %9 = icmp eq i32 %8, 0                          ; <i1> [#uses=1]
  br i1 %9, label %bb, label %return

bb:                                               ; preds = %entry
  %10 = load float, float* %Start, align 4               ; <float> [#uses=1]
  br label %bb2

bb1:                                              ; preds = %bb3
  %11 = load float, float* %I_addr.0, align 4            ; <float> [#uses=1]
  %12 = fmul float %11, %x.0                      ; <float> [#uses=1]
  store float %12, float* %O_addr.0, align 4
  %13 = fadd float %x.0, %0                       ; <float> [#uses=1]
  %indvar.next53 = add i64 %14, 1                 ; <i64> [#uses=1]
  br label %bb2

bb2:                                              ; preds = %bb1, %bb
  %14 = phi i64 [ %indvar.next53, %bb1 ], [ 0, %bb ] ; <i64> [#uses=21]
  %x.0 = phi float [ %13, %bb1 ], [ %10, %bb ]    ; <float> [#uses=6]
  %N_addr.0 = sub i64 %N, %14                     ; <i64> [#uses=4]
  %O_addr.0 = getelementptr float, float* %O, i64 %14    ; <float*> [#uses=4]
  %I_addr.0 = getelementptr float, float* %I, i64 %14    ; <float*> [#uses=3]
  %15 = icmp slt i64 %N_addr.0, 1                 ; <i1> [#uses=1]
  br i1 %15, label %bb4, label %bb3

bb3:                                              ; preds = %bb2
  %16 = ptrtoint float* %O_addr.0 to i64          ; <i64> [#uses=1]
  %17 = and i64 %16, 15                           ; <i64> [#uses=1]
  %18 = icmp eq i64 %17, 0                        ; <i1> [#uses=1]
  br i1 %18, label %bb4, label %bb1

bb4:                                              ; preds = %bb3, %bb2
  %19 = fmul float %0, 4.000000e+00               ; <float> [#uses=1]
  store float %19, float* %times4, align 4
  %20 = fmul float %0, 1.600000e+01               ; <float> [#uses=1]
  store float %20, float* %timesN, align 4
  %21 = fmul float %0, 0.000000e+00               ; <float> [#uses=1]
  %22 = fadd float %21, %x.0                      ; <float> [#uses=1]
  %23 = fadd float %x.0, %0                       ; <float> [#uses=1]
  %24 = fmul float %0, 2.000000e+00               ; <float> [#uses=1]
  %25 = fadd float %24, %x.0                      ; <float> [#uses=1]
  %26 = fmul float %0, 3.000000e+00               ; <float> [#uses=1]
  %27 = fadd float %26, %x.0                      ; <float> [#uses=1]
  %28 = insertelement <4 x float> undef, float %22, i32 0 ; <<4 x float>> [#uses=1]
  %29 = insertelement <4 x float> %28, float %23, i32 1 ; <<4 x float>> [#uses=1]
  %30 = insertelement <4 x float> %29, float %25, i32 2 ; <<4 x float>> [#uses=1]
  %31 = insertelement <4 x float> %30, float %27, i32 3 ; <<4 x float>> [#uses=5]
  %asmtmp.i = call <4 x float> asm "movss $1, $0\09\0Apshufd $$0, $0, $0", "=x,*m,~{dirflag},~{fpsr},~{flags}"(float* %times4) nounwind ; <<4 x float>> [#uses=3]
  %32 = fadd <4 x float> %31, %asmtmp.i           ; <<4 x float>> [#uses=3]
  %33 = fadd <4 x float> %32, %asmtmp.i           ; <<4 x float>> [#uses=3]
  %34 = fadd <4 x float> %33, %asmtmp.i           ; <<4 x float>> [#uses=2]
  %asmtmp.i18 = call <4 x float> asm "movss $1, $0\09\0Apshufd $$0, $0, $0", "=x,*m,~{dirflag},~{fpsr},~{flags}"(float* %timesN) nounwind ; <<4 x float>> [#uses=8]
  %35 = icmp sgt i64 %N_addr.0, 15                ; <i1> [#uses=2]
  br i1 %3, label %bb6.preheader, label %bb8

bb6.preheader:                                    ; preds = %bb4
  br i1 %35, label %bb.nph43, label %bb7

bb.nph43:                                         ; preds = %bb6.preheader
  %tmp108 = add i64 %14, 16                       ; <i64> [#uses=1]
  %tmp111 = add i64 %14, 4                        ; <i64> [#uses=1]
  %tmp115 = add i64 %14, 8                        ; <i64> [#uses=1]
  %tmp119 = add i64 %14, 12                       ; <i64> [#uses=1]
  %tmp134 = add i64 %N, -16                       ; <i64> [#uses=1]
  %tmp135 = sub i64 %tmp134, %14                  ; <i64> [#uses=1]
  call void asm sideeffect "# Start.", "~{dirflag},~{fpsr},~{flags}"() nounwind
  br label %bb5

bb5:                                              ; preds = %bb.nph43, %bb5
  %indvar102 = phi i64 [ 0, %bb.nph43 ], [ %indvar.next103, %bb5 ] ; <i64> [#uses=3]
  %vX3.041 = phi <4 x float> [ %34, %bb.nph43 ], [ %45, %bb5 ] ; <<4 x float>> [#uses=2]
  %vX0.039 = phi <4 x float> [ %31, %bb.nph43 ], [ %41, %bb5 ] ; <<4 x float>> [#uses=2]
  %vX2.037 = phi <4 x float> [ %33, %bb.nph43 ], [ %46, %bb5 ] ; <<4 x float>> [#uses=2]
  %vX1.036 = phi <4 x float> [ %32, %bb.nph43 ], [ %47, %bb5 ] ; <<4 x float>> [#uses=2]
  %tmp104 = shl i64 %indvar102, 4                 ; <i64> [#uses=5]
  %tmp105 = add i64 %14, %tmp104                  ; <i64> [#uses=2]
  %scevgep106 = getelementptr float, float* %I, i64 %tmp105 ; <float*> [#uses=1]
  %scevgep106107 = bitcast float* %scevgep106 to <4 x float>* ; <<4 x float>*> [#uses=1]
  %tmp109 = add i64 %tmp108, %tmp104              ; <i64> [#uses=2]
  %tmp112 = add i64 %tmp111, %tmp104              ; <i64> [#uses=2]
  %scevgep113 = getelementptr float, float* %I, i64 %tmp112 ; <float*> [#uses=1]
  %scevgep113114 = bitcast float* %scevgep113 to <4 x float>* ; <<4 x float>*> [#uses=1]
  %tmp116 = add i64 %tmp115, %tmp104              ; <i64> [#uses=2]
  %scevgep117 = getelementptr float, float* %I, i64 %tmp116 ; <float*> [#uses=1]
  %scevgep117118 = bitcast float* %scevgep117 to <4 x float>* ; <<4 x float>*> [#uses=1]
  %tmp120 = add i64 %tmp119, %tmp104              ; <i64> [#uses=2]
  %scevgep121 = getelementptr float, float* %I, i64 %tmp120 ; <float*> [#uses=1]
  %scevgep121122 = bitcast float* %scevgep121 to <4 x float>* ; <<4 x float>*> [#uses=1]
  %scevgep123 = getelementptr float, float* %O, i64 %tmp105 ; <float*> [#uses=1]
  %scevgep123124 = bitcast float* %scevgep123 to <4 x float>* ; <<4 x float>*> [#uses=1]
  %scevgep126 = getelementptr float, float* %O, i64 %tmp112 ; <float*> [#uses=1]
  %scevgep126127 = bitcast float* %scevgep126 to <4 x float>* ; <<4 x float>*> [#uses=1]
  %scevgep128 = getelementptr float, float* %O, i64 %tmp116 ; <float*> [#uses=1]
  %scevgep128129 = bitcast float* %scevgep128 to <4 x float>* ; <<4 x float>*> [#uses=1]
  %scevgep130 = getelementptr float, float* %O, i64 %tmp120 ; <float*> [#uses=1]
  %scevgep130131 = bitcast float* %scevgep130 to <4 x float>* ; <<4 x float>*> [#uses=1]
  %tmp132 = mul i64 %indvar102, -16               ; <i64> [#uses=1]
  %tmp136 = add i64 %tmp135, %tmp132              ; <i64> [#uses=2]
  %36 = load <4 x float>, <4 x float>* %scevgep106107, align 16 ; <<4 x float>> [#uses=1]
  %37 = load <4 x float>, <4 x float>* %scevgep113114, align 16 ; <<4 x float>> [#uses=1]
  %38 = load <4 x float>, <4 x float>* %scevgep117118, align 16 ; <<4 x float>> [#uses=1]
  %39 = load <4 x float>, <4 x float>* %scevgep121122, align 16 ; <<4 x float>> [#uses=1]
  %40 = fmul <4 x float> %36, %vX0.039            ; <<4 x float>> [#uses=1]
  %41 = fadd <4 x float> %vX0.039, %asmtmp.i18    ; <<4 x float>> [#uses=2]
  %42 = fmul <4 x float> %37, %vX1.036            ; <<4 x float>> [#uses=1]
  %43 = fmul <4 x float> %38, %vX2.037            ; <<4 x float>> [#uses=1]
  %44 = fmul <4 x float> %39, %vX3.041            ; <<4 x float>> [#uses=1]
  store <4 x float> %40, <4 x float>* %scevgep123124, align 16
  store <4 x float> %42, <4 x float>* %scevgep126127, align 16
  store <4 x float> %43, <4 x float>* %scevgep128129, align 16
  store <4 x float> %44, <4 x float>* %scevgep130131, align 16
  %45 = fadd <4 x float> %vX3.041, %asmtmp.i18    ; <<4 x float>> [#uses=1]
  %46 = fadd <4 x float> %vX2.037, %asmtmp.i18    ; <<4 x float>> [#uses=1]
  %47 = fadd <4 x float> %vX1.036, %asmtmp.i18    ; <<4 x float>> [#uses=1]
  %48 = icmp sgt i64 %tmp136, 15                  ; <i1> [#uses=1]
  %indvar.next103 = add i64 %indvar102, 1         ; <i64> [#uses=1]
  br i1 %48, label %bb5, label %bb6.bb7_crit_edge

bb6.bb7_crit_edge:                                ; preds = %bb5
  call void asm sideeffect "# Stop.", "~{dirflag},~{fpsr},~{flags}"() nounwind
  %scevgep110 = getelementptr float, float* %I, i64 %tmp109 ; <float*> [#uses=1]
  %scevgep125 = getelementptr float, float* %O, i64 %tmp109 ; <float*> [#uses=1]
  br label %bb7

bb7:                                              ; preds = %bb6.bb7_crit_edge, %bb6.preheader
  %I_addr.1.lcssa = phi float* [ %scevgep110, %bb6.bb7_crit_edge ], [ %I_addr.0, %bb6.preheader ] ; <float*> [#uses=1]
  %O_addr.1.lcssa = phi float* [ %scevgep125, %bb6.bb7_crit_edge ], [ %O_addr.0, %bb6.preheader ] ; <float*> [#uses=1]
  %vX0.0.lcssa = phi <4 x float> [ %41, %bb6.bb7_crit_edge ], [ %31, %bb6.preheader ] ; <<4 x float>> [#uses=1]
  %N_addr.1.lcssa = phi i64 [ %tmp136, %bb6.bb7_crit_edge ], [ %N_addr.0, %bb6.preheader ] ; <i64> [#uses=1]
  %asmtmp.i17 = call <4 x float> asm "movss $1, $0\09\0Apshufd $$0, $0, $0", "=x,*m,~{dirflag},~{fpsr},~{flags}"(float* %times4) nounwind ; <<4 x float>> [#uses=0]
  br label %bb11

bb8:                                              ; preds = %bb4
  br i1 %35, label %bb.nph, label %bb11

bb.nph:                                           ; preds = %bb8
  %I_addr.0.sum = add i64 %14, -1                 ; <i64> [#uses=1]
  %49 = getelementptr inbounds float, float* %I, i64 %I_addr.0.sum ; <float*> [#uses=1]
  %50 = bitcast float* %49 to <4 x float>*        ; <<4 x float>*> [#uses=1]
  %51 = load <4 x float>, <4 x float>* %50, align 16           ; <<4 x float>> [#uses=1]
  %tmp54 = add i64 %14, 16                        ; <i64> [#uses=1]
  %tmp56 = add i64 %14, 3                         ; <i64> [#uses=1]
  %tmp60 = add i64 %14, 7                         ; <i64> [#uses=1]
  %tmp64 = add i64 %14, 11                        ; <i64> [#uses=1]
  %tmp68 = add i64 %14, 15                        ; <i64> [#uses=1]
  %tmp76 = add i64 %14, 4                         ; <i64> [#uses=1]
  %tmp80 = add i64 %14, 8                         ; <i64> [#uses=1]
  %tmp84 = add i64 %14, 12                        ; <i64> [#uses=1]
  %tmp90 = add i64 %N, -16                        ; <i64> [#uses=1]
  %tmp91 = sub i64 %tmp90, %14                    ; <i64> [#uses=1]
  br label %bb9

bb9:                                              ; preds = %bb.nph, %bb9
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %bb9 ] ; <i64> [#uses=3]
  %vX3.125 = phi <4 x float> [ %34, %bb.nph ], [ %69, %bb9 ] ; <<4 x float>> [#uses=2]
  %vX0.223 = phi <4 x float> [ %31, %bb.nph ], [ %65, %bb9 ] ; <<4 x float>> [#uses=2]
  %vX2.121 = phi <4 x float> [ %33, %bb.nph ], [ %70, %bb9 ] ; <<4 x float>> [#uses=2]
  %vX1.120 = phi <4 x float> [ %32, %bb.nph ], [ %71, %bb9 ] ; <<4 x float>> [#uses=2]
  %vI0.019 = phi <4 x float> [ %51, %bb.nph ], [ %55, %bb9 ] ; <<4 x float>> [#uses=1]
  %tmp51 = shl i64 %indvar, 4                     ; <i64> [#uses=9]
  %tmp55 = add i64 %tmp54, %tmp51                 ; <i64> [#uses=2]
  %tmp57 = add i64 %tmp56, %tmp51                 ; <i64> [#uses=1]
  %scevgep58 = getelementptr float, float* %I, i64 %tmp57 ; <float*> [#uses=1]
  %scevgep5859 = bitcast float* %scevgep58 to <4 x float>* ; <<4 x float>*> [#uses=1]
  %tmp61 = add i64 %tmp60, %tmp51                 ; <i64> [#uses=1]
  %scevgep62 = getelementptr float, float* %I, i64 %tmp61 ; <float*> [#uses=1]
  %scevgep6263 = bitcast float* %scevgep62 to <4 x float>* ; <<4 x float>*> [#uses=1]
  %tmp65 = add i64 %tmp64, %tmp51                 ; <i64> [#uses=1]
  %scevgep66 = getelementptr float, float* %I, i64 %tmp65 ; <float*> [#uses=1]
  %scevgep6667 = bitcast float* %scevgep66 to <4 x float>* ; <<4 x float>*> [#uses=1]
  %tmp69 = add i64 %tmp68, %tmp51                 ; <i64> [#uses=1]
  %scevgep70 = getelementptr float, float* %I, i64 %tmp69 ; <float*> [#uses=1]
  %scevgep7071 = bitcast float* %scevgep70 to <4 x float>* ; <<4 x float>*> [#uses=1]
  %tmp72 = add i64 %14, %tmp51                    ; <i64> [#uses=1]
  %scevgep73 = getelementptr float, float* %O, i64 %tmp72 ; <float*> [#uses=1]
  %scevgep7374 = bitcast float* %scevgep73 to <4 x float>* ; <<4 x float>*> [#uses=1]
  %tmp77 = add i64 %tmp76, %tmp51                 ; <i64> [#uses=1]
  %scevgep78 = getelementptr float, float* %O, i64 %tmp77 ; <float*> [#uses=1]
  %scevgep7879 = bitcast float* %scevgep78 to <4 x float>* ; <<4 x float>*> [#uses=1]
  %tmp81 = add i64 %tmp80, %tmp51                 ; <i64> [#uses=1]
  %scevgep82 = getelementptr float, float* %O, i64 %tmp81 ; <float*> [#uses=1]
  %scevgep8283 = bitcast float* %scevgep82 to <4 x float>* ; <<4 x float>*> [#uses=1]
  %tmp85 = add i64 %tmp84, %tmp51                 ; <i64> [#uses=1]
  %scevgep86 = getelementptr float, float* %O, i64 %tmp85 ; <float*> [#uses=1]
  %scevgep8687 = bitcast float* %scevgep86 to <4 x float>* ; <<4 x float>*> [#uses=1]
  %tmp88 = mul i64 %indvar, -16                   ; <i64> [#uses=1]
  %tmp92 = add i64 %tmp91, %tmp88                 ; <i64> [#uses=2]
  %52 = load <4 x float>, <4 x float>* %scevgep5859, align 16  ; <<4 x float>> [#uses=2]
  %53 = load <4 x float>, <4 x float>* %scevgep6263, align 16  ; <<4 x float>> [#uses=2]
  %54 = load <4 x float>, <4 x float>* %scevgep6667, align 16  ; <<4 x float>> [#uses=2]
  %55 = load <4 x float>, <4 x float>* %scevgep7071, align 16  ; <<4 x float>> [#uses=2]
  %56 = shufflevector <4 x float> %vI0.019, <4 x float> %52, <4 x i32> <i32 4, i32 1, i32 2, i32 3> ; <<4 x float>> [#uses=1]
  %57 = shufflevector <4 x float> %56, <4 x float> undef, <4 x i32> <i32 1, i32 2, i32 3, i32 0> ; <<4 x float>> [#uses=1]
  %58 = shufflevector <4 x float> %52, <4 x float> %53, <4 x i32> <i32 4, i32 1, i32 2, i32 3> ; <<4 x float>> [#uses=1]
  %59 = shufflevector <4 x float> %58, <4 x float> undef, <4 x i32> <i32 1, i32 2, i32 3, i32 0> ; <<4 x float>> [#uses=1]
  %60 = shufflevector <4 x float> %53, <4 x float> %54, <4 x i32> <i32 4, i32 1, i32 2, i32 3> ; <<4 x float>> [#uses=1]
  %61 = shufflevector <4 x float> %60, <4 x float> undef, <4 x i32> <i32 1, i32 2, i32 3, i32 0> ; <<4 x float>> [#uses=1]
  %62 = shufflevector <4 x float> %54, <4 x float> %55, <4 x i32> <i32 4, i32 1, i32 2, i32 3> ; <<4 x float>> [#uses=1]
  %63 = shufflevector <4 x float> %62, <4 x float> undef, <4 x i32> <i32 1, i32 2, i32 3, i32 0> ; <<4 x float>> [#uses=1]
  %64 = fmul <4 x float> %57, %vX0.223            ; <<4 x float>> [#uses=1]
  %65 = fadd <4 x float> %vX0.223, %asmtmp.i18    ; <<4 x float>> [#uses=2]
  %66 = fmul <4 x float> %59, %vX1.120            ; <<4 x float>> [#uses=1]
  %67 = fmul <4 x float> %61, %vX2.121            ; <<4 x float>> [#uses=1]
  %68 = fmul <4 x float> %63, %vX3.125            ; <<4 x float>> [#uses=1]
  store <4 x float> %64, <4 x float>* %scevgep7374, align 16
  store <4 x float> %66, <4 x float>* %scevgep7879, align 16
  store <4 x float> %67, <4 x float>* %scevgep8283, align 16
  store <4 x float> %68, <4 x float>* %scevgep8687, align 16
  %69 = fadd <4 x float> %vX3.125, %asmtmp.i18    ; <<4 x float>> [#uses=1]
  %70 = fadd <4 x float> %vX2.121, %asmtmp.i18    ; <<4 x float>> [#uses=1]
  %71 = fadd <4 x float> %vX1.120, %asmtmp.i18    ; <<4 x float>> [#uses=1]
  %72 = icmp sgt i64 %tmp92, 15                   ; <i1> [#uses=1]
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br i1 %72, label %bb9, label %bb10.bb11.loopexit_crit_edge

bb10.bb11.loopexit_crit_edge:                     ; preds = %bb9
  %scevgep = getelementptr float, float* %I, i64 %tmp55  ; <float*> [#uses=1]
  %scevgep75 = getelementptr float, float* %O, i64 %tmp55 ; <float*> [#uses=1]
  br label %bb11

bb11:                                             ; preds = %bb8, %bb10.bb11.loopexit_crit_edge, %bb7
  %N_addr.2 = phi i64 [ %N_addr.1.lcssa, %bb7 ], [ %tmp92, %bb10.bb11.loopexit_crit_edge ], [ %N_addr.0, %bb8 ] ; <i64> [#uses=2]
  %vX0.1 = phi <4 x float> [ %vX0.0.lcssa, %bb7 ], [ %65, %bb10.bb11.loopexit_crit_edge ], [ %31, %bb8 ] ; <<4 x float>> [#uses=1]
  %O_addr.2 = phi float* [ %O_addr.1.lcssa, %bb7 ], [ %scevgep75, %bb10.bb11.loopexit_crit_edge ], [ %O_addr.0, %bb8 ] ; <float*> [#uses=1]
  %I_addr.2 = phi float* [ %I_addr.1.lcssa, %bb7 ], [ %scevgep, %bb10.bb11.loopexit_crit_edge ], [ %I_addr.0, %bb8 ] ; <float*> [#uses=1]
  %73 = extractelement <4 x float> %vX0.1, i32 0  ; <float> [#uses=2]
  %74 = icmp sgt i64 %N_addr.2, 0                 ; <i1> [#uses=1]
  br i1 %74, label %bb12, label %bb14

bb12:                                             ; preds = %bb11, %bb12
  %indvar94 = phi i64 [ %indvar.next95, %bb12 ], [ 0, %bb11 ] ; <i64> [#uses=3]
  %x.130 = phi float [ %77, %bb12 ], [ %73, %bb11 ] ; <float> [#uses=2]
  %I_addr.433 = getelementptr float, float* %I_addr.2, i64 %indvar94 ; <float*> [#uses=1]
  %O_addr.432 = getelementptr float, float* %O_addr.2, i64 %indvar94 ; <float*> [#uses=1]
  %75 = load float, float* %I_addr.433, align 4          ; <float> [#uses=1]
  %76 = fmul float %75, %x.130                    ; <float> [#uses=1]
  store float %76, float* %O_addr.432, align 4
  %77 = fadd float %x.130, %0                     ; <float> [#uses=2]
  %indvar.next95 = add i64 %indvar94, 1           ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %indvar.next95, %N_addr.2 ; <i1> [#uses=1]
  br i1 %exitcond, label %bb14, label %bb12

bb14:                                             ; preds = %bb12, %bb11
  %x.1.lcssa = phi float [ %73, %bb11 ], [ %77, %bb12 ] ; <float> [#uses=1]
  store float %x.1.lcssa, float* %Start, align 4
  ret void

return:                                           ; preds = %entry
  ret void
}

; Codegen shouldn't crash on this testcase.

define void @bar(i32 %a, i32 %b) nounwind {
entry:                           ; preds = %bb1, %entry, %for.end204
  br label %outer

outer:                                     ; preds = %bb1, %entry
  %i6 = phi i32 [ %storemerge171, %bb1 ], [ %a, %entry ] ; <i32> [#uses=2]
  %storemerge171 = add i32 %i6, 1      ; <i32> [#uses=1]
  br label %inner

inner:                                       ; preds = %bb0, %if.end275
  %i8 = phi i32 [ %a, %outer ], [ %indvar.next159, %bb0 ] ; <i32> [#uses=2]
  %t338 = load i32, i32* undef                     ; <i32> [#uses=1]
  %t191 = mul i32 %i8, %t338        ; <i32> [#uses=1]
  %t179 = add i32 %i6, %t191        ; <i32> [#uses=1]
  br label %bb0

bb0:                                     ; preds = %for.body332
  %indvar.next159 = add i32 %i8, 1     ; <i32> [#uses=1]
  br i1 undef, label %bb1, label %inner

bb1:                                     ; preds = %bb0, %outer
  %midx.4 = phi i32 [ %t179, %bb0 ] ; <i32> [#uses=0]
  br label %outer
}
