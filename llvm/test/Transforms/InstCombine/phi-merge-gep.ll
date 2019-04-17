; RUN: opt < %s -S -instcombine > %t
; RUN: grep "= getelementptr" %t | count 20
; RUN: grep "= phi" %t | count 13

; Don't push the geps through these phis, because they would require
; two phis each, which burdens the loop with high register pressure.

define void @foo(float* %Ar, float* %Ai, i64 %As, float* %Cr, float* %Ci, i64 %Cs, i64 %n) nounwind {
entry:
  %0 = getelementptr inbounds float, float* %Ar, i64 0   ; <float*> [#uses=1]
  %1 = getelementptr inbounds float, float* %Ai, i64 0   ; <float*> [#uses=1]
  %2 = mul i64 %n, %As                            ; <i64> [#uses=1]
  %3 = getelementptr inbounds float, float* %Ar, i64 %2  ; <float*> [#uses=1]
  %4 = mul i64 %n, %As                            ; <i64> [#uses=1]
  %5 = getelementptr inbounds float, float* %Ai, i64 %4  ; <float*> [#uses=1]
  %6 = mul i64 %n, 2                              ; <i64> [#uses=1]
  %7 = mul i64 %6, %As                            ; <i64> [#uses=1]
  %8 = getelementptr inbounds float, float* %Ar, i64 %7  ; <float*> [#uses=1]
  %9 = mul i64 %n, 2                              ; <i64> [#uses=1]
  %10 = mul i64 %9, %As                           ; <i64> [#uses=1]
  %11 = getelementptr inbounds float, float* %Ai, i64 %10 ; <float*> [#uses=1]
  %12 = getelementptr inbounds float, float* %Cr, i64 0  ; <float*> [#uses=1]
  %13 = getelementptr inbounds float, float* %Ci, i64 0  ; <float*> [#uses=1]
  %14 = mul i64 %n, %Cs                           ; <i64> [#uses=1]
  %15 = getelementptr inbounds float, float* %Cr, i64 %14 ; <float*> [#uses=1]
  %16 = mul i64 %n, %Cs                           ; <i64> [#uses=1]
  %17 = getelementptr inbounds float, float* %Ci, i64 %16 ; <float*> [#uses=1]
  %18 = mul i64 %n, 2                             ; <i64> [#uses=1]
  %19 = mul i64 %18, %Cs                          ; <i64> [#uses=1]
  %20 = getelementptr inbounds float, float* %Cr, i64 %19 ; <float*> [#uses=1]
  %21 = mul i64 %n, 2                             ; <i64> [#uses=1]
  %22 = mul i64 %21, %Cs                          ; <i64> [#uses=1]
  %23 = getelementptr inbounds float, float* %Ci, i64 %22 ; <float*> [#uses=1]
  br label %bb13

bb:                                               ; preds = %bb13
  %24 = load float, float* %A0r.0, align 4               ; <float> [#uses=1]
  %25 = load float, float* %A0i.0, align 4               ; <float> [#uses=1]
  %26 = load float, float* %A1r.0, align 4               ; <float> [#uses=2]
  %27 = load float, float* %A1i.0, align 4               ; <float> [#uses=2]
  %28 = load float, float* %A2r.0, align 4               ; <float> [#uses=2]
  %29 = load float, float* %A2i.0, align 4               ; <float> [#uses=2]
  %30 = fadd float %26, %28                       ; <float> [#uses=2]
  %31 = fadd float %27, %29                       ; <float> [#uses=2]
  %32 = fsub float %26, %28                       ; <float> [#uses=1]
  %33 = fsub float %27, %29                       ; <float> [#uses=1]
  %34 = fadd float %24, %30                       ; <float> [#uses=2]
  %35 = fadd float %25, %31                       ; <float> [#uses=2]
  %36 = fmul float %30, -1.500000e+00             ; <float> [#uses=1]
  %37 = fmul float %31, -1.500000e+00             ; <float> [#uses=1]
  %38 = fadd float %34, %36                       ; <float> [#uses=2]
  %39 = fadd float %35, %37                       ; <float> [#uses=2]
  %40 = fmul float %32, 0x3FEBB67AE0000000        ; <float> [#uses=2]
  %41 = fmul float %33, 0x3FEBB67AE0000000        ; <float> [#uses=2]
  %42 = fadd float %38, %41                       ; <float> [#uses=1]
  %43 = fsub float %39, %40                       ; <float> [#uses=1]
  %44 = fsub float %38, %41                       ; <float> [#uses=1]
  %45 = fadd float %39, %40                       ; <float> [#uses=1]
  store float %34, float* %C0r.0, align 4
  store float %35, float* %C0i.0, align 4
  store float %42, float* %C1r.0, align 4
  store float %43, float* %C1i.0, align 4
  store float %44, float* %C2r.0, align 4
  store float %45, float* %C2i.0, align 4
  %46 = getelementptr inbounds float, float* %A0r.0, i64 %As ; <float*> [#uses=1]
  %47 = getelementptr inbounds float, float* %A0i.0, i64 %As ; <float*> [#uses=1]
  %48 = getelementptr inbounds float, float* %A1r.0, i64 %As ; <float*> [#uses=1]
  %49 = getelementptr inbounds float, float* %A1i.0, i64 %As ; <float*> [#uses=1]
  %50 = getelementptr inbounds float, float* %A2r.0, i64 %As ; <float*> [#uses=1]
  %51 = getelementptr inbounds float, float* %A2i.0, i64 %As ; <float*> [#uses=1]
  %52 = getelementptr inbounds float, float* %C0r.0, i64 %Cs ; <float*> [#uses=1]
  %53 = getelementptr inbounds float, float* %C0i.0, i64 %Cs ; <float*> [#uses=1]
  %54 = getelementptr inbounds float, float* %C1r.0, i64 %Cs ; <float*> [#uses=1]
  %55 = getelementptr inbounds float, float* %C1i.0, i64 %Cs ; <float*> [#uses=1]
  %56 = getelementptr inbounds float, float* %C2r.0, i64 %Cs ; <float*> [#uses=1]
  %57 = getelementptr inbounds float, float* %C2i.0, i64 %Cs ; <float*> [#uses=1]
  %58 = add nsw i64 %i.0, 1                       ; <i64> [#uses=1]
  br label %bb13

bb13:                                             ; preds = %bb, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %58, %bb ]      ; <i64> [#uses=2]
  %C2i.0 = phi float* [ %23, %entry ], [ %57, %bb ] ; <float*> [#uses=2]
  %C2r.0 = phi float* [ %20, %entry ], [ %56, %bb ] ; <float*> [#uses=2]
  %C1i.0 = phi float* [ %17, %entry ], [ %55, %bb ] ; <float*> [#uses=2]
  %C1r.0 = phi float* [ %15, %entry ], [ %54, %bb ] ; <float*> [#uses=2]
  %C0i.0 = phi float* [ %13, %entry ], [ %53, %bb ] ; <float*> [#uses=2]
  %C0r.0 = phi float* [ %12, %entry ], [ %52, %bb ] ; <float*> [#uses=2]
  %A2i.0 = phi float* [ %11, %entry ], [ %51, %bb ] ; <float*> [#uses=2]
  %A2r.0 = phi float* [ %8, %entry ], [ %50, %bb ] ; <float*> [#uses=2]
  %A1i.0 = phi float* [ %5, %entry ], [ %49, %bb ] ; <float*> [#uses=2]
  %A1r.0 = phi float* [ %3, %entry ], [ %48, %bb ] ; <float*> [#uses=2]
  %A0i.0 = phi float* [ %1, %entry ], [ %47, %bb ] ; <float*> [#uses=2]
  %A0r.0 = phi float* [ %0, %entry ], [ %46, %bb ] ; <float*> [#uses=2]
  %59 = icmp slt i64 %i.0, %n                     ; <i1> [#uses=1]
  br i1 %59, label %bb, label %bb14

bb14:                                             ; preds = %bb13
  br label %return

return:                                           ; preds = %bb14
  ret void
}
