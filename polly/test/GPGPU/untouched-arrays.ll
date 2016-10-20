; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; REQUIRES: pollyacc

; CODE: Code
; CODE-NEXT: ====
; CODE-NEXT: # host
; CODE-NEXT: {
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_global_1, MemRef_global_1, (142) * sizeof(i32), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(10);
; CODE-NEXT:     dim3 k0_dimGrid(1);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_global_1);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_global_1, dev_MemRef_global_1, (142) * sizeof(i32), cudaMemcpyDeviceToHost));
; CODE-NEXT: }

; CODE: # kernel0
; CODE-NEXT: Stmt_bb33(t0, 0);


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.hoge = type { [23 x i16], [22 x i16], [14 x i16], [13 x i16] }

@global = external global [9 x %struct.hoge], align 16
@global.1 = external global [9 x [152 x i32]], align 16

; Function Attrs: nounwind uwtable
define void @widget() #0 {
bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  br i1 undef, label %bb1, label %bb2

bb2:                                              ; preds = %bb2, %bb1
  br i1 undef, label %bb2, label %bb3

bb3:                                              ; preds = %bb3, %bb2
  br i1 undef, label %bb3, label %bb4

bb4:                                              ; preds = %bb4, %bb3
  br i1 undef, label %bb4, label %bb5

bb5:                                              ; preds = %bb5, %bb4
  br i1 undef, label %bb5, label %bb6

bb6:                                              ; preds = %bb6, %bb5
  br i1 undef, label %bb6, label %bb7

bb7:                                              ; preds = %bb7, %bb6
  br i1 undef, label %bb7, label %bb8

bb8:                                              ; preds = %bb8, %bb7
  br i1 undef, label %bb8, label %bb9

bb9:                                              ; preds = %bb8
  br label %bb10

bb10:                                             ; preds = %bb12, %bb9
  br label %bb11

bb11:                                             ; preds = %bb11, %bb10
  br i1 undef, label %bb11, label %bb12

bb12:                                             ; preds = %bb11
  br i1 undef, label %bb10, label %bb13

bb13:                                             ; preds = %bb18, %bb12
  br i1 undef, label %bb16, label %bb14

bb14:                                             ; preds = %bb16, %bb13
  br i1 undef, label %bb15, label %bb18

bb15:                                             ; preds = %bb14
  br label %bb17

bb16:                                             ; preds = %bb16, %bb13
  br i1 undef, label %bb16, label %bb14

bb17:                                             ; preds = %bb17, %bb15
  br i1 undef, label %bb17, label %bb18

bb18:                                             ; preds = %bb17, %bb14
  br i1 undef, label %bb13, label %bb19

bb19:                                             ; preds = %bb25, %bb18
  br label %bb20

bb20:                                             ; preds = %bb24, %bb19
  br i1 undef, label %bb21, label %bb24

bb21:                                             ; preds = %bb20
  br i1 undef, label %bb23, label %bb22

bb22:                                             ; preds = %bb21
  br label %bb24

bb23:                                             ; preds = %bb21
  br label %bb24

bb24:                                             ; preds = %bb23, %bb22, %bb20
  br i1 undef, label %bb20, label %bb25

bb25:                                             ; preds = %bb24
  br i1 undef, label %bb19, label %bb26

bb26:                                             ; preds = %bb56, %bb25
  %tmp = phi [9 x [152 x i32]]* [ undef, %bb56 ], [ bitcast (i32* getelementptr inbounds ([9 x [152 x i32]], [9 x [152 x i32]]* @global.1, i64 0, i64 0, i64 32) to [9 x [152 x i32]]*), %bb25 ]
  br label %bb27

bb27:                                             ; preds = %bb27, %bb26
  br i1 undef, label %bb27, label %bb28

bb28:                                             ; preds = %bb27
  %tmp29 = bitcast [9 x [152 x i32]]* %tmp to i32*
  br label %bb30

bb30:                                             ; preds = %bb38, %bb28
  %tmp31 = phi i32 [ 3, %bb28 ], [ %tmp40, %bb38 ]
  %tmp32 = phi i32* [ %tmp29, %bb28 ], [ %tmp39, %bb38 ]
  br label %bb33

bb33:                                             ; preds = %bb33, %bb30
  %tmp34 = phi i32 [ 0, %bb30 ], [ %tmp37, %bb33 ]
  %tmp35 = phi i32* [ %tmp32, %bb30 ], [ undef, %bb33 ]
  %tmp36 = getelementptr inbounds i32, i32* %tmp35, i64 1
  store i32 undef, i32* %tmp36, align 4, !tbaa !1
  %tmp37 = add nuw nsw i32 %tmp34, 1
  br i1 false, label %bb33, label %bb38

bb38:                                             ; preds = %bb33
  %tmp39 = getelementptr i32, i32* %tmp32, i64 12
  %tmp40 = add nuw nsw i32 %tmp31, 1
  %tmp41 = icmp ne i32 %tmp40, 13
  br i1 %tmp41, label %bb30, label %bb42

bb42:                                             ; preds = %bb38
  %tmp43 = getelementptr inbounds [9 x %struct.hoge], [9 x %struct.hoge]* @global, i64 0, i64 0, i32 3, i64 0
  br label %bb44

bb44:                                             ; preds = %bb51, %bb42
  %tmp45 = phi i32 [ 0, %bb42 ], [ %tmp52, %bb51 ]
  %tmp46 = phi i16* [ %tmp43, %bb42 ], [ undef, %bb51 ]
  %tmp47 = load i16, i16* %tmp46, align 2, !tbaa !5
  br label %bb48

bb48:                                             ; preds = %bb48, %bb44
  %tmp49 = phi i32 [ 0, %bb44 ], [ %tmp50, %bb48 ]
  %tmp50 = add nuw nsw i32 %tmp49, 1
  br i1 false, label %bb48, label %bb51

bb51:                                             ; preds = %bb48
  %tmp52 = add nuw nsw i32 %tmp45, 1
  %tmp53 = icmp ne i32 %tmp52, 13
  br i1 %tmp53, label %bb44, label %bb54

bb54:                                             ; preds = %bb51
  br label %bb55

bb55:                                             ; preds = %bb55, %bb54
  br i1 undef, label %bb55, label %bb56

bb56:                                             ; preds = %bb55
  br i1 undef, label %bb26, label %bb57

bb57:                                             ; preds = %bb60, %bb56
  br label %bb58

bb58:                                             ; preds = %bb58, %bb57
  br i1 undef, label %bb58, label %bb59

bb59:                                             ; preds = %bb59, %bb58
  br i1 undef, label %bb59, label %bb60

bb60:                                             ; preds = %bb59
  br i1 undef, label %bb57, label %bb61

bb61:                                             ; preds = %bb65, %bb60
  br label %bb62

bb62:                                             ; preds = %bb64, %bb61
  br label %bb63

bb63:                                             ; preds = %bb63, %bb62
  br i1 undef, label %bb63, label %bb64

bb64:                                             ; preds = %bb63
  br i1 undef, label %bb62, label %bb65

bb65:                                             ; preds = %bb64
  br i1 undef, label %bb61, label %bb66

bb66:                                             ; preds = %bb70, %bb65
  br label %bb67

bb67:                                             ; preds = %bb69, %bb66
  br label %bb68

bb68:                                             ; preds = %bb68, %bb67
  br i1 undef, label %bb68, label %bb69

bb69:                                             ; preds = %bb68
  br i1 undef, label %bb67, label %bb70

bb70:                                             ; preds = %bb69
  br i1 undef, label %bb66, label %bb71

bb71:                                             ; preds = %bb73, %bb70
  br label %bb72

bb72:                                             ; preds = %bb72, %bb71
  br i1 undef, label %bb72, label %bb73

bb73:                                             ; preds = %bb72
  br i1 undef, label %bb71, label %bb74

bb74:                                             ; preds = %bb80, %bb73
  br label %bb75

bb75:                                             ; preds = %bb79, %bb74
  br label %bb76

bb76:                                             ; preds = %bb78, %bb75
  br label %bb77

bb77:                                             ; preds = %bb77, %bb76
  br i1 undef, label %bb77, label %bb78

bb78:                                             ; preds = %bb77
  br i1 undef, label %bb76, label %bb79

bb79:                                             ; preds = %bb78
  br i1 undef, label %bb75, label %bb80

bb80:                                             ; preds = %bb79
  br i1 undef, label %bb74, label %bb81

bb81:                                             ; preds = %bb85, %bb80
  br label %bb82

bb82:                                             ; preds = %bb84, %bb81
  br label %bb83

bb83:                                             ; preds = %bb83, %bb82
  br i1 undef, label %bb83, label %bb84

bb84:                                             ; preds = %bb83
  br i1 undef, label %bb82, label %bb85

bb85:                                             ; preds = %bb84
  br i1 undef, label %bb81, label %bb86

bb86:                                             ; preds = %bb85
  ret void
}

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 4.0.0 (http://llvm.org/git/clang.git 359c45534e46a8ef263db1a8b855740bbeca6998) (http://llvm.org/git/llvm.git 2628aff56683b7652abe9f9eb9e54a82d1716aa7)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"short", !3, i64 0}
