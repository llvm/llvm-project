; RUN: opt %loadPolly -polly-simplify -analyze < %s | FileCheck %s -match-full-lines
;
; The PHINode %cond91.sink.sink.us.sink.6 is in the middle of a region
; statement.
; Check that we are not expect a MemoryKind::PHI access for it, and no
; assertion guarding querying for it.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.pic_parameter_set_rbsp_t.3.45.87.129.192.255.465.927.969.990.1029 = type { i32, i32, i32, i32, i32, i32, [8 x i32], [6 x [16 x i32]], [2 x [64 x i32]], [6 x i32], [2 x i32], i32, i32, i32, [8 x i32], [8 x i32], [8 x i32], i32, i32, i32, i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }

@quant8_intra_default = external global [64 x i32], align 16
@quant_org = external global [16 x i32], align 16
@qmatrix = external local_unnamed_addr global [8 x i32*], align 16

; Function Attrs: nounwind uwtable
define void @AssignQuantParam(%struct.pic_parameter_set_rbsp_t.3.45.87.129.192.255.465.927.969.990.1029* %pps) local_unnamed_addr #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %pic_scaling_matrix_present_flag = getelementptr inbounds %struct.pic_parameter_set_rbsp_t.3.45.87.129.192.255.465.927.969.990.1029, %struct.pic_parameter_set_rbsp_t.3.45.87.129.192.255.465.927.969.990.1029* %pps, i64 0, i32 5
  %0 = load i32, i32* %pic_scaling_matrix_present_flag, align 4, !tbaa !1
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %land.lhs.true, label %if.else

land.lhs.true:                                    ; preds = %entry.split
  store i32* getelementptr inbounds ([16 x i32], [16 x i32]* @quant_org, i64 0, i64 0), i32** getelementptr inbounds ([8 x i32*], [8 x i32*]* @qmatrix, i64 0, i64 4), align 16, !tbaa !7
  br label %if.end161

if.else:                                          ; preds = %entry.split
  br label %if.else121.us.6

if.end161:                                        ; preds = %if.else121.us.7, %land.lhs.true
  ret void

if.else121.us.6:                                  ; preds = %if.else
  %arrayidx80.us.6 = getelementptr inbounds %struct.pic_parameter_set_rbsp_t.3.45.87.129.192.255.465.927.969.990.1029, %struct.pic_parameter_set_rbsp_t.3.45.87.129.192.255.465.927.969.990.1029* %pps, i64 0, i32 6, i64 6
  br i1 false, label %if.else121.us.7, label %if.else135.us.6

if.else135.us.6:                                  ; preds = %if.else121.us.6
  br label %if.else121.us.7

if.else121.us.7:                                  ; preds = %if.else135.us.6, %if.else121.us.6
  %cond91.sink.sink.us.sink.6 = phi i32* [ undef, %if.else135.us.6 ], [ getelementptr inbounds ([64 x i32], [64 x i32]* @quant8_intra_default, i64 0, i64 0), %if.else121.us.6 ]
  br label %if.end161
}

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 6.0.0 (trunk 308961)"}
!1 = !{!2, !3, i64 20}
!2 = !{!"", !3, i64 0, !5, i64 4, !5, i64 8, !3, i64 12, !3, i64 16, !3, i64 20, !3, i64 24, !3, i64 56, !3, i64 440, !3, i64 952, !3, i64 976, !3, i64 984, !5, i64 988, !5, i64 992, !3, i64 996, !3, i64 1028, !3, i64 1060, !3, i64 1092, !5, i64 1096, !5, i64 1100, !6, i64 1104, !5, i64 1112, !5, i64 1116, !3, i64 1120, !5, i64 1124, !5, i64 1128, !5, i64 1132, !5, i64 1136, !5, i64 1140, !3, i64 1144, !3, i64 1148, !3, i64 1152}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!"int", !3, i64 0}
!6 = !{!"any pointer", !3, i64 0}
!7 = !{!6, !6, i64 0}


; CHECK: SCoP could not be simplified
