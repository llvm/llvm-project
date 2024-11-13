; RUN: opt %loadNPMPolly -verify-dom-info -passes=polly-codegen -S < %s \
; RUN: -polly-invariant-load-hoisting=true | FileCheck %s
;
; This caused an infinite recursion during invariant load hoisting at some
; point. Check it does not and we add a "false" runtime check.
;
; CHECK:       polly.preload.begin:
; CHECK-NEXT:    br i1 false, label %polly.start, label %for.body.14.lr.ph
;
target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"

%struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573 = type { ptr, ptr, i32, i32, i32, i32, i32, [4 x i32], [4 x i32], double, %struct.AVRational.0.22.858.1188.1276.1298.1567 }
%struct.AVClass.10.32.868.1198.1286.1308.1566 = type { ptr, ptr, ptr, i32, i32, i32, ptr, ptr, i32, ptr, ptr }
%struct.AVOption.7.29.865.1195.1283.1305.1563 = type { ptr, ptr, i32, i32, %union.anon.6.28.864.1194.1282.1304.1562, double, double, i32, ptr }
%union.anon.6.28.864.1194.1282.1304.1562 = type { i64 }
%struct.AVOptionRanges.9.31.867.1197.1285.1307.1565 = type { ptr, i32, i32 }
%struct.AVOptionRange.8.30.866.1196.1284.1306.1564 = type { ptr, double, double, double, double, i32 }
%struct.AVFrame.5.27.863.1193.1281.1303.1572 = type { [8 x ptr], [8 x i32], ptr, i32, i32, i32, i32, i32, i32, %struct.AVRational.0.22.858.1188.1276.1298.1567, i64, i64, i64, i32, i32, i32, ptr, [8 x i64], i32, i32, i32, i32, i64, i32, i64, [8 x ptr], ptr, i32, ptr, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, ptr, i32, i32, i32, ptr, i32, i32, ptr }
%struct.AVFrameSideData.4.26.862.1192.1280.1302.1571 = type { i32, ptr, i32, ptr, ptr }
%struct.AVDictionary.3.25.861.1191.1279.1301.1570 = type opaque
%struct.AVBufferRef.2.24.860.1190.1278.1300.1569 = type { ptr, ptr, i32 }
%struct.AVBuffer.1.23.859.1189.1277.1299.1568 = type opaque
%struct.AVRational.0.22.858.1188.1276.1298.1567 = type { i32, i32 }

; Function Attrs: nounwind ssp
define void @fade(ptr %s) #0 {
entry:
  br label %for.cond.12.preheader.lr.ph

for.cond.12.preheader.lr.ph:                      ; preds = %entry
  %outpicref = getelementptr inbounds %struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573, ptr %s, i32 0, i32 1
  %arrayidx2 = getelementptr inbounds %struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573, ptr %s, i32 0, i32 8, i32 0
  %tobool = icmp eq i32 0, 0
  %arrayidx4 = getelementptr inbounds %struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573, ptr %s, i32 0, i32 8, i32 1
  %tmp = load i32, ptr %arrayidx4, align 4
  %tobool5 = icmp eq i32 %tmp, 0
  %h = getelementptr inbounds %struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573, ptr %s, i32 0, i32 3
  %tmp1 = load i32, ptr %h, align 4
  %cmp.48 = icmp sgt i32 %tmp1, 0
  %tmp2 = load ptr, ptr %outpicref, align 4
  %tmp3 = load ptr, ptr %tmp2, align 4
  br label %for.body.14.lr.ph

for.body.14.lr.ph:                                ; preds = %for.end, %for.cond.12.preheader.lr.ph
  %d.050 = phi ptr [ %tmp3, %for.cond.12.preheader.lr.ph ], [ poison, %for.end ]
  %w = getelementptr inbounds %struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573, ptr %s, i32 0, i32 2
  %tmp4 = load i32, ptr %w, align 4
  %cmp13.46 = icmp sgt i32 %tmp4, 0
  br label %for.body.14

for.body.14:                                      ; preds = %for.body.14, %for.body.14.lr.ph
  store i8 undef, ptr %d.050, align 1
  %arrayidx54 = getelementptr inbounds %struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573, ptr %s, i32 0, i32 8, i32 2
  %tmp5 = load i32, ptr %arrayidx54, align 4
  %add92 = add nuw nsw i32 0, 4
  %tmp6 = load i32, ptr %w, align 4
  %mul = shl nsw i32 %tmp6, 2
  %cmp13 = icmp slt i32 %add92, %mul
  br i1 %cmp13, label %for.body.14, label %for.end

for.end:                                          ; preds = %for.body.14
  %inc = add nuw nsw i32 0, 1
  %tmp7 = load i32, ptr %h, align 4
  %cmp = icmp slt i32 %inc, %tmp7
  br i1 %cmp, label %for.body.14.lr.ph, label %if.end.loopexit

if.end.loopexit:                                  ; preds = %for.end
  br label %if.end

if.end:                                           ; preds = %if.end.loopexit
  ret void
}
