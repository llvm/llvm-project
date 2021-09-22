; RUN: opt %loadPolly -verify-dom-info -polly-codegen -S < %s \
; RUN: -polly-invariant-load-hoisting=true | FileCheck %s
;
; This caused an infinite recursion during invariant load hoisting at some
; point. Check it does not and we add a "false" runtime check.
;
; CHECK:       polly.preload.begin:
; CHECK-NEXT:    br i1 false, label %polly.start, label %for.body.14.lr.ph
;
target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"

%struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573 = type { %struct.AVClass.10.32.868.1198.1286.1308.1566*, %struct.AVFrame.5.27.863.1193.1281.1303.1572*, i32, i32, i32, i32, i32, [4 x i32], [4 x i32], double, %struct.AVRational.0.22.858.1188.1276.1298.1567 }
%struct.AVClass.10.32.868.1198.1286.1308.1566 = type { i8*, i8* (i8*)*, %struct.AVOption.7.29.865.1195.1283.1305.1563*, i32, i32, i32, i8* (i8*, i8*)*, %struct.AVClass.10.32.868.1198.1286.1308.1566* (%struct.AVClass.10.32.868.1198.1286.1308.1566*)*, i32, i32 (i8*)*, i32 (%struct.AVOptionRanges.9.31.867.1197.1285.1307.1565**, i8*, i8*, i32)* }
%struct.AVOption.7.29.865.1195.1283.1305.1563 = type { i8*, i8*, i32, i32, %union.anon.6.28.864.1194.1282.1304.1562, double, double, i32, i8* }
%union.anon.6.28.864.1194.1282.1304.1562 = type { i64 }
%struct.AVOptionRanges.9.31.867.1197.1285.1307.1565 = type { %struct.AVOptionRange.8.30.866.1196.1284.1306.1564**, i32, i32 }
%struct.AVOptionRange.8.30.866.1196.1284.1306.1564 = type { i8*, double, double, double, double, i32 }
%struct.AVFrame.5.27.863.1193.1281.1303.1572 = type { [8 x i8*], [8 x i32], i8**, i32, i32, i32, i32, i32, i32, %struct.AVRational.0.22.858.1188.1276.1298.1567, i64, i64, i64, i32, i32, i32, i8*, [8 x i64], i32, i32, i32, i32, i64, i32, i64, [8 x %struct.AVBufferRef.2.24.860.1190.1278.1300.1569*], %struct.AVBufferRef.2.24.860.1190.1278.1300.1569**, i32, %struct.AVFrameSideData.4.26.862.1192.1280.1302.1571**, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, %struct.AVDictionary.3.25.861.1191.1279.1301.1570*, i32, i32, i32, i8*, i32, i32, %struct.AVBufferRef.2.24.860.1190.1278.1300.1569* }
%struct.AVFrameSideData.4.26.862.1192.1280.1302.1571 = type { i32, i8*, i32, %struct.AVDictionary.3.25.861.1191.1279.1301.1570*, %struct.AVBufferRef.2.24.860.1190.1278.1300.1569* }
%struct.AVDictionary.3.25.861.1191.1279.1301.1570 = type opaque
%struct.AVBufferRef.2.24.860.1190.1278.1300.1569 = type { %struct.AVBuffer.1.23.859.1189.1277.1299.1568*, i8*, i32 }
%struct.AVBuffer.1.23.859.1189.1277.1299.1568 = type opaque
%struct.AVRational.0.22.858.1188.1276.1298.1567 = type { i32, i32 }

; Function Attrs: nounwind ssp
define void @fade(%struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573* %s) #0 {
entry:
  br label %for.cond.12.preheader.lr.ph

for.cond.12.preheader.lr.ph:                      ; preds = %entry
  %outpicref = getelementptr inbounds %struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573, %struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573* %s, i32 0, i32 1
  %arrayidx2 = getelementptr inbounds %struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573, %struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573* %s, i32 0, i32 8, i32 0
  %tobool = icmp eq i32 0, 0
  %arrayidx4 = getelementptr inbounds %struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573, %struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573* %s, i32 0, i32 8, i32 1
  %tmp = load i32, i32* %arrayidx4, align 4
  %tobool5 = icmp eq i32 %tmp, 0
  %h = getelementptr inbounds %struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573, %struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573* %s, i32 0, i32 3
  %tmp1 = load i32, i32* %h, align 4
  %cmp.48 = icmp sgt i32 %tmp1, 0
  %tmp2 = load %struct.AVFrame.5.27.863.1193.1281.1303.1572*, %struct.AVFrame.5.27.863.1193.1281.1303.1572** %outpicref, align 4
  %arrayidx11 = getelementptr inbounds %struct.AVFrame.5.27.863.1193.1281.1303.1572, %struct.AVFrame.5.27.863.1193.1281.1303.1572* %tmp2, i32 0, i32 0, i32 0
  %tmp3 = load i8*, i8** %arrayidx11, align 4
  br label %for.body.14.lr.ph

for.body.14.lr.ph:                                ; preds = %for.end, %for.cond.12.preheader.lr.ph
  %d.050 = phi i8* [ %tmp3, %for.cond.12.preheader.lr.ph ], [ undef, %for.end ]
  %w = getelementptr inbounds %struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573, %struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573* %s, i32 0, i32 2
  %tmp4 = load i32, i32* %w, align 4
  %cmp13.46 = icmp sgt i32 %tmp4, 0
  br label %for.body.14

for.body.14:                                      ; preds = %for.body.14, %for.body.14.lr.ph
  %arrayidx30 = getelementptr inbounds i8, i8* %d.050, i32 0
  store i8 undef, i8* %arrayidx30, align 1
  %arrayidx54 = getelementptr inbounds %struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573, %struct.AudioVectorScopeContext.21.43.879.1209.1297.1319.1573* %s, i32 0, i32 8, i32 2
  %tmp5 = load i32, i32* %arrayidx54, align 4
  %add92 = add nuw nsw i32 0, 4
  %tmp6 = load i32, i32* %w, align 4
  %mul = shl nsw i32 %tmp6, 2
  %cmp13 = icmp slt i32 %add92, %mul
  br i1 %cmp13, label %for.body.14, label %for.end

for.end:                                          ; preds = %for.body.14
  %inc = add nuw nsw i32 0, 1
  %tmp7 = load i32, i32* %h, align 4
  %cmp = icmp slt i32 %inc, %tmp7
  br i1 %cmp, label %for.body.14.lr.ph, label %if.end.loopexit

if.end.loopexit:                                  ; preds = %for.end
  br label %if.end

if.end:                                           ; preds = %if.end.loopexit
  ret void
}
