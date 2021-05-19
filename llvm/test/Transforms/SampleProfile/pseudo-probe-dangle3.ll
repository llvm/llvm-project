; Test after FoldValueComparisonIntoPredecessors, one dangling probe is generated
; RUN: opt -S -passes='pseudo-probe,simplifycfg' < %s | FileCheck %s


; CHECK: if.end80:                                         ; preds = %if.end
; CHECK-NEXT:   call void @llvm.pseudoprobe(i64 -2281696412744416103, i64 3, i32 0, i64 -1)
; CHECK-NEXT:   call void @llvm.pseudoprobe(i64 -2281696412744416103, i64 4, i32 2, i64 -1)

define dso_local i32 @readCBPandCoeffsFromNAL(i1 %c, i32 %x, i32 %y) local_unnamed_addr {
;
if.end:
  br i1 %c, label %if.end80, label %if.then64

if.then64:                                        ; preds = %if.end
  ret i32 %y

if.end80:                                         ; preds = %if.end
  switch i32 %x, label %lor.lhs.false89 [
  i32 10, label %if.end172237
  i32 14, label %if.end172237
  i32 9, label %if.end172
  ]

lor.lhs.false89:                                  ; preds = %lor.lhs.false89, %if.end80
  %cmp91 = icmp eq i32 %x, 12
  br i1 %cmp91, label %if.end172, label %lor.lhs.false89

if.end172:                                        ; preds = %lor.lhs.false89, %if.end80
  br label %if.end239

if.end172237:                                     ; preds = %if.end80, %if.end80
  br label %if.end239

if.end239:                                        ; preds = %if.end172237, %if.end172
  %cbp.0 = phi i32 [ 1, %if.end172237 ], [ 0, %if.end172 ]
  ret i32 %cbp.0
}
