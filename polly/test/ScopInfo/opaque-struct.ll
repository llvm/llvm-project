; RUN: opt %loadPolly -polly-scops -disable-output < %s
;
; Check that we do not crash with unsized (opaque) types.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.s3 = type opaque
%struct.stmt = type opaque

; Function Attrs: nounwind uwtable
define void @columnMem(%struct.stmt* %pStmt) #0 {
entry:
  br label %if.else

if.else:                                          ; preds = %entry
  %db = bitcast %struct.stmt* %pStmt to %struct.s3**
  %0 = load %struct.s3*, %struct.s3** %db, align 8
  br i1 false, label %if.end9, label %if.then7

if.then7:                                         ; preds = %if.else
  br label %if.end9

if.end9:                                          ; preds = %if.then7, %if.else
  ret void
}
