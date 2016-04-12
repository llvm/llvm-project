; RUN: opt %loadPolly -analyze -polly-scops < %s | FileCheck %s
; RUN: opt %loadPolly -analyze -polly-scops -pass-remarks-analysis="polly-scops" 2>&1 < %s | FileCheck %s --check-prefix=REMARK
;
; This test case has a non-affine access (the memset call) that aliases with
; other accesses. Thus, we bail out.
;
; CHECK-NOT: Statements
;
; REMARK:        remark: <unknown>:0:0: SCoP begins here.
; REMARK-NEXT:   remark: <unknown>:0:0: Possibly aliasing pointer, use restrict keyword.
; REMARK-NEXT:   remark: <unknown>:0:0: Possibly aliasing pointer, use restrict keyword.
; REMARK-NEXT:   remark: <unknown>:0:0: No-aliasing assumption:	{  : 1 = 0 }
; REMARK-NEXT:   remark: <unknown>:0:0: SCoP ends here but was dismissed.
;
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.info = type { i32, %struct.ctr*, i32, %struct.ord*, %struct.ctr*, i32, i8*, i32, i32, double }
%struct.ctr = type { i32, i8, i8, i32 }
%struct.ord = type { i32, i8 }

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) #0

; Function Attrs: nounwind uwtable
define void @bestVirtualIndex(%struct.info** %ppIdxInfo) {
entry:
  %0 = load %struct.info*, %struct.info** %ppIdxInfo, align 8
  br label %if.end125

if.end125:                                        ; preds = %entry
  %1 = load %struct.ctr*, %struct.ctr** undef, align 8
  br label %for.end143

for.end143:                                       ; preds = %if.end125
  %2 = bitcast %struct.ctr* %1 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %2, i8 0, i64 32, i32 4, i1 false)
  %needToFreeIdxStr = getelementptr inbounds %struct.info, %struct.info* %0, i64 0, i32 7
  %3 = load i32, i32* %needToFreeIdxStr, align 8
  br i1 false, label %if.end149, label %if.then148

if.then148:                                       ; preds = %for.end143
  br label %if.end149

if.end149:                                        ; preds = %if.then148, %for.end143
  unreachable
}

attributes #0 = { argmemonly nounwind }
