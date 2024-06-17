; Test that instrumentation counter promotion for loops does not fail during
; compilation for loops that exit to a catchswitch block. In this case, counters
; do not get promoted out of the loop body.

; RUN: opt < %s -passes=pgo-instr-gen,instrprof -pgo-instrument-entry=false -do-counter-promotion=true -S | FileCheck %s --check-prefixes=CHECK,NOTENTRY
; RUN: opt < %s -passes=pgo-instr-gen,instrprof -pgo-instrument-entry=true -do-counter-promotion=true -S | FileCheck %s --check-prefixes=CHECK,ENTRY

; Source used to create test:
;
; extern void may_throw(int);
; char buffer[200];
; void run(int count) {
;   try {
;    for (int i = 0; i < count; ++i) {
;      if (buffer[i] == 0)
;        break;
;      may_throw(i);
;    }
;  }
;  catch (...) {
;     throw;
;  }
;}

%eh.ThrowInfo = type { i32, i32, i32, i32 }

@"?buffer@@3PADA" = dso_local local_unnamed_addr global [200 x i8] zeroinitializer, align 16
define dso_local void @"?run@@YAXH@Z"(i32 %count) local_unnamed_addr personality ptr @__CxxFrameHandler3 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %count
  br i1 %cmp, label %for.body, label %cleanup

for.body:                                         ; preds = %for.cond
; CHECK: for.body:
; NOTENTRY: %pgocount1 = load i64, ptr @"__profc_?run@@YAXH@Z"
; ENTRY: %pgocount1 = load i64, ptr getelementptr inbounds ([3 x i64], ptr @"__profc_?run@@YAXH@Z", i32 0, i32 1)
; CHECK: %1 = add i64 %pgocount1, 1
; NOTENTRY: store i64 %1, ptr @"__profc_?run@@YAXH@Z"
; ENTRY: store i64 %1, ptr getelementptr inbounds ([3 x i64], ptr @"__profc_?run@@YAXH@Z", i32 0, i32 1)
  %idxprom = zext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds [200 x i8], ptr @"?buffer@@3PADA", i64 0, i64 %idxprom
  %0 = load i8, ptr %arrayidx, align 1
  %cmp1 = icmp eq i8 %0, 0
  br i1 %cmp1, label %cleanup, label %if.end

if.end:                                           ; preds = %for.body
  invoke void @"?may_throw@@YAXH@Z"(i32 %i.0)
          to label %for.inc unwind label %catch.dispatch

for.inc:                                          ; preds = %if.end
; CHECK: for.inc:
; NOTENTRY: %pgocount2 = load i64, ptr getelementptr inbounds ([3 x i64], ptr @"__profc_?run@@YAXH@Z", i32 0, i32 1)
; ENTRY: %pgocount2 = load i64, ptr getelementptr inbounds ([3 x i64], ptr @"__profc_?run@@YAXH@Z", i32 0, i32 2)
; CHECK: %3 = add i64 %pgocount2, 1
; NOTENTRY: store i64 %3, ptr getelementptr inbounds ([3 x i64], ptr @"__profc_?run@@YAXH@Z", i32 0, i32 1)
; ENTRY: store i64 %3, ptr getelementptr inbounds ([3 x i64], ptr @"__profc_?run@@YAXH@Z", i32 0, i32 2)
  %inc = add nuw nsw i32 %i.0, 1
  br label %for.cond

cleanup:                                          ; preds = %for.body, %for.cond
  ret void

catch.dispatch:                                   ; preds = %if.end
  %1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %2 = catchpad within %1 [ptr null, i32 64, ptr null]
  call void @_CxxThrowException(ptr null, ptr null) #2 [ "funclet"(token %2) ]
  unreachable
}
declare dso_local void @"?may_throw@@YAXH@Z"(i32)
declare dso_local void @_CxxThrowException(ptr, ptr)
declare dso_local i32 @__CxxFrameHandler3(...)
