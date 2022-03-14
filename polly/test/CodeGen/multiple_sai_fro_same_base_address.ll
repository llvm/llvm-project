; RUN: opt %loadPolly -polly-position=before-vectorizer -polly-print-scops -disable-output < %s | FileCheck %s --check-prefix=SCOP
; RUN: opt %loadPolly -polly-position=before-vectorizer -polly-codegen -S < %s | FileCheck %s --check-prefix=IR

; The IR has two ScopArrayInfo for the value %next.0. This used to produce two
; phi nodes in polly.merge_new_and_old, one illegaly using the result of the
; other. There must be only one merge phi, no need to generate them for arrays
; of type MK_Array.
; Derived from test-suite/MultiSource/Applications/siod/slib.c

%struct.obj.2.290.322.338.354.482.546.594.626.818.898.914.962 = type { i16, i16, %union.anon.1.289.321.337.353.481.545.593.625.817.897.913.961 }
%union.anon.1.289.321.337.353.481.545.593.625.817.897.913.961 = type { %struct.anon.0.288.320.336.352.480.544.592.624.816.896.912.960 }
%struct.anon.0.288.320.336.352.480.544.592.624.816.896.912.960 = type { %struct.obj.2.290.322.338.354.482.546.594.626.818.898.914.962*, %struct.obj.2.290.322.338.354.482.546.594.626.818.898.914.962* }

define void @leval_or() {
entry:
  br label %while.cond

while.cond:                                       ; preds = %sw.bb1.i30, %cond.end.i28, %entry
  %next.0 = phi %struct.obj.2.290.322.338.354.482.546.594.626.818.898.914.962* [ null, %entry ], [ %1, %sw.bb1.i30 ], [ null, %cond.end.i28 ]
  br i1 undef, label %cond.end.i28, label %if.then

if.then:                                          ; preds = %while.cond
  ret void

cond.end.i28:                                     ; preds = %while.cond
  %type.i24 = getelementptr inbounds %struct.obj.2.290.322.338.354.482.546.594.626.818.898.914.962, %struct.obj.2.290.322.338.354.482.546.594.626.818.898.914.962* %next.0, i64 0, i32 1
  %0 = load i16, i16* %type.i24, align 2
  br i1 false, label %sw.bb1.i30, label %while.cond

sw.bb1.i30:                                       ; preds = %cond.end.i28
  %cdr.i29 = getelementptr inbounds %struct.obj.2.290.322.338.354.482.546.594.626.818.898.914.962, %struct.obj.2.290.322.338.354.482.546.594.626.818.898.914.962* %next.0, i64 0, i32 2, i32 0, i32 1
  %1 = load %struct.obj.2.290.322.338.354.482.546.594.626.818.898.914.962*, %struct.obj.2.290.322.338.354.482.546.594.626.818.898.914.962** %cdr.i29, align 8
  br label %while.cond
}

; SCOP:      Arrays {
; SCOP-NEXT:     %struct.obj.2.290.322.338.354.482.546.594.626.818.898.914.962* MemRef_next_0;
; SCOP-NEXT:     i16 MemRef_next_0[*];
; SCOP-NEXT: }

; IR:      polly.merge_new_and_old:
; IR-NEXT:   %next.0.ph.merge = phi %struct.obj.2.290.322.338.354.482.546.594.626.818.898.914.962* [ %next.0.ph.final_reload, %polly.exiting ], [ %next.0.ph, %while.cond.region_exiting ]
; IR-NEXT:   %indvar.next = add i64 %indvar, 1
; IR-NEXT:   br label %while.cond
