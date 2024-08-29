; RUN: opt %loadNPMPolly -polly-process-unprofitable -passes=polly-codegen -S < %s | FileCheck %s
;
; llvm.org/PR25412
; %synthgep caused %gep to be synthesized in subregion_if which was reused for
; %retval in subregion_exit, even though it is not dominating subregion_exit.
;
; CHECK-LABEL: polly.stmt.polly.merge_new_and_old.exit:
; CHECK:         %scevgep[[R1:[0-9]*]] = getelementptr i8, ptr %arg, i64 16
; CHECK:         store ptr %scevgep[[R1]], ptr %gep.s2a
; CHECK:         br label

%struct.hoge = type { double, double, double }

define double @func(ptr %arg) {
entry:
  br label %subregion_entry

subregion_entry:
  %gep = getelementptr inbounds %struct.hoge, ptr %arg, i64 0, i32 2
  %cond = fcmp ogt double undef, undef
  br i1 %cond, label %subregion_if, label %subregion_exit

subregion_if:
  %synthgep = load double, ptr %gep
  br label %subregion_exit

subregion_exit:
  %retval = load double, ptr %gep
  ret double %retval
}
