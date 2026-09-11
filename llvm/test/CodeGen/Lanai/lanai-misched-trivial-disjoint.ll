; REQUIRES: asserts
; RUN: llc %s -mtriple=lanai-unknown-unknown -debug-only=machine-scheduler -o /dev/null 2>&1 | FileCheck %s

; Make sure there are no control dependencies between memory operations that
; are trivially disjoint.

; Function Attrs: norecurse nounwind uwtable
define i32 @foo(ptr inreg nocapture %x) {
entry:
  %0 = bitcast ptr %x to ptr
  store i32 1, ptr %0, align 4
  %arrayidx1 = getelementptr inbounds i8, ptr %x, i32 4
  %1 = bitcast ptr %arrayidx1 to ptr
  store i32 2, ptr %1, align 4
  %arrayidx2 = getelementptr inbounds i8, ptr %x, i32 12
  %2 = bitcast ptr %arrayidx2 to ptr
  %3 = load i32, ptr %2, align 4
  %arrayidx3 = getelementptr inbounds i8, ptr %x, i32 10
  %4 = bitcast ptr %arrayidx3 to ptr
  store i16 3, ptr %4, align 2
  %5 = bitcast ptr %arrayidx2 to ptr
  store i16 4, ptr %5, align 2
  %arrayidx5 = getelementptr inbounds i8, ptr %x, i32 14
  store i8 5, ptr %arrayidx5, align 1
  %arrayidx6 = getelementptr inbounds i8, ptr %x, i32 15
  store i8 6, ptr %arrayidx6, align 1
  %arrayidx7 = getelementptr inbounds i8, ptr %x, i32 16
  store i8 7, ptr %arrayidx7, align 1
  ret i32 %3
}

; CHECK-LABEL: foo
; CHECK-LABEL: SU({{.*}}):   SW_RI{{.*}}, 0,
; CHECK:  # preds left       : 2
; CHECK:  # succs left       : 0
; CHECK-LABEL: SU({{.*}}):   SW_RI{{.*}}, 4,
; CHECK:  # preds left       : 2
; CHECK:  # succs left       : 0
; CHECK-LABEL: SU({{.*}}):   %{{.*}} = LDW_RI{{.*}}, 12,
; CHECK:  # preds left       : 1
; CHECK:  # succs left       : 4
; CHECK-LABEL: SU({{.*}}):   STH_RI{{.*}}, 10,
; CHECK:  # preds left       : 2
; CHECK:  # succs left       : 0
; CHECK-LABEL: SU({{.*}}):   STH_RI{{.*}}, 12,
; CHECK:  # preds left       : 3
; CHECK:  # succs left       : 0
; CHECK-LABEL: SU({{.*}}):   STB_RI{{.*}}, 14,
; CHECK:  # preds left       : 3
; CHECK:  # succs left       : 0
; CHECK-LABEL: SU({{.*}}):   STB_RI{{.*}}, 15,
; CHECK:  # preds left       : 3
; CHECK:  # succs left       : 0
; CHECK-LABEL: SU({{.*}}):   STB_RI{{.*}}, 16,
; CHECK:  # preds left       : 2
; CHECK:  # succs left       : 0
