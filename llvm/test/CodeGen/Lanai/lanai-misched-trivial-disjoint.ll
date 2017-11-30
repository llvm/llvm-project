; REQUIRES: asserts
; RUN: llc %s -mtriple=lanai-unknown-unknown -debug-only=machine-scheduler -o /dev/null 2>&1 | FileCheck %s

; Make sure there are no control dependencies between memory operations that
; are trivially disjoint.

; Function Attrs: norecurse nounwind uwtable
define i32 @foo(i8* inreg nocapture %x) {
entry:
  %0 = bitcast i8* %x to i32*
  store i32 1, i32* %0, align 4
  %arrayidx1 = getelementptr inbounds i8, i8* %x, i32 4
  %1 = bitcast i8* %arrayidx1 to i32*
  store i32 2, i32* %1, align 4
  %arrayidx2 = getelementptr inbounds i8, i8* %x, i32 12
  %2 = bitcast i8* %arrayidx2 to i32*
  %3 = load i32, i32* %2, align 4
  %arrayidx3 = getelementptr inbounds i8, i8* %x, i32 10
  %4 = bitcast i8* %arrayidx3 to i16*
  store i16 3, i16* %4, align 2
  %5 = bitcast i8* %arrayidx2 to i16*
  store i16 4, i16* %5, align 2
  %arrayidx5 = getelementptr inbounds i8, i8* %x, i32 14
  store i8 5, i8* %arrayidx5, align 1
  %arrayidx6 = getelementptr inbounds i8, i8* %x, i32 15
  store i8 6, i8* %arrayidx6, align 1
  %arrayidx7 = getelementptr inbounds i8, i8* %x, i32 16
  store i8 7, i8* %arrayidx7, align 1
  ret i32 %3
}

; CHECK-LABEL: foo
; CHECK-LABEL: SU({{.*}}):   SW_RI{{.*}}, 0,
; CHECK:  # preds left       : 2
; CHECK:  # succs left       : 0
; CHECK-LABEL: SU({{.*}}):   SW_RI{{.*}}, 4,
; CHECK:  # preds left       : 2
; CHECK:  # succs left       : 0
; CHECK-LABEL: SU({{.*}}):   %{{.*}}<def> = LDW_RI{{.*}}, 12,
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
