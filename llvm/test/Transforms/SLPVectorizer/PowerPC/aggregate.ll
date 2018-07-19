; RUN: opt -S -mtriple=powerpc64-linux-gnu -mcpu=pwr9 -mattr=+vsx -slp-vectorizer < %s | FileCheck %s

%struct.S = type { i8*, i8* }

@kS0 = common global %struct.S zeroinitializer, align 8

define { i64, i64 } @getS() {
entry:
  %0 = load i64, i64* bitcast (%struct.S* @kS0 to i64*), align 8
  %1 = load i64, i64* bitcast (i8** getelementptr inbounds (%struct.S, %struct.S* @kS0, i64 0, i32 1) to i64*), align 8
  %2 = insertvalue { i64, i64 } undef, i64 %0, 0
  %3 = insertvalue { i64, i64 } %2, i64 %1, 1
  ret { i64, i64 } %3
}

; CHECK: load i64
; CHECK-NOT: load <2 x i64>
; CHECK-NOT: extractelement

