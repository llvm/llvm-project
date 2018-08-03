; RUN: llc -march=hexagon < %s | FileCheck %s

; Test that the store widening optimization correctly transforms to a wider
; instruction with a sub register. Recently, the store widening occurs in the
; DAG combiner, so this test doesn't fail any more.

; CHECK: memh({{r[0-9]+}}+#{{[0-9]+}}) =

%s.0 = type { %s.1, %s.2, %s.3*, %s.0*, i32, i8, i8, i32, i8, i8, i32, i32, i8, i32, %s.4*, [2 x %s.4*], %s.13, i8*, %s.15*, %s.26, i32, i32, i32 }
%s.1 = type { i64, [8 x i8] }
%s.2 = type { i64*, i32, i8 }
%s.3 = type { %s.1, %s.26, %s.26, i32, i32, i32, void (%s.1*)*, void (%s.1*)*, i32 (%s.1*)*, void (%s.1*)*, i32, i64* }
%s.4 = type { %s.5, %s.12 }
%s.5 = type { i32, i32, i32, i32, i32, i32, i32, i32, %s.6 }
%s.6 = type { %s.7 }
%s.7 = type { i32, i32, %s.8, %s.9, i32, [4 x %s.10], %s.11 }
%s.8 = type { i32, i32, i32 }
%s.9 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%s.10 = type { i32, i32 }
%s.11 = type { i32, i32, i32, i32, i32, i32, i32, i32 }
%s.12 = type { i32, i32, i32, i32, i32, i32, i32 }
%s.13 = type { i32, i32, i32, %s.14*, %s.14*, %s.14*, %s.14*, i32 }
%s.14 = type { %s.14*, i8, i32, %s.4*, i32, %s.4* }
%s.15 = type { %s.16, %s.17, %s.19, %s.20, %s.21, %s.24 }
%s.16 = type { i64, i64, i64, i32 }
%s.17 = type { i16, i16, i8, [4 x %s.18], i8, i8 }
%s.18 = type { i32, i32 }
%s.19 = type { i32*, i32, i32* }
%s.20 = type { i8, i8, i32, i32, i8, i32, i32, i32, i32, i32 }
%s.21 = type { i32, %s.22 }
%s.22 = type { %s.23 }
%s.23 = type { i32, i32, i32, i32, i32, i32, i32 }
%s.24 = type { %s.25 }
%s.25 = type { i32, i32, i32, i32, i32, i32, i32 }
%s.26 = type { %s.27 }
%s.27 = type { i16, i16, i32, i32, i32 }

; Function Attrs: nounwind
define void @f0(i64* %a0, i1 %a1) #0 {
b0:
  %v0 = load i64, i64* %a0, align 8
  br i1 %a1, label %b1, label %b2

b1:                                               ; preds = %b0
  %v1 = trunc i64 %v0 to i32
  %v2 = inttoptr i32 %v1 to %s.0*
  %v3 = getelementptr inbounds %s.0, %s.0* %v2, i32 0, i32 8
  store i8 0, i8* %v3, align 8
  %v4 = getelementptr inbounds %s.0, %s.0* %v2, i32 0, i32 9
  store i8 1, i8* %v4, align 1
  %v5 = getelementptr inbounds %s.0, %s.0* %v2, i32 0, i32 6
  store i8 1, i8* %v5, align 1
  ret void

b2:                                               ; preds = %b0
  ret void
}

attributes #0 = { nounwind }
