; RUN: opt < %s -basic-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct = type { i32, i32, i32 }

; CHECK-LABEL: test_simple

; CHECK-DAG: MayAlias: %struct* %st, i32* %x
; CHECK-DAG: MayAlias: %struct* %st, i32* %y
; CHECK-DAG: MayAlias: %struct* %st, i32* %z

; CHECK-DAG: NoAlias: i32* %x, i32* %y
; CHECK-DAG: NoAlias: i32* %x, i32* %z
; CHECK-DAG: NoAlias: i32* %y, i32* %z

; CHECK-DAG: MayAlias: %struct* %st, %struct* %y_12
; CHECK-DAG: MayAlias: %struct* %y_12, i32* %x
; CHECK-DAG: MayAlias: i32* %x, i80* %y_10

; CHECK-DAG: MayAlias: %struct* %st, i64* %y_8
; CHECK-DAG: MayAlias: i32* %z, i64* %y_8
; CHECK-DAG: NoAlias: i32* %x, i64* %y_8

; CHECK-DAG: MustAlias: %struct* %y_12, i32* %y
; CHECK-DAG: MustAlias: i32* %y, i64* %y_8
; CHECK-DAG: MustAlias: i32* %y, i80* %y_10

define void @test_simple(%struct* %st, i64 %i, i64 %j, i64 %k) {
  %x = getelementptr %struct, %struct* %st, i64 %i, i32 0
  %y = getelementptr %struct, %struct* %st, i64 %j, i32 1
  %z = getelementptr %struct, %struct* %st, i64 %k, i32 2
  %y_12 = bitcast i32* %y to %struct*
  %y_10 = bitcast i32* %y to i80*
  %y_8 = bitcast i32* %y to i64*
  ret void
}

; CHECK-LABEL: test_in_array

; CHECK-DAG: MayAlias: [1 x %struct]* %st, i32* %x
; CHECK-DAG: MayAlias: [1 x %struct]* %st, i32* %y
; CHECK-DAG: MayAlias: [1 x %struct]* %st, i32* %z

; CHECK-DAG: NoAlias: i32* %x, i32* %y
; CHECK-DAG: NoAlias: i32* %x, i32* %z
; CHECK-DAG: NoAlias: i32* %y, i32* %z

; CHECK-DAG: MayAlias: %struct* %y_12, [1 x %struct]* %st
; CHECK-DAG: MayAlias: %struct* %y_12, i32* %x
; CHECK-DAG: MayAlias: i32* %x, i80* %y_10

; CHECK-DAG: MayAlias: [1 x %struct]* %st, i64* %y_8
; CHECK-DAG: MayAlias: i32* %z, i64* %y_8
; CHECK-DAG: NoAlias: i32* %x, i64* %y_8

; CHECK-DAG: MustAlias: %struct* %y_12, i32* %y
; CHECK-DAG: MustAlias: i32* %y, i64* %y_8
; CHECK-DAG: MustAlias: i32* %y, i80* %y_10

define void @test_in_array([1 x %struct]* %st, i64 %i, i64 %j, i64 %k, i64 %i1, i64 %j1, i64 %k1) {
  %x = getelementptr [1 x %struct], [1 x %struct]* %st, i64 %i, i64 %i1, i32 0
  %y = getelementptr [1 x %struct], [1 x %struct]* %st, i64 %j, i64 %j1, i32 1
  %z = getelementptr [1 x %struct], [1 x %struct]* %st, i64 %k, i64 %k1, i32 2
  %y_12 = bitcast i32* %y to %struct*
  %y_10 = bitcast i32* %y to i80*
  %y_8 = bitcast i32* %y to i64*
  ret void
}

; CHECK-LABEL: test_in_3d_array

; CHECK-DAG: MayAlias: [1 x [1 x [1 x %struct]]]* %st, i32* %x
; CHECK-DAG: MayAlias: [1 x [1 x [1 x %struct]]]* %st, i32* %y
; CHECK-DAG: MayAlias: [1 x [1 x [1 x %struct]]]* %st, i32* %z

; CHECK-DAG: NoAlias: i32* %x, i32* %y
; CHECK-DAG: NoAlias: i32* %x, i32* %z
; CHECK-DAG: NoAlias: i32* %y, i32* %z

; CHECK-DAG: MayAlias: %struct* %y_12, [1 x [1 x [1 x %struct]]]* %st
; CHECK-DAG: MayAlias: %struct* %y_12, i32* %x
; CHECK-DAG: MayAlias: i32* %x, i80* %y_10

; CHECK-DAG: MayAlias: [1 x [1 x [1 x %struct]]]* %st, i64* %y_8
; CHECK-DAG: MayAlias: i32* %z, i64* %y_8
; CHECK-DAG: NoAlias: i32* %x, i64* %y_8

; CHECK-DAG: MustAlias: %struct* %y_12, i32* %y
; CHECK-DAG: MustAlias: i32* %y, i64* %y_8
; CHECK-DAG: MustAlias: i32* %y, i80* %y_10

define void @test_in_3d_array([1 x [1 x [1 x %struct]]]* %st, i64 %i, i64 %j, i64 %k, i64 %i1, i64 %j1, i64 %k1, i64 %i2, i64 %j2, i64 %k2, i64 %i3, i64 %j3, i64 %k3) {
  %x = getelementptr [1 x [1 x [1 x %struct]]], [1 x [1 x [1 x %struct]]]* %st, i64 %i, i64 %i1, i64 %i2, i64 %i3, i32 0
  %y = getelementptr [1 x [1 x [1 x %struct]]], [1 x [1 x [1 x %struct]]]* %st, i64 %j, i64 %j1, i64 %j2, i64 %j3, i32 1
  %z = getelementptr [1 x [1 x [1 x %struct]]], [1 x [1 x [1 x %struct]]]* %st, i64 %k, i64 %k1, i64 %k2, i64 %k3, i32 2
  %y_12 = bitcast i32* %y to %struct*
  %y_10 = bitcast i32* %y to i80*
  %y_8 = bitcast i32* %y to i64*
  ret void
}

; CHECK-LABEL: test_same_underlying_object_same_indices

; CHECK-DAG: NoAlias: i32* %x, i32* %x2
; CHECK-DAG: NoAlias: i32* %y, i32* %y2
; CHECK-DAG: NoAlias: i32* %z, i32* %z2

; CHECK-DAG: NoAlias: i32* %x, i32* %y2
; CHECK-DAG: NoAlias: i32* %x, i32* %z2

; CHECK-DAG: NoAlias: i32* %x2, i32* %y
; CHECK-DAG: NoAlias: i32* %y, i32* %z2

; CHECK-DAG: NoAlias: i32* %x2, i32* %z
; CHECK-DAG: NoAlias: i32* %y2, i32* %z

define void @test_same_underlying_object_same_indices(%struct* %st, i64 %i, i64 %j, i64 %k) {
  %st2 = getelementptr %struct, %struct* %st, i32 10
  %x2 = getelementptr %struct, %struct* %st2, i64 %i, i32 0
  %y2 = getelementptr %struct, %struct* %st2, i64 %j, i32 1
  %z2 = getelementptr %struct, %struct* %st2, i64 %k, i32 2
  %x = getelementptr %struct, %struct* %st, i64 %i, i32 0
  %y = getelementptr %struct, %struct* %st, i64 %j, i32 1
  %z = getelementptr %struct, %struct* %st, i64 %k, i32 2
  ret void
}

; CHECK-LABEL: test_same_underlying_object_different_indices

; CHECK-DAG: MayAlias: i32* %x, i32* %x2
; CHECK-DAG: MayAlias: i32* %y, i32* %y2
; CHECK-DAG: MayAlias: i32* %z, i32* %z2

; CHECK-DAG: NoAlias: i32* %x, i32* %y2
; CHECK-DAG: NoAlias: i32* %x, i32* %z2

; CHECK-DAG: NoAlias: i32* %x2, i32* %y
; CHECK-DAG: NoAlias: i32* %y, i32* %z2

; CHECK-DAG: NoAlias: i32* %x2, i32* %z
; CHECK-DAG: NoAlias: i32* %y2, i32* %z

define void @test_same_underlying_object_different_indices(%struct* %st, i64 %i1, i64 %j1, i64 %k1, i64 %i2, i64 %k2, i64 %j2) {
  %st2 = getelementptr %struct, %struct* %st, i32 10
  %x2 = getelementptr %struct, %struct* %st2, i64 %i2, i32 0
  %y2 = getelementptr %struct, %struct* %st2, i64 %j2, i32 1
  %z2 = getelementptr %struct, %struct* %st2, i64 %k2, i32 2
  %x = getelementptr %struct, %struct* %st, i64 %i1, i32 0
  %y = getelementptr %struct, %struct* %st, i64 %j1, i32 1
  %z = getelementptr %struct, %struct* %st, i64 %k1, i32 2
  ret void
}


%struct2 = type { [1 x { i32, i32 }], [2 x { i32 }] }

; CHECK-LABEL: test_struct_in_array
; CHECK-DAG: MustAlias: i32* %x, i32* %y
define void @test_struct_in_array(%struct2* %st, i64 %i, i64 %j, i64 %k) {
  %x = getelementptr %struct2, %struct2* %st, i32 0, i32 1, i32 1, i32 0
  %y = getelementptr %struct2, %struct2* %st, i32 0, i32 0, i32 1, i32 1
  ret void
}

; PR27418 - Treat GEP indices with the same value but different types the same
; CHECK-LABEL: test_different_index_types
; CHECK: MustAlias: i16* %tmp1, i16* %tmp2
define void @test_different_index_types([2 x i16]* %arr) {
  %tmp1 = getelementptr [2 x i16], [2 x i16]* %arr, i16 0, i32 1
  %tmp2 = getelementptr [2 x i16], [2 x i16]* %arr, i16 0, i16 1
  ret void
}
