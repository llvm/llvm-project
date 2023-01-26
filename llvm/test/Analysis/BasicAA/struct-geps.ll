; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct = type { i32, i32, i32 }

; CHECK-LABEL: test_simple

; CHECK-DAG: MayAlias: %struct* %st, i32* %x
; CHECK-DAG: MayAlias: %struct* %st, i32* %y
; CHECK-DAG: MayAlias: %struct* %st, i32* %z

; CHECK-DAG: NoAlias: i32* %x, i32* %y
; CHECK-DAG: NoAlias: i32* %x, i32* %z
; CHECK-DAG: NoAlias: i32* %y, i32* %z

; CHECK-DAG: MayAlias: %struct* %st, %struct* %y
; CHECK-DAG: MayAlias: i32* %x, %struct* %y
; CHECK-DAG: MayAlias: i32* %x, i80* %y

; CHECK-DAG: MayAlias: %struct* %st, i64* %y
; CHECK-DAG: MayAlias: i64* %y, i32* %z
; CHECK-DAG: NoAlias: i32* %x, i64* %y

; CHECK-DAG: MustAlias: %struct* %y, i32* %y
; CHECK-DAG: MustAlias: i64* %y, i32* %y
; CHECK-DAG: MustAlias: i80* %y, i32* %y

define void @test_simple(ptr %st, i64 %i, i64 %j, i64 %k) {
  %x = getelementptr inbounds %struct, ptr %st, i64 %i, i32 0
  %y = getelementptr inbounds %struct, ptr %st, i64 %j, i32 1
  %z = getelementptr inbounds %struct, ptr %st, i64 %k, i32 2
  load %struct, ptr %st
  load i32, ptr %x
  load i32, ptr %y
  load i32, ptr %z
  load %struct, ptr %y
  load i80, ptr %y
  load i64, ptr %y
  ret void
}

; As the GEP is not inbounds, these pointers may alias due to overflow.
; CHECK-LABEL: test_not_inbounds
; CHECK: MayAlias: i32* %x, i32* %y
define void @test_not_inbounds(ptr %st, i64 %i, i64 %j, i64 %k) {
  %x = getelementptr %struct, ptr %st, i64 %i, i32 0
  %y = getelementptr %struct, ptr %st, i64 %j, i32 1
  load i32, ptr %x
  load i32, ptr %y
  ret void
}

; CHECK-LABEL: test_in_array

; CHECK-DAG: MayAlias: [1 x %struct]* %st, i32* %x
; CHECK-DAG: MayAlias: [1 x %struct]* %st, i32* %y
; CHECK-DAG: MayAlias: [1 x %struct]* %st, i32* %z

; CHECK-DAG: NoAlias: i32* %x, i32* %y
; CHECK-DAG: NoAlias: i32* %x, i32* %z
; CHECK-DAG: NoAlias: i32* %y, i32* %z

; CHECK-DAG: MayAlias: [1 x %struct]* %st, %struct* %y
; CHECK-DAG: MayAlias: i32* %x, %struct* %y
; CHECK-DAG: MayAlias: i32* %x, i80* %y

; CHECK-DAG: MayAlias: [1 x %struct]* %st, i64* %y
; CHECK-DAG: MayAlias: i64* %y, i32* %z
; CHECK-DAG: NoAlias: i32* %x, i64* %y

; CHECK-DAG: MustAlias: %struct* %y, i32* %y
; CHECK-DAG: MustAlias: i64* %y, i32* %y
; CHECK-DAG: MustAlias: i80* %y, i32* %y

define void @test_in_array(ptr %st, i64 %i, i64 %j, i64 %k, i64 %i1, i64 %j1, i64 %k1) {
  %x = getelementptr inbounds [1 x %struct], ptr %st, i64 %i, i64 %i1, i32 0
  %y = getelementptr inbounds [1 x %struct], ptr %st, i64 %j, i64 %j1, i32 1
  %z = getelementptr inbounds [1 x %struct], ptr %st, i64 %k, i64 %k1, i32 2
  load [1 x %struct], ptr %st
  load i32, ptr %x
  load i32, ptr %y
  load i32, ptr %z
  load %struct, ptr %y
  load i80, ptr %y
  load i64, ptr %y
  ret void
}

; CHECK-LABEL: test_in_3d_array

; CHECK-DAG: MayAlias: [1 x [1 x [1 x %struct]]]* %st, i32* %x
; CHECK-DAG: MayAlias: [1 x [1 x [1 x %struct]]]* %st, i32* %y
; CHECK-DAG: MayAlias: [1 x [1 x [1 x %struct]]]* %st, i32* %z

; CHECK-DAG: NoAlias: i32* %x, i32* %y
; CHECK-DAG: NoAlias: i32* %x, i32* %z
; CHECK-DAG: NoAlias: i32* %y, i32* %z

; CHECK-DAG: MayAlias: [1 x [1 x [1 x %struct]]]* %st, %struct* %y
; CHECK-DAG: MayAlias: i32* %x, %struct* %y
; CHECK-DAG: MayAlias: i32* %x, i80* %y

; CHECK-DAG: MayAlias: [1 x [1 x [1 x %struct]]]* %st, i64* %y
; CHECK-DAG: MayAlias: i64* %y, i32* %z
; CHECK-DAG: NoAlias: i32* %x, i64* %y

; CHECK-DAG: MustAlias: %struct* %y, i32* %y
; CHECK-DAG: MustAlias: i64* %y, i32* %y
; CHECK-DAG: MustAlias: i80* %y, i32* %y

define void @test_in_3d_array(ptr %st, i64 %i, i64 %j, i64 %k, i64 %i1, i64 %j1, i64 %k1, i64 %i2, i64 %j2, i64 %k2, i64 %i3, i64 %j3, i64 %k3) {
  %x = getelementptr inbounds [1 x [1 x [1 x %struct]]], ptr %st, i64 %i, i64 %i1, i64 %i2, i64 %i3, i32 0
  %y = getelementptr inbounds [1 x [1 x [1 x %struct]]], ptr %st, i64 %j, i64 %j1, i64 %j2, i64 %j3, i32 1
  %z = getelementptr inbounds [1 x [1 x [1 x %struct]]], ptr %st, i64 %k, i64 %k1, i64 %k2, i64 %k3, i32 2
  load [1 x [1 x [1 x %struct]]], ptr %st
  load i32, ptr %x
  load i32, ptr %y
  load i32, ptr %z
  load %struct, ptr %y
  load i80, ptr %y
  load i64, ptr %y
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

define void @test_same_underlying_object_same_indices(ptr %st, i64 %i, i64 %j, i64 %k) {
  %st2 = getelementptr inbounds %struct, ptr %st, i32 10
  %x2 = getelementptr inbounds %struct, ptr %st2, i64 %i, i32 0
  %y2 = getelementptr inbounds %struct, ptr %st2, i64 %j, i32 1
  %z2 = getelementptr inbounds %struct, ptr %st2, i64 %k, i32 2
  %x = getelementptr inbounds %struct, ptr %st, i64 %i, i32 0
  %y = getelementptr inbounds %struct, ptr %st, i64 %j, i32 1
  %z = getelementptr inbounds %struct, ptr %st, i64 %k, i32 2
  load i32, ptr %x
  load i32, ptr %y
  load i32, ptr %z
  load i32, ptr %x2
  load i32, ptr %y2
  load i32, ptr %z2
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

define void @test_same_underlying_object_different_indices(ptr %st, i64 %i1, i64 %j1, i64 %k1, i64 %i2, i64 %k2, i64 %j2) {
  %st2 = getelementptr inbounds %struct, ptr %st, i32 10
  %x2 = getelementptr inbounds %struct, ptr %st2, i64 %i2, i32 0
  %y2 = getelementptr inbounds %struct, ptr %st2, i64 %j2, i32 1
  %z2 = getelementptr inbounds %struct, ptr %st2, i64 %k2, i32 2
  %x = getelementptr inbounds %struct, ptr %st, i64 %i1, i32 0
  %y = getelementptr inbounds %struct, ptr %st, i64 %j1, i32 1
  %z = getelementptr inbounds %struct, ptr %st, i64 %k1, i32 2
  load i32, ptr %x
  load i32, ptr %y
  load i32, ptr %z
  load i32, ptr %x2
  load i32, ptr %y2
  load i32, ptr %z2
  ret void
}


%struct2 = type { [1 x { i32, i32 }], [2 x { i32 }] }

; CHECK-LABEL: test_struct_in_array
; CHECK-DAG: MustAlias: i32* %x, i32* %y
define void @test_struct_in_array(ptr %st, i64 %i, i64 %j, i64 %k) {
  %x = getelementptr inbounds %struct2, ptr %st, i32 0, i32 1, i32 1, i32 0
  %y = getelementptr inbounds %struct2, ptr %st, i32 0, i32 0, i32 1, i32 1
  load i32, ptr %x
  load i32, ptr %y
  ret void
}

; PR27418 - Treat GEP indices with the same value but different types the same
; CHECK-LABEL: test_different_index_types
; CHECK: MustAlias: i16* %tmp1, i16* %tmp2
define void @test_different_index_types(ptr %arr) {
  %tmp1 = getelementptr inbounds [2 x i16], ptr %arr, i16 0, i32 1
  %tmp2 = getelementptr inbounds [2 x i16], ptr %arr, i16 0, i16 1
  load i16, ptr %tmp1
  load i16, ptr %tmp2
  ret void
}
