; RUN: opt -passes=consthoist -S -o - %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m-none--musleabi"

; Check that for i8 type, the maximum legal offset is 31.
; Also check that an constant used as value to be stored rather than
; pointer in a store instruction is hoisted.
; CHECK: foo_i8
; CHECK-DAG:  %[[C1:const[0-9]?]] = bitcast i32 805874720 to i32
; CHECK-DAG:  %[[C2:const[0-9]?]] = bitcast i32 805874688 to i32
; CHECK-DAG:  %[[C3:const[0-9]?]] = bitcast i32 805873720 to i32
; CHECK-DAG:  %[[C4:const[0-9]?]] = bitcast i32 805873688 to i32
; CHECK:  %0 = inttoptr i32 %[[C2]] to ptr
; CHECK-NEXT:  %1 = load volatile i8, ptr %0
; CHECK-NEXT:  %[[M1:const_mat[0-9]?]] = add i32 %[[C2]], 4
; CHECK-NEXT:  %2 = inttoptr i32 %[[M1]] to ptr
; CHECK-NEXT:  %3 = load volatile i8, ptr %2
; CHECK-NEXT:  %[[M2:const_mat[0-9]?]] = add i32 %[[C2]], 31
; CHECK-NEXT:  %4 = inttoptr i32 %[[M2]] to ptr
; CHECK-NEXT:  %5 = load volatile i8, ptr %4
; CHECK-NEXT:  %6 = inttoptr i32 %[[C1]] to ptr
; CHECK-NEXT:  %7 = load volatile i8, ptr %6
; CHECK-NEXT:  %[[M3:const_mat[0-9]?]] = add i32 %[[C1]], 7
; CHECK-NEXT:  %8 = inttoptr i32 %[[M3]] to ptr
; CHECK-NEXT:  %9 = load volatile i8, ptr %8
; CHECK-NEXT:  %10 = inttoptr i32 %[[C4]] to ptr
; CHECK-NEXT:  store i8 %9, ptr %10
; CHECK-NEXT:  %[[M4:const_mat[0-9]?]] = add i32 %[[C4]], 31
; CHECK-NEXT:  %11 = inttoptr i32 %[[M4]] to ptr
; CHECK-NEXT:  store i8 %7, ptr %11
; CHECK-NEXT:  %12 = inttoptr i32 %[[C3]] to ptr
; CHECK-NEXT:  store i8 %5, ptr %12
; CHECK-NEXT:  %[[M5:const_mat[0-9]?]] = add i32 %[[C3]], 7
; CHECK-NEXT:  %13 = inttoptr i32 %[[M5]] to ptr
; CHECK-NEXT:  store i8 %3, ptr %13
; CHECK-NEXT:  %[[M6:const_mat[0-9]?]] = add i32 %[[C1]], 80
; CHECK-NEXT:  %14 = inttoptr i32 %[[M6]] to ptr
; CHECK-NEXT:  store ptr %14, ptr @goo

@goo = global ptr undef

define void @foo_i8() {
entry:
  %0 = load volatile i8, ptr inttoptr (i32 805874688 to ptr)
  %1 = load volatile i8, ptr inttoptr (i32 805874692 to ptr)
  %2 = load volatile i8, ptr inttoptr (i32 805874719 to ptr)
  %3 = load volatile i8, ptr inttoptr (i32 805874720 to ptr)
  %4 = load volatile i8, ptr inttoptr (i32 805874727 to ptr)
  store i8 %4, ptr inttoptr(i32 805873688 to ptr)
  store i8 %3, ptr inttoptr(i32 805873719 to ptr)
  store i8 %2, ptr inttoptr(i32 805873720 to ptr)
  store i8 %1, ptr inttoptr(i32 805873727 to ptr)
  store ptr inttoptr(i32 805874800 to ptr), ptr @goo
  ret void
}

; Check that for i16 type, the maximum legal offset is 62.
; CHECK: foo_i16
; CHECK-DAG: %[[C1:const[0-9]?]] = bitcast i32 805874752 to i32
; CHECK-DAG: %[[C2:const[0-9]?]] = bitcast i32 805874688 to i32
; CHECK: %0 = inttoptr i32 %[[C2]] to ptr
; CHECK-NEXT: %1 = load volatile i16, ptr %0, align 2
; CHECK-NEXT: %[[M1:const_mat[0-9]?]] = add i32 %[[C2]], 4
; CHECK-NEXT: %2 = inttoptr i32 %[[M1]] to ptr
; CHECK-NEXT: %3 = load volatile i16, ptr %2, align 2
; CHECK-NEXT: %[[M2:const_mat[0-9]?]] = add i32 %[[C2]], 32
; CHECK-NEXT: %4 = inttoptr i32 %[[M2]] to ptr
; CHECK-NEXT: %5 = load volatile i16, ptr %4, align 2
; CHECK-NEXT: %[[M3:const_mat[0-9]?]] = add i32 %[[C2]], 62
; CHECK-NEXT: %6 = inttoptr i32 %[[M3]] to ptr
; CHECK-NEXT: %7 = load volatile i16, ptr %6, align 2
; CHECK-NEXT: %8 = inttoptr i32 %[[C1]] to ptr
; CHECK-NEXT: %9 = load volatile i16, ptr %8, align 2
; CHECK-NEXT: %[[M4:const_mat[0-9]?]] = add i32 %[[C1]], 22
; CHECK-NEXT: %10 = inttoptr i32 %[[M4]] to ptr
; CHECK-NEXT: %11 = load volatile i16, ptr %10, align 2

define void @foo_i16() {
entry:
  %0 = load volatile i16, ptr inttoptr (i32 805874688 to ptr), align 2
  %1 = load volatile i16, ptr inttoptr (i32 805874692 to ptr), align 2
  %2 = load volatile i16, ptr inttoptr (i32 805874720 to ptr), align 2
  %3 = load volatile i16, ptr inttoptr (i32 805874750 to ptr), align 2
  %4 = load volatile i16, ptr inttoptr (i32 805874752 to ptr), align 2
  %5 = load volatile i16, ptr inttoptr (i32 805874774 to ptr), align 2
  ret void
}

; Check that for i32 type, the maximum legal offset is 124.
; CHECK: foo_i32
; CHECK-DAG:  %[[C1:const[0-9]?]] = bitcast i32 805874816 to i32
; CHECK-DAG:  %[[C2:const[0-9]?]] = bitcast i32 805874688 to i32
; CHECK:  %0 = inttoptr i32 %[[C2]] to ptr
; CHECK-NEXT:  %1 = load volatile i32, ptr %0, align 4
; CHECK-NEXT:  %[[M1:const_mat[0-9]?]] = add i32 %[[C2]], 4
; CHECK-NEXT:  %2 = inttoptr i32 %[[M1]] to ptr
; CHECK-NEXT:  %3 = load volatile i32, ptr %2, align 4
; CHECK-NEXT:  %[[M2:const_mat[0-9]?]] = add i32 %[[C2]], 124
; CHECK-NEXT:  %4 = inttoptr i32 %[[M2]] to ptr
; CHECK-NEXT:  %5 = load volatile i32, ptr %4, align 4
; CHECK-NEXT:  %6 = inttoptr i32 %[[C1]] to ptr
; CHECK-NEXT:  %7 = load volatile i32, ptr %6, align 4
; CHECK-NEXT:  %[[M3:const_mat[0-9]?]] = add i32 %[[C1]], 8
; CHECK-NEXT:  %8 = inttoptr i32 %[[M3]] to ptr
; CHECK-NEXT:  %9 = load volatile i32, ptr %8, align 4
; CHECK-NEXT:  %[[M4:const_mat[0-9]?]] = add i32 %[[C1]], 12
; CHECK-NEXT:  %10 = inttoptr i32 %[[M4]] to ptr
; CHECK-NEXT:  %11 = load volatile i32, ptr %10, align 4

define void @foo_i32() {
entry:
  %0 = load volatile i32, ptr inttoptr (i32 805874688 to ptr), align 4
  %1 = load volatile i32, ptr inttoptr (i32 805874692 to ptr), align 4
  %2 = load volatile i32, ptr inttoptr (i32 805874812 to ptr), align 4
  %3 = load volatile i32, ptr inttoptr (i32 805874816 to ptr), align 4
  %4 = load volatile i32, ptr inttoptr (i32 805874824 to ptr), align 4
  %5 = load volatile i32, ptr inttoptr (i32 805874828 to ptr), align 4
  ret void
}

