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
; CHECK:  %0 = inttoptr i32 %[[C2]] to i8*
; CHECK-NEXT:  %1 = load volatile i8, i8* %0
; CHECK-NEXT:  %[[M1:const_mat[0-9]?]] = add i32 %[[C2]], 4
; CHECK-NEXT:  %2 = inttoptr i32 %[[M1]] to i8*
; CHECK-NEXT:  %3 = load volatile i8, i8* %2
; CHECK-NEXT:  %[[M2:const_mat[0-9]?]] = add i32 %[[C2]], 31
; CHECK-NEXT:  %4 = inttoptr i32 %[[M2]] to i8*
; CHECK-NEXT:  %5 = load volatile i8, i8* %4
; CHECK-NEXT:  %6 = inttoptr i32 %[[C1]] to i8*
; CHECK-NEXT:  %7 = load volatile i8, i8* %6
; CHECK-NEXT:  %[[M3:const_mat[0-9]?]] = add i32 %[[C1]], 7
; CHECK-NEXT:  %8 = inttoptr i32 %[[M3]] to i8*
; CHECK-NEXT:  %9 = load volatile i8, i8* %8
; CHECK-NEXT:  %10 = inttoptr i32 %[[C4]] to i8*
; CHECK-NEXT:  store i8 %9, i8* %10
; CHECK-NEXT:  %[[M4:const_mat[0-9]?]] = add i32 %[[C4]], 31
; CHECK-NEXT:  %11 = inttoptr i32 %[[M4]] to i8*
; CHECK-NEXT:  store i8 %7, i8* %11
; CHECK-NEXT:  %12 = inttoptr i32 %[[C3]] to i8*
; CHECK-NEXT:  store i8 %5, i8* %12
; CHECK-NEXT:  %[[M5:const_mat[0-9]?]] = add i32 %[[C3]], 7
; CHECK-NEXT:  %13 = inttoptr i32 %[[M5]] to i8*
; CHECK-NEXT:  store i8 %3, i8* %13
; CHECK-NEXT:  %[[M6:const_mat[0-9]?]] = add i32 %[[C1]], 80
; CHECK-NEXT:  %14 = inttoptr i32 %[[M6]] to i8*
; CHECK-NEXT:  store i8* %14, i8** @goo

@goo = global i8* undef

define void @foo_i8() {
entry:
  %0 = load volatile i8, i8* inttoptr (i32 805874688 to i8*)
  %1 = load volatile i8, i8* inttoptr (i32 805874692 to i8*)
  %2 = load volatile i8, i8* inttoptr (i32 805874719 to i8*)
  %3 = load volatile i8, i8* inttoptr (i32 805874720 to i8*)
  %4 = load volatile i8, i8* inttoptr (i32 805874727 to i8*)
  store i8 %4, i8* inttoptr(i32 805873688 to i8*)
  store i8 %3, i8* inttoptr(i32 805873719 to i8*)
  store i8 %2, i8* inttoptr(i32 805873720 to i8*)
  store i8 %1, i8* inttoptr(i32 805873727 to i8*)
  store i8* inttoptr(i32 805874800 to i8*), i8** @goo
  ret void
}

; Check that for i16 type, the maximum legal offset is 62.
; CHECK: foo_i16
; CHECK-DAG: %[[C1:const[0-9]?]] = bitcast i32 805874752 to i32
; CHECK-DAG: %[[C2:const[0-9]?]] = bitcast i32 805874688 to i32
; CHECK: %0 = inttoptr i32 %[[C2]] to i16*
; CHECK-NEXT: %1 = load volatile i16, i16* %0, align 2
; CHECK-NEXT: %[[M1:const_mat[0-9]?]] = add i32 %[[C2]], 4
; CHECK-NEXT: %2 = inttoptr i32 %[[M1]] to i16*
; CHECK-NEXT: %3 = load volatile i16, i16* %2, align 2
; CHECK-NEXT: %[[M2:const_mat[0-9]?]] = add i32 %[[C2]], 32
; CHECK-NEXT: %4 = inttoptr i32 %[[M2]] to i16*
; CHECK-NEXT: %5 = load volatile i16, i16* %4, align 2
; CHECK-NEXT: %[[M3:const_mat[0-9]?]] = add i32 %[[C2]], 62
; CHECK-NEXT: %6 = inttoptr i32 %[[M3]] to i16*
; CHECK-NEXT: %7 = load volatile i16, i16* %6, align 2
; CHECK-NEXT: %8 = inttoptr i32 %[[C1]] to i16*
; CHECK-NEXT: %9 = load volatile i16, i16* %8, align 2
; CHECK-NEXT: %[[M4:const_mat[0-9]?]] = add i32 %[[C1]], 22
; CHECK-NEXT: %10 = inttoptr i32 %[[M4]] to i16*
; CHECK-NEXT: %11 = load volatile i16, i16* %10, align 2

define void @foo_i16() {
entry:
  %0 = load volatile i16, i16* inttoptr (i32 805874688 to i16*), align 2
  %1 = load volatile i16, i16* inttoptr (i32 805874692 to i16*), align 2
  %2 = load volatile i16, i16* inttoptr (i32 805874720 to i16*), align 2
  %3 = load volatile i16, i16* inttoptr (i32 805874750 to i16*), align 2
  %4 = load volatile i16, i16* inttoptr (i32 805874752 to i16*), align 2
  %5 = load volatile i16, i16* inttoptr (i32 805874774 to i16*), align 2
  ret void
}

; Check that for i32 type, the maximum legal offset is 124.
; CHECK: foo_i32
; CHECK-DAG:  %[[C1:const[0-9]?]] = bitcast i32 805874816 to i32
; CHECK-DAG:  %[[C2:const[0-9]?]] = bitcast i32 805874688 to i32
; CHECK:  %0 = inttoptr i32 %[[C2]] to i32*
; CHECK-NEXT:  %1 = load volatile i32, i32* %0, align 4
; CHECK-NEXT:  %[[M1:const_mat[0-9]?]] = add i32 %[[C2]], 4
; CHECK-NEXT:  %2 = inttoptr i32 %[[M1]] to i32*
; CHECK-NEXT:  %3 = load volatile i32, i32* %2, align 4
; CHECK-NEXT:  %[[M2:const_mat[0-9]?]] = add i32 %[[C2]], 124
; CHECK-NEXT:  %4 = inttoptr i32 %[[M2]] to i32*
; CHECK-NEXT:  %5 = load volatile i32, i32* %4, align 4
; CHECK-NEXT:  %6 = inttoptr i32 %[[C1]] to i32*
; CHECK-NEXT:  %7 = load volatile i32, i32* %6, align 4
; CHECK-NEXT:  %[[M3:const_mat[0-9]?]] = add i32 %[[C1]], 8
; CHECK-NEXT:  %8 = inttoptr i32 %[[M3]] to i32*
; CHECK-NEXT:  %9 = load volatile i32, i32* %8, align 4
; CHECK-NEXT:  %[[M4:const_mat[0-9]?]] = add i32 %[[C1]], 12
; CHECK-NEXT:  %10 = inttoptr i32 %[[M4]] to i32*
; CHECK-NEXT:  %11 = load volatile i32, i32* %10, align 4

define void @foo_i32() {
entry:
  %0 = load volatile i32, i32* inttoptr (i32 805874688 to i32*), align 4
  %1 = load volatile i32, i32* inttoptr (i32 805874692 to i32*), align 4
  %2 = load volatile i32, i32* inttoptr (i32 805874812 to i32*), align 4
  %3 = load volatile i32, i32* inttoptr (i32 805874816 to i32*), align 4
  %4 = load volatile i32, i32* inttoptr (i32 805874824 to i32*), align 4
  %5 = load volatile i32, i32* inttoptr (i32 805874828 to i32*), align 4
  ret void
}

