; RUN: llc --function-sections -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN: FileCheck %s

; RUN: llc --function-sections -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN: FileCheck %s

@i = external local_unnamed_addr global i32, align 4
@__llvm_profile_raw_version = weak hidden local_unnamed_addr constant i64 72057594037927944
@__profc_func1 = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8
@__profd_func1 = private global { i64, i64, i32, ptr, ptr, i32, [4 x i16] } { i64 -2545542355363006406, i64 742261418966908927, i32 sub (i32 ptrtoint (ptr @__profc_func1 to i32), i32 ptrtoint (ptr @__profd_func1 to i32)), ptr @func1.local, ptr null, i32 1, [4 x i16] zeroinitializer }, section "__llvm_prf_data", align 8, !pgo.associated !0
@__profc_func2 = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8
@__profd_func2 = private global { i64, i64, i32, ptr, ptr, i32, [4 x i16] } { i64 -4377547752858689819, i64 742261418966908927, i32 sub (i32 ptrtoint (ptr @__profc_func2 to i32), i32 ptrtoint (ptr @__profd_func2 to i32)), ptr @func2.local, ptr null, i32 1, [4 x i16] zeroinitializer }, section "__llvm_prf_data", align 8, !pgo.associated !1
@__llvm_prf_nm = private constant [13 x i8] c"\0B\00func1\01func2", section "__llvm_prf_names", align 1
@__llvm_profile_filename = weak hidden local_unnamed_addr constant [19 x i8] c"default_%m.profraw\00"
@llvm.used = appending global [3 x ptr] [ptr @__llvm_prf_nm, ptr @__profd_func1, ptr @__profd_func2], section "llvm.metadata"

@func1.local = private alias i32 (), ptr @func1
@func2.local = private alias i32 (), ptr @func2

define i32 @func1() {
entry:
  %pgocount = load i64, ptr @__profc_func1, align 8
  %0 = add i64 %pgocount, 1
  store i64 %0, ptr @__profc_func1, align 8
  %1 = load i32, ptr @i, align 4
  ret i32 %1
}

define i32 @func2() {
entry:
  %pgocount = load i64, ptr @__profc_func2, align 8
  %0 = add i64 %pgocount, 1
  store i64 %0, ptr @__profc_func2, align 8
  %1 = load i32, ptr @i, align 4
  %call = tail call i32 @external_func(i32 noundef %1)
  ret i32 %call
}

declare i32 @external_func(i32 noundef)

!0 = !{ptr @__profc_func1}
!1 = !{ptr @__profc_func2}

; CHECK-DAG:        .csect __llvm_prf_cnts.__profc_func1[RW]
; CHECK-DAG:        .csect __llvm_prf_data.__profd_func1[RW]
; CHECK-DAG:        .csect __llvm_prf_cnts.__profc_func2[RW]
; CHECK-DAG:        .csect __llvm_prf_data.__profd_func2[RW]
; CHECK-DAG:        .csect __llvm_prf_names[RO]

; CHECK:             .csect __llvm_prf_cnts.__profc_func1[RW]
; CHECK-NEXT:        .ref __llvm_prf_names[RO]
; CHECK-NEXT:        .ref __llvm_prf_data.__profd_func1[RW]
; CHECK-NEXT:        .rename __llvm_prf_cnts.__profc_func1[RW],"__llvm_prf_cnts"

; CHECK:             .csect __llvm_prf_data.__profd_func1[RW]
; CHECK-NEXT:        .rename __llvm_prf_data.__profd_func1[RW],"__llvm_prf_data"

; CHECK:             .csect __llvm_prf_cnts.__profc_func2[RW]
; CHECK-NEXT:        .ref __llvm_prf_names[RO]
; CHECK-NEXT:        .ref __llvm_prf_data.__profd_func2[RW]
; CHECK-NEXT:        .rename __llvm_prf_cnts.__profc_func2[RW],"__llvm_prf_cnts"

; CHECK:             .csect __llvm_prf_data.__profd_func2[RW]
; CHECK-NEXT:        .rename __llvm_prf_data.__profd_func2[RW],"__llvm_prf_data"
