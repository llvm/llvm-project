; RUN: llc -mtriple=aarch64-linux-gnu -aarch64-enable-global-merge  -global-merge-group-by-use=false < %s | FileCheck %s
; CHECK-NOT: _MergedGlobals

$__profc_begin = comdat nodeduplicate
$__profc_end = comdat nodeduplicate

@__profc_begin = private global [2 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8
@__profd_begin = private global { i64, i64, i64, i64, ptr, ptr, i32, [3 x i16], i32 } { i64 -1301828029439649651, i64 172590168, i64 sub (i64 ptrtoint (ptr @__profc_begin to i64), i64 ptrtoint (ptr @__profd_begin to i64)), i64 0, ptr null, ptr null, i32 2, [3 x i16] zeroinitializer, i32 0 }, section "__llvm_prf_data", comdat($__profc_begin), align 8
@__profc_end = private global [2 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8
@__profd_end = private global { i64, i64, i64, i64, ptr, ptr, i32, [3 x i16], i32 } { i64 3274037854792712831, i64 172590168, i64 sub (i64 ptrtoint (ptr @__profc_end to i64), i64 ptrtoint (ptr @__profd_end to i64)), i64 0, ptr null, ptr null, i32 2, [3 x i16] zeroinitializer, i32 0 }, section "__llvm_prf_data", comdat($__profc_end), align 8

