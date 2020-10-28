;; RUN: llc --mattr=+v8.3a %s -o - | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux"

@__llvm_gcov_ctr = internal global [1 x i64] zeroinitializer
@0 = private unnamed_addr constant [7 x i8] c"m.gcda\00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 0, void ()* @__llvm_gcov_init, i8* null }]

define dso_local i32 @f() local_unnamed_addr #0 {
entry:
  ret i32 0
}
;; CHECK-LABEL: f:
;; CHECK: pacibsp

declare void @llvm_gcda_start_file(i8*, i32, i32) local_unnamed_addr

declare void @llvm_gcda_emit_function(i32, i32, i32) local_unnamed_addr

declare void @llvm_gcda_emit_arcs(i32, i64*) local_unnamed_addr

declare void @llvm_gcda_summary_info() local_unnamed_addr

declare void @llvm_gcda_end_file() local_unnamed_addr

define internal void @__llvm_gcov_writeout() unnamed_addr #1 {
entry:
  tail call void @llvm_gcda_start_file(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @0, i64 0, i64 0), i32 875575338, i32 0)
  tail call void @llvm_gcda_emit_function(i32 0, i32 0, i32 0)
  tail call void @llvm_gcda_emit_arcs(i32 1, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__llvm_gcov_ctr, i64 0, i64 0))
  tail call void @llvm_gcda_summary_info()
  tail call void @llvm_gcda_end_file()
  ret void
}
;; CHECK-LABEL: __llvm_gcov_writeout:
;; CHECK:       .cfi_b_key_frame
;; CHECK-NEXT:  pacibsp
;; CHECK-NEXT: .cfi_negate_ra_state

define internal void @__llvm_gcov_reset() unnamed_addr #2 {
entry:
  store i64 0, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__llvm_gcov_ctr, i64 0, i64 0), align 8
  ret void
}
;; CHECK-LABEL: __llvm_gcov_reset:
;; CHECK:       pacibsp

declare void @llvm_gcov_init(void ()*, void ()*) local_unnamed_addr

define internal void @__llvm_gcov_init() unnamed_addr #1 {
entry:
  tail call void @llvm_gcov_init(void ()* nonnull @__llvm_gcov_writeout, void ()* nonnull @__llvm_gcov_reset)
  ret void
}
;; CHECK-LABEL: __llvm_gcov_init:
;; CHECK:      .cfi_b_key_frame
;; CHECK-NEXT:  pacibsp
;; CHECK-NEXT: .cfi_negate_ra_state

attributes #0 = { norecurse nounwind readnone "sign-return-address"="all" "sign-return-address-key"="b_key" }
attributes #1 = { noinline }
attributes #2 = { nofree noinline norecurse nounwind writeonly }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 1, !"branch-target-enforcement", i32 0}
!3 = !{i32 1, !"sign-return-address", i32 1}
!4 = !{i32 1, !"sign-return-address-all", i32 1}
!5 = !{i32 1, !"sign-return-address-with-bkey", i32 1}
