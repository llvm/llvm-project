;; IR + Profile to split machine functions with -fsplit-machine-functions
;; Source used to generate this file
;; int k;

;; int baz() {
;;   return 0;
;; }
;;
;; int bar() {
;;   return 0;
;; }
;;
;; int foo() {
;;   if (k)
;;     baz();
;;   else
;;     bar();
;;   return 0;
;; }
;;
;; int main() {
;;   foo();
;;   return 0;
;; }

;; Commands:
;; clang -fprofile-generate -O0 source.cc
;; ./a.out && llvm-profdata merge -o default.profdata *.profraw
;; clang -fprofile-use -O0 source.cc -emit-llvm -S

target triple = "x86_64-unknown-linux-gnu"

@k = dso_local global i32 0, align 4

define dso_local i32 @_Z3bazv() !prof !31 {
  ret i32 0
}

define dso_local i32 @_Z3barv() !prof !32 {
  ret i32 0
}

define dso_local i32 @_Z3foov() !prof !32 {
  %1 = load i32, i32* @k, align 4
  %2 = icmp ne i32 %1, 0
  br i1 %2, label %3, label %5, !prof !33

3:                                                ; preds = %0
  %4 = call i32 @_Z3bazv()
  br label %7

5:                                                ; preds = %0
  %6 = call i32 @_Z3barv()
  br label %7

7:                                                ; preds = %5, %3
  ret i32 0
}

define dso_local i32 @main() !prof !32 {
  %1 = call i32 @_Z3foov()
  ret i32 0
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10, !11, !12}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 3}
!5 = !{!"MaxCount", i64 1}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1}
!8 = !{!"NumCounts", i64 5}
!9 = !{!"NumFunctions", i64 4}
!10 = !{!"IsPartialProfile", i64 0}
!11 = !{!"PartialProfileRatio", double 0.000000e+00}
!12 = !{!"DetailedSummary", !13}
!13 = !{!14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29}
!14 = !{i32 10000, i64 0, i32 0}
!15 = !{i32 100000, i64 0, i32 0}
!16 = !{i32 200000, i64 0, i32 0}
!17 = !{i32 300000, i64 0, i32 0}
!18 = !{i32 400000, i64 1, i32 3}
!19 = !{i32 500000, i64 1, i32 3}
!20 = !{i32 600000, i64 1, i32 3}
!21 = !{i32 700000, i64 1, i32 3}
!22 = !{i32 800000, i64 1, i32 3}
!23 = !{i32 900000, i64 1, i32 3}
!24 = !{i32 950000, i64 1, i32 3}
!25 = !{i32 990000, i64 1, i32 3}
!26 = !{i32 999000, i64 1, i32 3}
!27 = !{i32 999900, i64 1, i32 3}
!28 = !{i32 999990, i64 1, i32 3}
!29 = !{i32 999999, i64 1, i32 3}
!31 = !{!"function_entry_count", i64 0}
!32 = !{!"function_entry_count", i64 1}
!33 = !{!"branch_weights", i32 1, i32 0}
