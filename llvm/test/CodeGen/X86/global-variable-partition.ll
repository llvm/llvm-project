

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-split-machine-functions \
; RUN:     -partition-static-data-sections=true -data-sections=true \
; RUN:     -unique-section-names=true \
; RUN:     %s -o - 2>&1 | FileCheck %s --check-prefix=DATA

; DATA: .rodata.str1.1.hot

@.str = private unnamed_addr constant [5 x i8] c"hot\09\00", align 1, !section_prefix !0
@.str.1 = private unnamed_addr constant [10 x i8] c"%d\09%d\09%d\0A\00", align 1, !section_prefix !0
@_ZL5group = internal unnamed_addr constant [2 x ptr] [ptr @_ZL4bss2, ptr @_ZL5data3], align 16, !section_prefix !0
@_ZL8hot_data = internal unnamed_addr global i32 5, align 4, !section_prefix !0
@_ZL7hot_bss = internal unnamed_addr global i32 0, align 4, !section_prefix !0
@.str.2 = private unnamed_addr constant [14 x i8] c"cold%d\09%d\09%d\0A\00", align 1, !section_prefix !1
@_ZL8cold_bss = internal unnamed_addr global i32 0, align 4, !section_prefix !1
@_ZL9cold_data = internal unnamed_addr global i32 4, align 4, !section_prefix !1
@_ZL10cold_group = internal unnamed_addr constant [2 x ptr] [ptr @_ZL5data3, ptr @_ZL4bss2], align 16, !section_prefix !1
@_ZL4bss2 = internal global i32 0, align 4
@_ZL5data3 = internal global i32 3, align 4

; Function Attrs: hot inlinehint mustprogress nofree noinline norecurse nounwind uwtable
define internal fastcc void @_Z12hotJumptablei(i32 noundef %0) unnamed_addr #0 align 32 !prof !51 !section_prefix !0 {
  %2 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str)
  %3 = srem i32 %0, 2
  %4 = sext i32 %3 to i64
  %5 = getelementptr inbounds [2 x ptr], ptr @_ZL5group, i64 0, i64 %4
  %6 = load ptr, ptr %5, align 8
  %7 = load i32, ptr %6, align 4
  %8 = load i32, ptr @_ZL8hot_data, align 4
  %9 = load i32, ptr @_ZL7hot_bss, align 4
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %7, i32 noundef %8, i32 noundef %9)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: cold mustprogress nofree noinline norecurse nounwind optsize uwtable
define internal fastcc void @_Z13coldJumptablei(i32 noundef %0) unnamed_addr #2 align 32 !prof !52 !section_prefix !1 {
  %2 = load i32, ptr @_ZL8cold_bss, align 4
  %3 = load i32, ptr @_ZL9cold_data, align 4
  %4 = srem i32 %0, 2
  %5 = sext i32 %4 to i64
  %6 = getelementptr inbounds [2 x ptr], ptr @_ZL10cold_group, i64 0, i64 %5
  %7 = load ptr, ptr %6, align 8
  %8 = load i32, ptr %7, align 4
  %9 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %2, i32 noundef %3, i32 noundef %8)
  ret void
}

; Function Attrs: mustprogress norecurse nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #3 align 32 !prof !52 {
  %3 = tail call i64 @time(ptr noundef null) #5
  %4 = trunc i64 %3 to i32
  tail call void @srand(i32 noundef %4) #5
  br label %11

5:                                                ; preds = %11
  %6 = tail call i32 @rand() #5
  store i32 %6, ptr @_ZL8cold_bss, align 4
  %7 = tail call i32 @rand() #5
  store i32 %7, ptr @_ZL9cold_data, align 4
  %8 = tail call i32 @rand() #5
  store i32 %8, ptr @_ZL4bss2, align 4
  %9 = tail call i32 @rand() #5
  store i32 %9, ptr @_ZL5data3, align 4
  %10 = tail call i32 @rand() #5
  tail call fastcc void @_Z13coldJumptablei(i32 noundef %10) #6
  ret i32 0

11:                                               ; preds = %11, %2
  %12 = phi i32 [ 0, %2 ], [ %19, %11 ]
  %13 = tail call i32 @rand() #5
  %14 = srem i32 %13, 2
  %15 = sext i32 %14 to i64
  %16 = getelementptr inbounds [2 x ptr], ptr @_ZL5group, i64 0, i64 %15
  %17 = load ptr, ptr %16, align 8
  store i32 %13, ptr %17, align 4
  store i32 %13, ptr @_ZL8hot_data, align 4
  %18 = add nsw i32 %13, 1
  store i32 %18, ptr @_ZL7hot_bss, align 4
  tail call fastcc void @_Z12hotJumptablei(i32 noundef %12) #7
  %19 = add nuw nsw i32 %12, 1
  %20 = icmp eq i32 %19, 100000
  br i1 %20, label %5, label %11, !prof !53, !llvm.loop !54
}

; Function Attrs: nounwind
declare void @srand(i32 noundef) local_unnamed_addr #4

; Function Attrs: nounwind
declare i64 @time(ptr noundef) local_unnamed_addr #4

; Function Attrs: nounwind
declare i32 @rand() local_unnamed_addr #4

attributes #0 = { hot inlinehint mustprogress nofree noinline norecurse nounwind uwtable "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "prefer-vector-width"="128" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+aes,+avx,+cmov,+crc32,+cx16,+cx8,+fxsr,+mmx,+pclmul,+popcnt,+prfchw,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "prefer-vector-width"="128" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+aes,+avx,+cmov,+crc32,+cx16,+cx8,+fxsr,+mmx,+pclmul,+popcnt,+prfchw,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" }
attributes #2 = { cold mustprogress nofree noinline norecurse nounwind optsize uwtable "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "prefer-vector-width"="128" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+aes,+avx,+cmov,+crc32,+cx16,+cx8,+fxsr,+mmx,+pclmul,+popcnt,+prfchw,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" }
attributes #3 = { mustprogress norecurse nounwind uwtable "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "prefer-vector-width"="128" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+aes,+avx,+cmov,+crc32,+cx16,+cx8,+fxsr,+mmx,+pclmul,+popcnt,+prfchw,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" }
attributes #4 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "prefer-vector-width"="128" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+aes,+avx,+cmov,+crc32,+cx16,+cx8,+fxsr,+mmx,+pclmul,+popcnt,+prfchw,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" }
attributes #5 = { nounwind }
attributes #6 = { cold }
attributes #7 = { hot }

!llvm.linker.options = !{}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !41}
!llvm.ident = !{!50}

!0 = !{!"section_prefix", !"hot"}
!1 = !{!"section_prefix", !"unlikely"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 1, !"Virtual Function Elim", i32 0}
!4 = !{i32 8, !"PIC Level", i32 2}
!5 = !{i32 7, !"PIE Level", i32 2}
!6 = !{i32 1, !"Code Model", i32 3}
!7 = !{i32 1, !"Large Data Threshold", i64 65536}
!8 = !{i32 7, !"direct-access-external-data", i32 1}
!9 = !{i32 7, !"uwtable", i32 2}
!10 = !{i32 7, !"frame-pointer", i32 1}
!11 = !{i32 1, !"EnableSplitLTOUnit", i32 0}
!12 = !{i32 1, !"ProfileSummary", !13}
!13 = !{!14, !15, !16, !17, !18, !19, !20, !21, !22, !23}
!14 = !{!"ProfileFormat", !"InstrProf"}
!15 = !{!"TotalCount", i64 1460183}
!16 = !{!"MaxCount", i64 849024}
!17 = !{!"MaxInternalCount", i64 32769}
!18 = !{!"MaxFunctionCount", i64 849024}
!19 = !{!"NumCounts", i64 23627}
!20 = !{!"NumFunctions", i64 3271}
!21 = !{!"IsPartialProfile", i64 0}
!22 = !{!"PartialProfileRatio", double 0.000000e+00}
!23 = !{!"DetailedSummary", !24}
!24 = !{!25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40}
!25 = !{i32 10000, i64 849024, i32 1}
!26 = !{i32 100000, i64 849024, i32 1}
!27 = !{i32 200000, i64 849024, i32 1}
!28 = !{i32 300000, i64 849024, i32 1}
!29 = !{i32 400000, i64 849024, i32 1}
!30 = !{i32 500000, i64 849024, i32 1}
!31 = !{i32 600000, i64 100000, i32 3}
!32 = !{i32 700000, i64 100000, i32 3}
!33 = !{i32 800000, i64 32640, i32 10}
!34 = !{i32 900000, i64 26532, i32 11}
!35 = !{i32 950000, i64 7904, i32 18}
!36 = !{i32 990000, i64 166, i32 73}
!37 = !{i32 999000, i64 5, i32 468}
!38 = !{i32 999900, i64 1, i32 1443}
!39 = !{i32 999990, i64 1, i32 1443}
!40 = !{i32 999999, i64 1, i32 1443}
!41 = !{i32 5, !"CG Profile", !42}
!42 = distinct !{!43, !44, !45, !46, !47, !48, !49}
!43 = !{ptr @_Z12hotJumptablei, ptr @printf, i64 200000}
!44 = !{ptr @_Z13coldJumptablei, ptr @printf, i64 1}
!45 = !{ptr @main, ptr @time, i64 1}
!46 = !{ptr @main, ptr @srand, i64 1}
!47 = !{ptr @main, ptr @rand, i64 100004}
!48 = !{ptr @main, ptr @_Z13coldJumptablei, i64 1}
!49 = !{ptr @main, ptr @_Z12hotJumptablei, i64 99999}
!50 = !{!"google3 clang version 9999.0.0 (386af4a5c64ab75eaee2448dc38f2e34a40bfed0)"}
!51 = !{!"function_entry_count", i64 100000}
!52 = !{!"function_entry_count", i64 1}
!53 = !{!"branch_weights", i32 1, i32 99999}
!54 = distinct !{!54, !55}
!55 = !{!"llvm.loop.mustprogress"}
