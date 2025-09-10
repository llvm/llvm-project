; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/random.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/random.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }

$_ZZ10gen_randomdE4last = comdat any

@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@_ZZ10gen_randomdE4last = linkonce_odr dso_local local_unnamed_addr global i64 42, comdat, align 8

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %0, 2
  br i1 %3, label %4, label %10

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !6
  %7 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %6, ptr noundef null, i32 noundef 10) #4
  %8 = trunc i64 %7 to i32
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %25, label %10

10:                                               ; preds = %2, %4
  %11 = phi i32 [ %8, %4 ], [ 400000000, %2 ]
  %12 = load i64, ptr @_ZZ10gen_randomdE4last, align 8
  br label %13

13:                                               ; preds = %10, %13
  %14 = phi i32 [ %16, %13 ], [ %11, %10 ]
  %15 = phi i64 [ %19, %13 ], [ %12, %10 ]
  %16 = add nsw i32 %14, -1
  %17 = mul nsw i64 %15, 3877
  %18 = add nsw i64 %17, 29573
  %19 = srem i64 %18, 139968
  %20 = icmp eq i32 %16, 0
  br i1 %20, label %21, label %13, !llvm.loop !11

21:                                               ; preds = %13
  store i64 %19, ptr @_ZZ10gen_randomdE4last, align 8, !tbaa !13
  %22 = sitofp i64 %19 to double
  %23 = fmul double %22, 1.000000e+02
  %24 = fdiv double %23, 1.399680e+05
  br label %25

25:                                               ; preds = %21, %4
  %26 = phi double [ %24, %21 ], [ 0.000000e+00, %4 ]
  %27 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !15
  %28 = getelementptr i8, ptr %27, i64 -24
  %29 = load i64, ptr %28, align 8
  %30 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %29
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 8
  store i64 9, ptr %31, align 8, !tbaa !17
  %32 = load i64, ptr %28, align 8
  %33 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %32
  %34 = getelementptr inbounds nuw i8, ptr %33, i64 24
  %35 = load i32, ptr %34, align 8, !tbaa !27
  %36 = or i32 %35, 4
  store i32 %36, ptr %34, align 8, !tbaa !28
  %37 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, double noundef %26)
  %38 = load ptr, ptr %37, align 8, !tbaa !15
  %39 = getelementptr i8, ptr %38, i64 -24
  %40 = load i64, ptr %39, align 8
  %41 = getelementptr inbounds i8, ptr %37, i64 %40
  %42 = getelementptr inbounds nuw i8, ptr %41, i64 240
  %43 = load ptr, ptr %42, align 8, !tbaa !29
  %44 = icmp eq ptr %43, null
  br i1 %44, label %45, label %46

45:                                               ; preds = %25
  tail call void @_ZSt16__throw_bad_castv() #5
  unreachable

46:                                               ; preds = %25
  %47 = getelementptr inbounds nuw i8, ptr %43, i64 56
  %48 = load i8, ptr %47, align 8, !tbaa !37
  %49 = icmp eq i8 %48, 0
  br i1 %49, label %53, label %50

50:                                               ; preds = %46
  %51 = getelementptr inbounds nuw i8, ptr %43, i64 67
  %52 = load i8, ptr %51, align 1, !tbaa !43
  br label %58

53:                                               ; preds = %46
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %43)
  %54 = load ptr, ptr %43, align 8, !tbaa !15
  %55 = getelementptr inbounds nuw i8, ptr %54, i64 48
  %56 = load ptr, ptr %55, align 8
  %57 = tail call noundef i8 %56(ptr noundef nonnull align 8 dereferenceable(570) %43, i8 noundef 10)
  br label %58

58:                                               ; preds = %50, %53
  %59 = phi i8 [ %52, %50 ], [ %57, %53 ]
  %60 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) %37, i8 noundef %59)
  %61 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %60)
  ret i32 0
}

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #1

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8), double noundef) local_unnamed_addr #2

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #2

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #2

; Function Attrs: cold noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #3

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570)) local_unnamed_addr #2

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind }
attributes #5 = { cold noreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
!13 = !{!14, !14, i64 0}
!14 = !{!"long", !9, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"vtable pointer", !10, i64 0}
!17 = !{!18, !14, i64 8}
!18 = !{!"_ZTSSt8ios_base", !14, i64 8, !14, i64 16, !19, i64 24, !20, i64 28, !20, i64 32, !21, i64 40, !22, i64 48, !9, i64 64, !23, i64 192, !24, i64 200, !25, i64 208}
!19 = !{!"_ZTSSt13_Ios_Fmtflags", !9, i64 0}
!20 = !{!"_ZTSSt12_Ios_Iostate", !9, i64 0}
!21 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !8, i64 0}
!22 = !{!"_ZTSNSt8ios_base6_WordsE", !8, i64 0, !14, i64 8}
!23 = !{!"int", !9, i64 0}
!24 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !8, i64 0}
!25 = !{!"_ZTSSt6locale", !26, i64 0}
!26 = !{!"p1 _ZTSNSt6locale5_ImplE", !8, i64 0}
!27 = !{!18, !19, i64 24}
!28 = !{!19, !19, i64 0}
!29 = !{!30, !34, i64 240}
!30 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !18, i64 0, !31, i64 216, !9, i64 224, !32, i64 225, !33, i64 232, !34, i64 240, !35, i64 248, !36, i64 256}
!31 = !{!"p1 _ZTSSo", !8, i64 0}
!32 = !{!"bool", !9, i64 0}
!33 = !{!"p1 _ZTSSt15basic_streambufIcSt11char_traitsIcEE", !8, i64 0}
!34 = !{!"p1 _ZTSSt5ctypeIcE", !8, i64 0}
!35 = !{!"p1 _ZTSSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!36 = !{!"p1 _ZTSSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!37 = !{!38, !9, i64 56}
!38 = !{!"_ZTSSt5ctypeIcE", !39, i64 0, !40, i64 16, !32, i64 24, !41, i64 32, !41, i64 40, !42, i64 48, !9, i64 56, !9, i64 57, !9, i64 313, !9, i64 569}
!39 = !{!"_ZTSNSt6locale5facetE", !23, i64 8}
!40 = !{!"p1 _ZTS15__locale_struct", !8, i64 0}
!41 = !{!"p1 int", !8, i64 0}
!42 = !{!"p1 short", !8, i64 0}
!43 = !{!9, !9, i64 0}
