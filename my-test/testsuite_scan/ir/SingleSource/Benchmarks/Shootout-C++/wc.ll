; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/wc.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/wc.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::basic_istream" = type { ptr, i64, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }
%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }

@_ZSt3cin = external local_unnamed_addr global %"class.std::basic_istream", align 8
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [2 x i8] c" \00", align 1

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #0 {
  %3 = alloca [4096 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #4
  %4 = load ptr, ptr @_ZSt3cin, align 8, !tbaa !6
  %5 = getelementptr i8, ptr %4, i64 -24
  %6 = load i64, ptr %5, align 8
  %7 = getelementptr inbounds i8, ptr @_ZSt3cin, i64 %6
  %8 = getelementptr inbounds nuw i8, ptr %7, i64 232
  %9 = load ptr, ptr %8, align 8, !tbaa !9
  %10 = load ptr, ptr %9, align 8, !tbaa !6
  %11 = getelementptr inbounds nuw i8, ptr %10, i64 24
  %12 = load ptr, ptr %11, align 8
  %13 = call noundef ptr %12(ptr noundef nonnull align 8 dereferenceable(64) %9, ptr noundef nonnull %3, i64 noundef 4096)
  %14 = load ptr, ptr @_ZSt3cin, align 8, !tbaa !6
  %15 = getelementptr i8, ptr %14, i64 -24
  %16 = load i64, ptr %15, align 8
  %17 = getelementptr inbounds i8, ptr @_ZSt3cin, i64 %16
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 232
  %19 = load ptr, ptr %18, align 8, !tbaa !9
  %20 = getelementptr inbounds nuw i8, ptr %19, i64 16
  %21 = getelementptr inbounds nuw i8, ptr %19, i64 24
  br label %22

22:                                               ; preds = %74, %2
  %23 = phi i32 [ %75, %74 ], [ 0, %2 ]
  %24 = phi i32 [ 0, %74 ], [ 1, %2 ]
  %25 = phi i32 [ %78, %74 ], [ 0, %2 ]
  %26 = phi i32 [ %76, %74 ], [ 0, %2 ]
  %27 = load ptr, ptr %20, align 8, !tbaa !29
  %28 = load ptr, ptr %21, align 8, !tbaa !32
  %29 = icmp ult ptr %27, %28
  br i1 %29, label %36, label %30, !prof !33

30:                                               ; preds = %22
  %31 = load ptr, ptr %19, align 8, !tbaa !6
  %32 = getelementptr inbounds nuw i8, ptr %31, i64 80
  %33 = load ptr, ptr %32, align 8
  %34 = call noundef i32 %33(ptr noundef nonnull align 8 dereferenceable(64) %19)
  %35 = icmp eq i32 %34, -1
  br i1 %35, label %79, label %40

36:                                               ; preds = %22
  %37 = load i8, ptr %27, align 1, !tbaa !34
  %38 = zext i8 %37 to i32
  %39 = getelementptr inbounds nuw i8, ptr %27, i64 1
  store ptr %39, ptr %20, align 8, !tbaa !29
  br label %40

40:                                               ; preds = %36, %30
  %41 = phi i32 [ %38, %36 ], [ %34, %30 ]
  %42 = add nuw nsw i32 %23, 1
  %43 = and i32 %41, 255
  %44 = icmp eq i32 %43, 10
  %45 = zext i1 %44 to i32
  %46 = add nuw nsw i32 %26, %45
  %47 = trunc i32 %41 to i8
  switch i8 %47, label %74 [
    i8 32, label %48
    i8 10, label %48
    i8 9, label %48
  ]

48:                                               ; preds = %40, %40, %40
  br label %49

49:                                               ; preds = %73, %48
  %50 = phi i32 [ %42, %48 ], [ %67, %73 ]
  %51 = phi i32 [ %46, %48 ], [ %71, %73 ]
  %52 = load ptr, ptr %20, align 8, !tbaa !29
  %53 = load ptr, ptr %21, align 8, !tbaa !32
  %54 = icmp ult ptr %52, %53
  br i1 %54, label %55, label %59, !prof !33

55:                                               ; preds = %49
  %56 = load i8, ptr %52, align 1, !tbaa !34
  %57 = zext i8 %56 to i32
  %58 = getelementptr inbounds nuw i8, ptr %52, i64 1
  store ptr %58, ptr %20, align 8, !tbaa !29
  br label %65

59:                                               ; preds = %49
  %60 = load ptr, ptr %19, align 8, !tbaa !6
  %61 = getelementptr inbounds nuw i8, ptr %60, i64 80
  %62 = load ptr, ptr %61, align 8
  %63 = call noundef i32 %62(ptr noundef nonnull align 8 dereferenceable(64) %19)
  %64 = icmp eq i32 %63, -1
  br i1 %64, label %79, label %65

65:                                               ; preds = %55, %59
  %66 = phi i32 [ %57, %55 ], [ %63, %59 ]
  %67 = add nuw nsw i32 %50, 1
  %68 = and i32 %66, 255
  %69 = icmp eq i32 %68, 10
  %70 = zext i1 %69 to i32
  %71 = add nuw nsw i32 %51, %70
  %72 = trunc i32 %66 to i8
  switch i8 %72, label %74 [
    i8 32, label %73
    i8 10, label %73
    i8 9, label %73
  ]

73:                                               ; preds = %65, %65, %65
  br label %49, !llvm.loop !35

74:                                               ; preds = %65, %40
  %75 = phi i32 [ %42, %40 ], [ %67, %65 ]
  %76 = phi i32 [ %46, %40 ], [ %71, %65 ]
  %77 = phi i32 [ %24, %40 ], [ 1, %65 ]
  %78 = add nsw i32 %25, %77
  br label %22, !llvm.loop !38

79:                                               ; preds = %30, %59
  %80 = phi i32 [ %50, %59 ], [ %23, %30 ]
  %81 = phi i32 [ %51, %59 ], [ %26, %30 ]
  %82 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i32 noundef %81)
  %83 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %82, ptr noundef nonnull @.str, i64 noundef 1)
  %84 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) %82, i32 noundef %25)
  %85 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %84, ptr noundef nonnull @.str, i64 noundef 1)
  %86 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) %84, i32 noundef %80)
  %87 = load ptr, ptr %86, align 8, !tbaa !6
  %88 = getelementptr i8, ptr %87, i64 -24
  %89 = load i64, ptr %88, align 8
  %90 = getelementptr inbounds i8, ptr %86, i64 %89
  %91 = getelementptr inbounds nuw i8, ptr %90, i64 240
  %92 = load ptr, ptr %91, align 8, !tbaa !39
  %93 = icmp eq ptr %92, null
  br i1 %93, label %94, label %95

94:                                               ; preds = %79
  call void @_ZSt16__throw_bad_castv() #5
  unreachable

95:                                               ; preds = %79
  %96 = getelementptr inbounds nuw i8, ptr %92, i64 56
  %97 = load i8, ptr %96, align 8, !tbaa !40
  %98 = icmp eq i8 %97, 0
  br i1 %98, label %102, label %99

99:                                               ; preds = %95
  %100 = getelementptr inbounds nuw i8, ptr %92, i64 67
  %101 = load i8, ptr %100, align 1, !tbaa !34
  br label %107

102:                                              ; preds = %95
  call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %92)
  %103 = load ptr, ptr %92, align 8, !tbaa !6
  %104 = getelementptr inbounds nuw i8, ptr %103, i64 48
  %105 = load ptr, ptr %104, align 8
  %106 = call noundef i8 %105(ptr noundef nonnull align 8 dereferenceable(570) %92, i8 noundef 10)
  br label %107

107:                                              ; preds = %99, %102
  %108 = phi i8 [ %101, %99 ], [ %106, %102 ]
  %109 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) %86, i8 noundef %108)
  %110 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %109)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #4
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8), i32 noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #2

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #2

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #2

; Function Attrs: cold noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #3

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570)) local_unnamed_addr #2

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
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
!7 = !{!"vtable pointer", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !25, i64 232}
!10 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !11, i64 0, !23, i64 216, !13, i64 224, !24, i64 225, !25, i64 232, !26, i64 240, !27, i64 248, !28, i64 256}
!11 = !{!"_ZTSSt8ios_base", !12, i64 8, !12, i64 16, !14, i64 24, !15, i64 28, !15, i64 32, !16, i64 40, !18, i64 48, !13, i64 64, !19, i64 192, !20, i64 200, !21, i64 208}
!12 = !{!"long", !13, i64 0}
!13 = !{!"omnipotent char", !8, i64 0}
!14 = !{!"_ZTSSt13_Ios_Fmtflags", !13, i64 0}
!15 = !{!"_ZTSSt12_Ios_Iostate", !13, i64 0}
!16 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !17, i64 0}
!17 = !{!"any pointer", !13, i64 0}
!18 = !{!"_ZTSNSt8ios_base6_WordsE", !17, i64 0, !12, i64 8}
!19 = !{!"int", !13, i64 0}
!20 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !17, i64 0}
!21 = !{!"_ZTSSt6locale", !22, i64 0}
!22 = !{!"p1 _ZTSNSt6locale5_ImplE", !17, i64 0}
!23 = !{!"p1 _ZTSSo", !17, i64 0}
!24 = !{!"bool", !13, i64 0}
!25 = !{!"p1 _ZTSSt15basic_streambufIcSt11char_traitsIcEE", !17, i64 0}
!26 = !{!"p1 _ZTSSt5ctypeIcE", !17, i64 0}
!27 = !{!"p1 _ZTSSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE", !17, i64 0}
!28 = !{!"p1 _ZTSSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE", !17, i64 0}
!29 = !{!30, !31, i64 16}
!30 = !{!"_ZTSSt15basic_streambufIcSt11char_traitsIcEE", !31, i64 8, !31, i64 16, !31, i64 24, !31, i64 32, !31, i64 40, !31, i64 48, !21, i64 56}
!31 = !{!"p1 omnipotent char", !17, i64 0}
!32 = !{!30, !31, i64 24}
!33 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!34 = !{!13, !13, i64 0}
!35 = distinct !{!35, !36, !37}
!36 = !{!"llvm.loop.mustprogress"}
!37 = !{!"llvm.loop.peeled.count", i32 1}
!38 = distinct !{!38, !36}
!39 = !{!10, !26, i64 240}
!40 = !{!41, !13, i64 56}
!41 = !{!"_ZTSSt5ctypeIcE", !42, i64 0, !43, i64 16, !24, i64 24, !44, i64 32, !44, i64 40, !45, i64 48, !13, i64 56, !13, i64 57, !13, i64 313, !13, i64 569}
!42 = !{!"_ZTSNSt6locale5facetE", !19, i64 8}
!43 = !{!"p1 _ZTS15__locale_struct", !17, i64 0}
!44 = !{!"p1 int", !17, i64 0}
!45 = !{!"p1 short", !17, i64 0}
