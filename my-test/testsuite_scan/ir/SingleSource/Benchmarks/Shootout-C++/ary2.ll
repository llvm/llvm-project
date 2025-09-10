; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/ary2.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/ary2.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }

@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [49 x i8] c"cannot create std::vector larger than max_size()\00", align 1

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %3 = icmp eq i32 %0, 2
  br i1 %3, label %4, label %17

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !6
  %7 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %6, ptr noundef null, i32 noundef 10) #8
  %8 = trunc i64 %7 to i32
  %9 = icmp slt i32 %8, 0
  br i1 %9, label %10, label %11

10:                                               ; preds = %4
  tail call void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str) #9
  unreachable

11:                                               ; preds = %4
  %12 = mul nuw nsw i32 %8, 10
  %13 = zext nneg i32 %12 to i64
  %14 = icmp ne i32 %8, 0
  tail call void @llvm.assume(i1 %14)
  %15 = zext nneg i32 %12 to i64
  %16 = shl nuw nsw i64 %13, 2
  br label %17

17:                                               ; preds = %11, %2
  %18 = phi i64 [ %15, %11 ], [ 9000000, %2 ]
  %19 = phi i64 [ %16, %11 ], [ 36000000, %2 ]
  %20 = tail call noalias noundef nonnull ptr @_Znwm(i64 noundef %19) #10
  store i32 0, ptr %20, align 4, !tbaa !11
  %21 = getelementptr i8, ptr %20, i64 4
  %22 = add nsw i64 %19, -4
  tail call void @llvm.memset.p0.i64(ptr align 4 %21, i8 0, i64 %22, i1 false), !tbaa !11
  %23 = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %19) #10
          to label %24 unwind label %60

24:                                               ; preds = %17
  store i32 0, ptr %23, align 4, !tbaa !11
  %25 = getelementptr i8, ptr %23, i64 4
  tail call void @llvm.memset.p0.i64(ptr align 4 %25, i8 0, i64 %22, i1 false), !tbaa !11
  %26 = getelementptr i8, ptr %23, i64 %19
  br label %27

27:                                               ; preds = %24, %27
  %28 = phi i64 [ 0, %24 ], [ %58, %27 ]
  %29 = getelementptr inbounds nuw i32, ptr %20, i64 %28
  %30 = trunc nuw nsw i64 %28 to i32
  store i32 %30, ptr %29, align 4, !tbaa !11
  %31 = or disjoint i64 %28, 1
  %32 = getelementptr inbounds nuw i32, ptr %20, i64 %31
  %33 = trunc nuw nsw i64 %31 to i32
  store i32 %33, ptr %32, align 4, !tbaa !11
  %34 = add nuw nsw i64 %28, 2
  %35 = getelementptr inbounds nuw i32, ptr %20, i64 %34
  %36 = trunc nuw nsw i64 %34 to i32
  store i32 %36, ptr %35, align 4, !tbaa !11
  %37 = add nuw nsw i64 %28, 3
  %38 = getelementptr inbounds nuw i32, ptr %20, i64 %37
  %39 = trunc nuw nsw i64 %37 to i32
  store i32 %39, ptr %38, align 4, !tbaa !11
  %40 = add nuw nsw i64 %28, 4
  %41 = getelementptr inbounds nuw i32, ptr %20, i64 %40
  %42 = trunc nuw nsw i64 %40 to i32
  store i32 %42, ptr %41, align 4, !tbaa !11
  %43 = add nuw nsw i64 %28, 5
  %44 = getelementptr inbounds nuw i32, ptr %20, i64 %43
  %45 = trunc nuw nsw i64 %43 to i32
  store i32 %45, ptr %44, align 4, !tbaa !11
  %46 = add nuw nsw i64 %28, 6
  %47 = getelementptr inbounds nuw i32, ptr %20, i64 %46
  %48 = trunc nuw nsw i64 %46 to i32
  store i32 %48, ptr %47, align 4, !tbaa !11
  %49 = add nuw nsw i64 %28, 7
  %50 = getelementptr inbounds nuw i32, ptr %20, i64 %49
  %51 = trunc nuw nsw i64 %49 to i32
  store i32 %51, ptr %50, align 4, !tbaa !11
  %52 = add nuw nsw i64 %28, 8
  %53 = getelementptr inbounds nuw i32, ptr %20, i64 %52
  %54 = trunc nuw nsw i64 %52 to i32
  store i32 %54, ptr %53, align 4, !tbaa !11
  %55 = add nuw nsw i64 %28, 9
  %56 = getelementptr inbounds nuw i32, ptr %20, i64 %55
  %57 = trunc nuw nsw i64 %55 to i32
  store i32 %57, ptr %56, align 4, !tbaa !11
  %58 = add nuw nsw i64 %28, 10
  %59 = icmp samesign ult i64 %58, %18
  br i1 %59, label %27, label %66, !llvm.loop !13

60:                                               ; preds = %17
  %61 = landingpad { ptr, i32 }
          cleanup
  br label %112

62:                                               ; preds = %66
  %63 = getelementptr inbounds i8, ptr %26, i64 -4
  %64 = load i32, ptr %63, align 4, !tbaa !11
  %65 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i32 noundef %64)
          to label %81 unwind label %110

66:                                               ; preds = %27, %66
  %67 = phi i64 [ %76, %66 ], [ %18, %27 ]
  %68 = add nsw i64 %67, -2
  %69 = getelementptr inbounds nuw i32, ptr %20, i64 %68
  %70 = getelementptr inbounds nuw i32, ptr %23, i64 %68
  %71 = load <2 x i32>, ptr %69, align 4, !tbaa !11
  store <2 x i32> %71, ptr %70, align 4, !tbaa !11
  %72 = add nsw i64 %67, -6
  %73 = getelementptr inbounds nuw i32, ptr %20, i64 %72
  %74 = getelementptr inbounds nuw i32, ptr %23, i64 %72
  %75 = load <4 x i32>, ptr %73, align 4, !tbaa !11
  store <4 x i32> %75, ptr %74, align 4, !tbaa !11
  %76 = add nsw i64 %67, -10
  %77 = getelementptr inbounds nuw i32, ptr %20, i64 %76
  %78 = getelementptr inbounds nuw i32, ptr %23, i64 %76
  %79 = load <4 x i32>, ptr %77, align 4, !tbaa !11
  store <4 x i32> %79, ptr %78, align 4, !tbaa !11
  %80 = icmp samesign ugt i64 %67, 10
  br i1 %80, label %66, label %62, !llvm.loop !15

81:                                               ; preds = %62
  %82 = load ptr, ptr %65, align 8, !tbaa !16
  %83 = getelementptr i8, ptr %82, i64 -24
  %84 = load i64, ptr %83, align 8
  %85 = getelementptr inbounds i8, ptr %65, i64 %84
  %86 = getelementptr inbounds nuw i8, ptr %85, i64 240
  %87 = load ptr, ptr %86, align 8, !tbaa !18
  %88 = icmp eq ptr %87, null
  br i1 %88, label %89, label %91

89:                                               ; preds = %81
  invoke void @_ZSt16__throw_bad_castv() #9
          to label %90 unwind label %110

90:                                               ; preds = %89
  unreachable

91:                                               ; preds = %81
  %92 = getelementptr inbounds nuw i8, ptr %87, i64 56
  %93 = load i8, ptr %92, align 8, !tbaa !35
  %94 = icmp eq i8 %93, 0
  br i1 %94, label %98, label %95

95:                                               ; preds = %91
  %96 = getelementptr inbounds nuw i8, ptr %87, i64 67
  %97 = load i8, ptr %96, align 1, !tbaa !41
  br label %104

98:                                               ; preds = %91
  invoke void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %87)
          to label %99 unwind label %110

99:                                               ; preds = %98
  %100 = load ptr, ptr %87, align 8, !tbaa !16
  %101 = getelementptr inbounds nuw i8, ptr %100, i64 48
  %102 = load ptr, ptr %101, align 8
  %103 = invoke noundef i8 %102(ptr noundef nonnull align 8 dereferenceable(570) %87, i8 noundef 10)
          to label %104 unwind label %110

104:                                              ; preds = %99, %95
  %105 = phi i8 [ %97, %95 ], [ %103, %99 ]
  %106 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) %65, i8 noundef %105)
          to label %107 unwind label %110

107:                                              ; preds = %104
  %108 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %106)
          to label %109 unwind label %110

109:                                              ; preds = %107
  tail call void @_ZdlPvm(ptr noundef nonnull %23, i64 noundef %19) #11
  tail call void @_ZdlPvm(ptr noundef nonnull %20, i64 noundef %19) #11
  ret i32 0

110:                                              ; preds = %107, %104, %99, %98, %89, %62
  %111 = landingpad { ptr, i32 }
          cleanup
  tail call void @_ZdlPvm(ptr noundef nonnull %23, i64 noundef %19) #11
  br label %112

112:                                              ; preds = %110, %60
  %113 = phi { ptr, i32 } [ %61, %60 ], [ %111, %110 ]
  tail call void @_ZdlPvm(ptr noundef nonnull %20, i64 noundef %19) #11
  resume { ptr, i32 } %113
}

declare i32 @__gxx_personality_v0(...)

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8), i32 noundef) local_unnamed_addr #1

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: cold noreturn
declare void @_ZSt20__throw_length_errorPKc(ptr noundef) local_unnamed_addr #3

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #4

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #5

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #1

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #1

; Function Attrs: cold noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #3

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570)) local_unnamed_addr #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #6

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #7

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #7 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #8 = { nounwind }
attributes #9 = { cold noreturn }
attributes #10 = { builtin allocsize(0) }
attributes #11 = { builtin nounwind }

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
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !9, i64 0}
!13 = distinct !{!13, !14}
!14 = !{!"llvm.loop.mustprogress"}
!15 = distinct !{!15, !14}
!16 = !{!17, !17, i64 0}
!17 = !{!"vtable pointer", !10, i64 0}
!18 = !{!19, !32, i64 240}
!19 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !20, i64 0, !29, i64 216, !9, i64 224, !30, i64 225, !31, i64 232, !32, i64 240, !33, i64 248, !34, i64 256}
!20 = !{!"_ZTSSt8ios_base", !21, i64 8, !21, i64 16, !22, i64 24, !23, i64 28, !23, i64 32, !24, i64 40, !25, i64 48, !9, i64 64, !12, i64 192, !26, i64 200, !27, i64 208}
!21 = !{!"long", !9, i64 0}
!22 = !{!"_ZTSSt13_Ios_Fmtflags", !9, i64 0}
!23 = !{!"_ZTSSt12_Ios_Iostate", !9, i64 0}
!24 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !8, i64 0}
!25 = !{!"_ZTSNSt8ios_base6_WordsE", !8, i64 0, !21, i64 8}
!26 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !8, i64 0}
!27 = !{!"_ZTSSt6locale", !28, i64 0}
!28 = !{!"p1 _ZTSNSt6locale5_ImplE", !8, i64 0}
!29 = !{!"p1 _ZTSSo", !8, i64 0}
!30 = !{!"bool", !9, i64 0}
!31 = !{!"p1 _ZTSSt15basic_streambufIcSt11char_traitsIcEE", !8, i64 0}
!32 = !{!"p1 _ZTSSt5ctypeIcE", !8, i64 0}
!33 = !{!"p1 _ZTSSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!34 = !{!"p1 _ZTSSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!35 = !{!36, !9, i64 56}
!36 = !{!"_ZTSSt5ctypeIcE", !37, i64 0, !38, i64 16, !30, i64 24, !39, i64 32, !39, i64 40, !40, i64 48, !9, i64 56, !9, i64 57, !9, i64 313, !9, i64 569}
!37 = !{!"_ZTSNSt6locale5facetE", !12, i64 8}
!38 = !{!"p1 _ZTS15__locale_struct", !8, i64 0}
!39 = !{!"p1 int", !8, i64 0}
!40 = !{!"p1 short", !8, i64 0}
!41 = !{!9, !9, i64 0}
