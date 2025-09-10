; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/ary.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/ary.cpp"
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
  br i1 %3, label %4, label %15

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !6
  %7 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %6, ptr noundef null, i32 noundef 10) #8
  %8 = shl i64 %7, 32
  %9 = ashr exact i64 %8, 32
  %10 = icmp ugt i64 %9, 2305843009213693951
  br i1 %10, label %11, label %12

11:                                               ; preds = %4
  tail call void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str) #9
  unreachable

12:                                               ; preds = %4
  %13 = trunc i64 %7 to i32
  %14 = icmp eq i64 %8, 0
  br i1 %14, label %36, label %15

15:                                               ; preds = %2, %12
  %16 = phi i32 [ %13, %12 ], [ 9000000, %2 ]
  %17 = phi i64 [ %9, %12 ], [ 9000000, %2 ]
  %18 = shl nuw nsw i64 %17, 2
  %19 = tail call noalias noundef nonnull ptr @_Znwm(i64 noundef %18) #10
  %20 = getelementptr inbounds nuw i32, ptr %19, i64 %17
  store i32 0, ptr %19, align 4, !tbaa !11
  %21 = icmp eq i64 %17, 1
  br i1 %21, label %25, label %22

22:                                               ; preds = %15
  %23 = getelementptr i8, ptr %19, i64 4
  %24 = add nsw i64 %18, -4
  tail call void @llvm.memset.p0.i64(ptr align 4 %23, i8 0, i64 %24, i1 false), !tbaa !11
  br label %25

25:                                               ; preds = %15, %22
  %26 = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %18) #10
          to label %27 unwind label %71

27:                                               ; preds = %25
  %28 = getelementptr inbounds nuw i32, ptr %26, i64 %17
  store i32 0, ptr %26, align 4, !tbaa !11
  %29 = getelementptr i8, ptr %26, i64 4
  %30 = add nsw i64 %17, -1
  %31 = icmp eq i64 %30, 0
  br i1 %31, label %36, label %32

32:                                               ; preds = %27
  %33 = add nsw i64 %18, -4
  tail call void @llvm.memset.p0.i64(ptr align 4 %29, i8 0, i64 %33, i1 false), !tbaa !11
  %34 = shl nuw nsw i64 %30, 2
  %35 = getelementptr inbounds nuw i8, ptr %29, i64 %34
  br label %36

36:                                               ; preds = %32, %27, %12
  %37 = phi ptr [ %19, %27 ], [ %19, %32 ], [ null, %12 ]
  %38 = phi ptr [ %20, %27 ], [ %20, %32 ], [ null, %12 ]
  %39 = phi i32 [ %16, %27 ], [ %16, %32 ], [ %13, %12 ]
  %40 = phi ptr [ %26, %27 ], [ %26, %32 ], [ null, %12 ]
  %41 = phi ptr [ %28, %27 ], [ %28, %32 ], [ null, %12 ]
  %42 = phi ptr [ %29, %27 ], [ %35, %32 ], [ null, %12 ]
  %43 = icmp sgt i32 %39, 0
  br i1 %43, label %44, label %73

44:                                               ; preds = %36
  %45 = zext nneg i32 %39 to i64
  %46 = icmp ult i32 %39, 8
  br i1 %46, label %60, label %47

47:                                               ; preds = %44
  %48 = and i64 %45, 2147483640
  br label %49

49:                                               ; preds = %49, %47
  %50 = phi i64 [ 0, %47 ], [ %55, %49 ]
  %51 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %47 ], [ %56, %49 ]
  %52 = add <4 x i32> %51, splat (i32 4)
  %53 = getelementptr inbounds nuw i32, ptr %37, i64 %50
  %54 = getelementptr inbounds nuw i8, ptr %53, i64 16
  store <4 x i32> %51, ptr %53, align 4, !tbaa !11
  store <4 x i32> %52, ptr %54, align 4, !tbaa !11
  %55 = add nuw i64 %50, 8
  %56 = add <4 x i32> %51, splat (i32 8)
  %57 = icmp eq i64 %55, %48
  br i1 %57, label %58, label %49, !llvm.loop !13

58:                                               ; preds = %49
  %59 = icmp eq i64 %48, %45
  br i1 %59, label %62, label %60

60:                                               ; preds = %44, %58
  %61 = phi i64 [ 0, %44 ], [ %48, %58 ]
  br label %65

62:                                               ; preds = %65, %58
  %63 = zext nneg i32 %39 to i64
  %64 = shl nuw nsw i64 %63, 2
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 %40, ptr nonnull align 4 %37, i64 %64, i1 false), !tbaa !11
  br label %73

65:                                               ; preds = %60, %65
  %66 = phi i64 [ %69, %65 ], [ %61, %60 ]
  %67 = getelementptr inbounds nuw i32, ptr %37, i64 %66
  %68 = trunc nuw nsw i64 %66 to i32
  store i32 %68, ptr %67, align 4, !tbaa !11
  %69 = add nuw nsw i64 %66, 1
  %70 = icmp eq i64 %69, %45
  br i1 %70, label %62, label %65, !llvm.loop !17

71:                                               ; preds = %25
  %72 = landingpad { ptr, i32 }
          cleanup
  br label %127

73:                                               ; preds = %36, %62
  %74 = getelementptr inbounds i8, ptr %42, i64 -4
  %75 = load i32, ptr %74, align 4, !tbaa !11
  %76 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i32 noundef %75)
          to label %77 unwind label %118

77:                                               ; preds = %73
  %78 = load ptr, ptr %76, align 8, !tbaa !18
  %79 = getelementptr i8, ptr %78, i64 -24
  %80 = load i64, ptr %79, align 8
  %81 = getelementptr inbounds i8, ptr %76, i64 %80
  %82 = getelementptr inbounds nuw i8, ptr %81, i64 240
  %83 = load ptr, ptr %82, align 8, !tbaa !20
  %84 = icmp eq ptr %83, null
  br i1 %84, label %85, label %87

85:                                               ; preds = %77
  invoke void @_ZSt16__throw_bad_castv() #9
          to label %86 unwind label %118

86:                                               ; preds = %85
  unreachable

87:                                               ; preds = %77
  %88 = getelementptr inbounds nuw i8, ptr %83, i64 56
  %89 = load i8, ptr %88, align 8, !tbaa !37
  %90 = icmp eq i8 %89, 0
  br i1 %90, label %94, label %91

91:                                               ; preds = %87
  %92 = getelementptr inbounds nuw i8, ptr %83, i64 67
  %93 = load i8, ptr %92, align 1, !tbaa !43
  br label %100

94:                                               ; preds = %87
  invoke void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %83)
          to label %95 unwind label %118

95:                                               ; preds = %94
  %96 = load ptr, ptr %83, align 8, !tbaa !18
  %97 = getelementptr inbounds nuw i8, ptr %96, i64 48
  %98 = load ptr, ptr %97, align 8
  %99 = invoke noundef i8 %98(ptr noundef nonnull align 8 dereferenceable(570) %83, i8 noundef 10)
          to label %100 unwind label %118

100:                                              ; preds = %95, %91
  %101 = phi i8 [ %93, %91 ], [ %99, %95 ]
  %102 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) %76, i8 noundef %101)
          to label %103 unwind label %118

103:                                              ; preds = %100
  %104 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %102)
          to label %105 unwind label %118

105:                                              ; preds = %103
  %106 = icmp eq ptr %40, null
  br i1 %106, label %111, label %107

107:                                              ; preds = %105
  %108 = ptrtoint ptr %41 to i64
  %109 = ptrtoint ptr %40 to i64
  %110 = sub i64 %108, %109
  tail call void @_ZdlPvm(ptr noundef nonnull %40, i64 noundef %110) #11
  br label %111

111:                                              ; preds = %105, %107
  %112 = icmp eq ptr %37, null
  br i1 %112, label %117, label %113

113:                                              ; preds = %111
  %114 = ptrtoint ptr %38 to i64
  %115 = ptrtoint ptr %37 to i64
  %116 = sub i64 %114, %115
  tail call void @_ZdlPvm(ptr noundef nonnull %37, i64 noundef %116) #11
  br label %117

117:                                              ; preds = %111, %113
  ret i32 0

118:                                              ; preds = %103, %100, %95, %94, %85, %73
  %119 = landingpad { ptr, i32 }
          cleanup
  %120 = icmp eq ptr %40, null
  br i1 %120, label %125, label %121

121:                                              ; preds = %118
  %122 = ptrtoint ptr %41 to i64
  %123 = ptrtoint ptr %40 to i64
  %124 = sub i64 %122, %123
  tail call void @_ZdlPvm(ptr noundef nonnull %40, i64 noundef %124) #11
  br label %125

125:                                              ; preds = %121, %118
  %126 = icmp eq ptr %37, null
  br i1 %126, label %134, label %127

127:                                              ; preds = %71, %125
  %128 = phi { ptr, i32 } [ %72, %71 ], [ %119, %125 ]
  %129 = phi ptr [ %20, %71 ], [ %38, %125 ]
  %130 = phi ptr [ %19, %71 ], [ %37, %125 ]
  %131 = ptrtoint ptr %129 to i64
  %132 = ptrtoint ptr %130 to i64
  %133 = sub i64 %131, %132
  tail call void @_ZdlPvm(ptr noundef nonnull %130, i64 noundef %133) #11
  br label %134

134:                                              ; preds = %127, %125
  %135 = phi { ptr, i32 } [ %119, %125 ], [ %128, %127 ]
  resume { ptr, i32 } %135
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

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #7

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #7 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
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
!13 = distinct !{!13, !14, !15, !16}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!"llvm.loop.isvectorized", i32 1}
!16 = !{!"llvm.loop.unroll.runtime.disable"}
!17 = distinct !{!17, !14, !16, !15}
!18 = !{!19, !19, i64 0}
!19 = !{!"vtable pointer", !10, i64 0}
!20 = !{!21, !34, i64 240}
!21 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !22, i64 0, !31, i64 216, !9, i64 224, !32, i64 225, !33, i64 232, !34, i64 240, !35, i64 248, !36, i64 256}
!22 = !{!"_ZTSSt8ios_base", !23, i64 8, !23, i64 16, !24, i64 24, !25, i64 28, !25, i64 32, !26, i64 40, !27, i64 48, !9, i64 64, !12, i64 192, !28, i64 200, !29, i64 208}
!23 = !{!"long", !9, i64 0}
!24 = !{!"_ZTSSt13_Ios_Fmtflags", !9, i64 0}
!25 = !{!"_ZTSSt12_Ios_Iostate", !9, i64 0}
!26 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !8, i64 0}
!27 = !{!"_ZTSNSt8ios_base6_WordsE", !8, i64 0, !23, i64 8}
!28 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !8, i64 0}
!29 = !{!"_ZTSSt6locale", !30, i64 0}
!30 = !{!"p1 _ZTSNSt6locale5_ImplE", !8, i64 0}
!31 = !{!"p1 _ZTSSo", !8, i64 0}
!32 = !{!"bool", !9, i64 0}
!33 = !{!"p1 _ZTSSt15basic_streambufIcSt11char_traitsIcEE", !8, i64 0}
!34 = !{!"p1 _ZTSSt5ctypeIcE", !8, i64 0}
!35 = !{!"p1 _ZTSSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!36 = !{!"p1 _ZTSSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!37 = !{!38, !9, i64 56}
!38 = !{!"_ZTSSt5ctypeIcE", !39, i64 0, !40, i64 16, !32, i64 24, !41, i64 32, !41, i64 40, !42, i64 48, !9, i64 56, !9, i64 57, !9, i64 313, !9, i64 569}
!39 = !{!"_ZTSNSt6locale5facetE", !12, i64 8}
!40 = !{!"p1 _ZTS15__locale_struct", !8, i64 0}
!41 = !{!"p1 int", !8, i64 0}
!42 = !{!"p1 short", !8, i64 0}
!43 = !{!9, !9, i64 0}
