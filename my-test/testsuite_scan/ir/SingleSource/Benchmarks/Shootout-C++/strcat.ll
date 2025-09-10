; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/strcat.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/strcat.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }
%"class.std::__cxx11::basic_string" = type { %"struct.std::__cxx11::basic_string<char>::_Alloc_hider", i64, %union.anon }
%"struct.std::__cxx11::basic_string<char>::_Alloc_hider" = type { ptr }
%union.anon = type { i64, [8 x i8] }

@.str = private unnamed_addr constant [7 x i8] c"hello\0A\00", align 1
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str.1 = private unnamed_addr constant [21 x i8] c"basic_string::append\00", align 1

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %3 = alloca %"class.std::__cxx11::basic_string", align 8
  %4 = icmp eq i32 %0, 2
  br i1 %4, label %5, label %10

5:                                                ; preds = %2
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %7 = load ptr, ptr %6, align 8, !tbaa !6
  %8 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %7, ptr noundef null, i32 noundef 10) #7
  %9 = trunc i64 %8 to i32
  br label %10

10:                                               ; preds = %2, %5
  %11 = phi i32 [ %9, %5 ], [ 5000000, %2 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #7
  %12 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store ptr %12, ptr %3, align 8, !tbaa !11
  %13 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store i64 0, ptr %13, align 8, !tbaa !13
  store i8 0, ptr %12, align 8, !tbaa !16
  invoke void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm(ptr noundef nonnull align 8 dereferenceable(32) %3, i64 noundef 31)
          to label %14 unwind label %23

14:                                               ; preds = %10
  %15 = icmp sgt i32 %11, 0
  br i1 %15, label %16, label %42

16:                                               ; preds = %14, %38
  %17 = phi i64 [ %39, %38 ], [ 6, %14 ]
  %18 = phi i64 [ %30, %38 ], [ 31, %14 ]
  %19 = phi i32 [ %40, %38 ], [ 0, %14 ]
  %20 = icmp ugt i64 %17, %18
  br i1 %20, label %21, label %29

21:                                               ; preds = %16
  %22 = shl i64 %18, 1
  invoke void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm(ptr noundef nonnull align 8 dereferenceable(32) %3, i64 noundef %22)
          to label %29 unwind label %25

23:                                               ; preds = %10
  %24 = landingpad { ptr, i32 }
          cleanup
  br label %83

25:                                               ; preds = %21, %36
  %26 = landingpad { ptr, i32 }
          cleanup
  br label %83

27:                                               ; preds = %34, %42, %53, %62, %63, %68, %71
  %28 = landingpad { ptr, i32 }
          cleanup
  br label %83

29:                                               ; preds = %21, %16
  %30 = phi i64 [ %22, %21 ], [ %18, %16 ]
  %31 = load i64, ptr %13, align 8, !tbaa !13
  %32 = add i64 %31, -4611686018427387898
  %33 = icmp ult i64 %32, 6
  br i1 %33, label %34, label %36

34:                                               ; preds = %29
  invoke void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.1) #8
          to label %35 unwind label %27

35:                                               ; preds = %34
  unreachable

36:                                               ; preds = %29
  %37 = invoke noundef nonnull align 8 dereferenceable(32) ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_appendEPKcm(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull @.str, i64 noundef 6)
          to label %38 unwind label %25

38:                                               ; preds = %36
  %39 = add nuw nsw i64 %17, 6
  %40 = add nuw nsw i32 %19, 1
  %41 = icmp eq i32 %40, %11
  br i1 %41, label %42, label %16, !llvm.loop !17

42:                                               ; preds = %38, %14
  %43 = load i64, ptr %13, align 8, !tbaa !13
  %44 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i64 noundef %43)
          to label %45 unwind label %27

45:                                               ; preds = %42
  %46 = load ptr, ptr %44, align 8, !tbaa !19
  %47 = getelementptr i8, ptr %46, i64 -24
  %48 = load i64, ptr %47, align 8
  %49 = getelementptr inbounds i8, ptr %44, i64 %48
  %50 = getelementptr inbounds nuw i8, ptr %49, i64 240
  %51 = load ptr, ptr %50, align 8, !tbaa !21
  %52 = icmp eq ptr %51, null
  br i1 %52, label %53, label %55

53:                                               ; preds = %45
  invoke void @_ZSt16__throw_bad_castv() #8
          to label %54 unwind label %27

54:                                               ; preds = %53
  unreachable

55:                                               ; preds = %45
  %56 = getelementptr inbounds nuw i8, ptr %51, i64 56
  %57 = load i8, ptr %56, align 8, !tbaa !38
  %58 = icmp eq i8 %57, 0
  br i1 %58, label %62, label %59

59:                                               ; preds = %55
  %60 = getelementptr inbounds nuw i8, ptr %51, i64 67
  %61 = load i8, ptr %60, align 1, !tbaa !16
  br label %68

62:                                               ; preds = %55
  invoke void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %51)
          to label %63 unwind label %27

63:                                               ; preds = %62
  %64 = load ptr, ptr %51, align 8, !tbaa !19
  %65 = getelementptr inbounds nuw i8, ptr %64, i64 48
  %66 = load ptr, ptr %65, align 8
  %67 = invoke noundef i8 %66(ptr noundef nonnull align 8 dereferenceable(570) %51, i8 noundef 10)
          to label %68 unwind label %27

68:                                               ; preds = %63, %59
  %69 = phi i8 [ %61, %59 ], [ %67, %63 ]
  %70 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) %44, i8 noundef %69)
          to label %71 unwind label %27

71:                                               ; preds = %68
  %72 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %70)
          to label %73 unwind label %27

73:                                               ; preds = %71
  %74 = load ptr, ptr %3, align 8, !tbaa !44
  %75 = icmp eq ptr %74, %12
  br i1 %75, label %76, label %79

76:                                               ; preds = %73
  %77 = load i64, ptr %13, align 8, !tbaa !13
  %78 = icmp ult i64 %77, 16
  call void @llvm.assume(i1 %78)
  br label %82

79:                                               ; preds = %73
  %80 = load i64, ptr %12, align 8, !tbaa !16
  %81 = add i64 %80, 1
  call void @_ZdlPvm(ptr noundef %74, i64 noundef %81) #9
  br label %82

82:                                               ; preds = %76, %79
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #7
  ret i32 0

83:                                               ; preds = %25, %27, %23
  %84 = phi { ptr, i32 } [ %24, %23 ], [ %26, %25 ], [ %28, %27 ]
  %85 = load ptr, ptr %3, align 8, !tbaa !44
  %86 = icmp eq ptr %85, %12
  br i1 %86, label %87, label %90

87:                                               ; preds = %83
  %88 = load i64, ptr %13, align 8, !tbaa !13
  %89 = icmp ult i64 %88, 16
  call void @llvm.assume(i1 %89)
  br label %93

90:                                               ; preds = %83
  %91 = load i64, ptr %12, align 8, !tbaa !16
  %92 = add i64 %91, 1
  call void @_ZdlPvm(ptr noundef %85, i64 noundef %92) #9
  br label %93

93:                                               ; preds = %87, %90
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #7
  resume { ptr, i32 } %84
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

declare void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm(ptr noundef nonnull align 8 dereferenceable(32), i64 noundef) local_unnamed_addr #2

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #3

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #4

declare noundef nonnull align 8 dereferenceable(32) ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_appendEPKcm(ptr noundef nonnull align 8 dereferenceable(32), ptr noundef, i64 noundef) local_unnamed_addr #2

; Function Attrs: cold noreturn
declare void @_ZSt20__throw_length_errorPKc(ptr noundef) local_unnamed_addr #5

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8), i64 noundef) local_unnamed_addr #2

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #2

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #2

; Function Attrs: cold noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #5

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570)) local_unnamed_addr #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #6

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #7 = { nounwind }
attributes #8 = { cold noreturn }
attributes #9 = { builtin nounwind }

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
!11 = !{!12, !7, i64 0}
!12 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderE", !7, i64 0}
!13 = !{!14, !15, i64 8}
!14 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", !12, i64 0, !15, i64 8, !9, i64 16}
!15 = !{!"long", !9, i64 0}
!16 = !{!9, !9, i64 0}
!17 = distinct !{!17, !18}
!18 = !{!"llvm.loop.mustprogress"}
!19 = !{!20, !20, i64 0}
!20 = !{!"vtable pointer", !10, i64 0}
!21 = !{!22, !35, i64 240}
!22 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !23, i64 0, !32, i64 216, !9, i64 224, !33, i64 225, !34, i64 232, !35, i64 240, !36, i64 248, !37, i64 256}
!23 = !{!"_ZTSSt8ios_base", !15, i64 8, !15, i64 16, !24, i64 24, !25, i64 28, !25, i64 32, !26, i64 40, !27, i64 48, !9, i64 64, !28, i64 192, !29, i64 200, !30, i64 208}
!24 = !{!"_ZTSSt13_Ios_Fmtflags", !9, i64 0}
!25 = !{!"_ZTSSt12_Ios_Iostate", !9, i64 0}
!26 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !8, i64 0}
!27 = !{!"_ZTSNSt8ios_base6_WordsE", !8, i64 0, !15, i64 8}
!28 = !{!"int", !9, i64 0}
!29 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !8, i64 0}
!30 = !{!"_ZTSSt6locale", !31, i64 0}
!31 = !{!"p1 _ZTSNSt6locale5_ImplE", !8, i64 0}
!32 = !{!"p1 _ZTSSo", !8, i64 0}
!33 = !{!"bool", !9, i64 0}
!34 = !{!"p1 _ZTSSt15basic_streambufIcSt11char_traitsIcEE", !8, i64 0}
!35 = !{!"p1 _ZTSSt5ctypeIcE", !8, i64 0}
!36 = !{!"p1 _ZTSSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!37 = !{!"p1 _ZTSSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!38 = !{!39, !9, i64 56}
!39 = !{!"_ZTSSt5ctypeIcE", !40, i64 0, !41, i64 16, !33, i64 24, !42, i64 32, !42, i64 40, !43, i64 48, !9, i64 56, !9, i64 57, !9, i64 313, !9, i64 569}
!40 = !{!"_ZTSNSt6locale5facetE", !28, i64 8}
!41 = !{!"p1 _ZTS15__locale_struct", !8, i64 0}
!42 = !{!"p1 int", !8, i64 0}
!43 = !{!"p1 short", !8, i64 0}
!44 = !{!14, !7, i64 0}
