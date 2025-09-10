; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/sieve.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/sieve.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }
%"class.std::vector" = type { %"struct.std::_Vector_base" }
%"struct.std::_Vector_base" = type { %"struct.std::_Vector_base<int, std::allocator<int>>::_Vector_impl" }
%"struct.std::_Vector_base<int, std::allocator<int>>::_Vector_impl" = type { %"struct.std::_Vector_base<int, std::allocator<int>>::_Vector_impl_data" }
%"struct.std::_Vector_base<int, std::allocator<int>>::_Vector_impl_data" = type { ptr, ptr, ptr }
%"class.std::__cxx11::list" = type { %"class.std::__cxx11::_List_base" }
%"class.std::__cxx11::_List_base" = type { %"struct.std::__cxx11::_List_base<int, std::allocator<int>>::_List_impl" }
%"struct.std::__cxx11::_List_base<int, std::allocator<int>>::_List_impl" = type { %"struct.std::__detail::_List_node_header" }
%"struct.std::__detail::_List_node_header" = type { %"struct.std::__detail::_List_node_base", i64 }
%"struct.std::__detail::_List_node_base" = type { ptr, ptr }

@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [8 x i8] c"Count: \00", align 1
@.str.1 = private unnamed_addr constant [26 x i8] c"vector::_M_realloc_append\00", align 1

; Function Attrs: mustprogress uwtable
define dso_local void @_Z5sieveRNSt7__cxx114listIiSaIiEEERSt6vectorIiS1_E(ptr noundef nonnull align 8 captures(address) dereferenceable(24) %0, ptr noundef nonnull align 8 captures(none) dereferenceable(24) %1) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %3 = load ptr, ptr %0, align 8, !tbaa !6
  %4 = icmp eq ptr %3, %0
  br i1 %4, label %64, label %5

5:                                                ; preds = %2
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 16
  br label %9

9:                                                ; preds = %5, %61
  %10 = phi ptr [ %3, %5 ], [ %62, %61 ]
  %11 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %12 = load i32, ptr %11, align 4, !tbaa !12
  %13 = load i64, ptr %6, align 8, !tbaa !14
  %14 = add i64 %13, -1
  store i64 %14, ptr %6, align 8, !tbaa !14
  tail call void @_ZNSt8__detail15_List_node_base9_M_unhookEv(ptr noundef nonnull align 8 dereferenceable(16) %10) #12
  tail call void @_ZdlPvm(ptr noundef nonnull %10, i64 noundef 24) #13
  %15 = load ptr, ptr %0, align 8, !tbaa !6
  %16 = icmp eq ptr %15, %0
  br i1 %16, label %29, label %17

17:                                               ; preds = %9, %27
  %18 = phi ptr [ %23, %27 ], [ %15, %9 ]
  %19 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %20 = load i32, ptr %19, align 4, !tbaa !12
  %21 = srem i32 %20, %12
  %22 = icmp eq i32 %21, 0
  %23 = load ptr, ptr %18, align 8, !tbaa !6
  br i1 %22, label %24, label %27

24:                                               ; preds = %17
  %25 = load i64, ptr %6, align 8, !tbaa !14
  %26 = add i64 %25, -1
  store i64 %26, ptr %6, align 8, !tbaa !14
  tail call void @_ZNSt8__detail15_List_node_base9_M_unhookEv(ptr noundef nonnull align 8 dereferenceable(16) %18) #12
  tail call void @_ZdlPvm(ptr noundef nonnull %18, i64 noundef 24) #13
  br label %27

27:                                               ; preds = %17, %24
  %28 = icmp eq ptr %23, %0
  br i1 %28, label %29, label %17, !llvm.loop !19

29:                                               ; preds = %27, %9
  %30 = load ptr, ptr %7, align 8, !tbaa !21
  %31 = load ptr, ptr %8, align 8, !tbaa !24
  %32 = icmp eq ptr %30, %31
  br i1 %32, label %35, label %33

33:                                               ; preds = %29
  store i32 %12, ptr %30, align 4, !tbaa !12
  %34 = getelementptr inbounds nuw i8, ptr %30, i64 4
  store ptr %34, ptr %7, align 8, !tbaa !21
  br label %61

35:                                               ; preds = %29
  %36 = load ptr, ptr %1, align 8, !tbaa !25
  %37 = ptrtoint ptr %30 to i64
  %38 = ptrtoint ptr %36 to i64
  %39 = sub i64 %37, %38
  %40 = icmp eq i64 %39, 9223372036854775804
  br i1 %40, label %41, label %42

41:                                               ; preds = %35
  tail call void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.1) #14
  unreachable

42:                                               ; preds = %35
  %43 = ashr exact i64 %39, 2
  %44 = tail call i64 @llvm.umax.i64(i64 %43, i64 1)
  %45 = add nsw i64 %44, %43
  %46 = icmp ult i64 %45, %43
  %47 = tail call i64 @llvm.umin.i64(i64 %45, i64 2305843009213693951)
  %48 = select i1 %46, i64 2305843009213693951, i64 %47
  %49 = icmp ne i64 %48, 0
  tail call void @llvm.assume(i1 %49)
  %50 = shl nuw nsw i64 %48, 2
  %51 = tail call noalias noundef nonnull ptr @_Znwm(i64 noundef %50) #15
  %52 = getelementptr inbounds i8, ptr %51, i64 %39
  store i32 %12, ptr %52, align 4, !tbaa !12
  %53 = icmp sgt i64 %39, 0
  br i1 %53, label %54, label %55

54:                                               ; preds = %42
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 4 %51, ptr align 4 %36, i64 %39, i1 false)
  br label %55

55:                                               ; preds = %54, %42
  %56 = icmp eq ptr %36, null
  br i1 %56, label %58, label %57

57:                                               ; preds = %55
  tail call void @_ZdlPvm(ptr noundef nonnull %36, i64 noundef %39) #13
  br label %58

58:                                               ; preds = %57, %55
  %59 = getelementptr inbounds nuw i8, ptr %52, i64 4
  store ptr %51, ptr %1, align 8, !tbaa !25
  store ptr %59, ptr %7, align 8, !tbaa !21
  %60 = getelementptr inbounds nuw i32, ptr %51, i64 %48
  store ptr %60, ptr %8, align 8, !tbaa !24
  br label %61

61:                                               ; preds = %33, %58
  %62 = load ptr, ptr %0, align 8, !tbaa !6
  %63 = icmp eq ptr %62, %0
  br i1 %63, label %64, label %9, !llvm.loop !26

64:                                               ; preds = %61, %2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #3 personality ptr @__gxx_personality_v0 {
  %3 = alloca %"class.std::vector", align 8
  %4 = alloca %"class.std::__cxx11::list", align 8
  %5 = icmp eq i32 %0, 2
  br i1 %5, label %6, label %12

6:                                                ; preds = %2
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %8 = load ptr, ptr %7, align 8, !tbaa !27
  %9 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %8, ptr noundef null, i32 noundef 10) #12
  %10 = trunc i64 %9 to i32
  %11 = icmp slt i32 %10, 1
  br i1 %11, label %12, label %14

12:                                               ; preds = %2, %6
  %13 = phi i64 [ 500, %2 ], [ 1, %6 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %3, i8 0, i64 24, i1 false)
  br label %17

14:                                               ; preds = %6
  %15 = and i64 %9, 2147483647
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %3, i8 0, i64 24, i1 false)
  %16 = icmp eq i64 %15, 0
  br i1 %16, label %58, label %17

17:                                               ; preds = %12, %14
  %18 = phi i64 [ %13, %12 ], [ %15, %14 ]
  %19 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %20 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %21 = getelementptr inbounds nuw i8, ptr %3, i64 8
  br label %22

22:                                               ; preds = %17, %45
  %23 = phi i64 [ %18, %17 ], [ %24, %45 ]
  %24 = add nsw i64 %23, -1
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #12
  store ptr %4, ptr %19, align 8, !tbaa !29
  store ptr %4, ptr %4, align 8, !tbaa !6
  store i64 0, ptr %20, align 8, !tbaa !30
  br label %27

25:                                               ; preds = %30
  %26 = load ptr, ptr %3, align 8, !tbaa !25
  store ptr %26, ptr %21, align 8
  invoke void @_Z5sieveRNSt7__cxx114listIiSaIiEEERSt6vectorIiS1_E(ptr noundef nonnull align 8 dereferenceable(24) %4, ptr noundef nonnull align 8 dereferenceable(24) %3)
          to label %38 unwind label %47

27:                                               ; preds = %22, %30
  %28 = phi i32 [ 2, %22 ], [ %34, %30 ]
  %29 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #15
          to label %30 unwind label %36

30:                                               ; preds = %27
  %31 = getelementptr inbounds nuw i8, ptr %29, i64 16
  store i32 %28, ptr %31, align 4, !tbaa !12
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16) %29, ptr noundef nonnull align 8 dereferenceable(24) %4) #12
  %32 = load i64, ptr %20, align 8, !tbaa !14
  %33 = add i64 %32, 1
  store i64 %33, ptr %20, align 8, !tbaa !14
  %34 = add nuw nsw i32 %28, 1
  %35 = icmp eq i32 %34, 8192
  br i1 %35, label %25, label %27, !llvm.loop !31

36:                                               ; preds = %27
  %37 = landingpad { ptr, i32 }
          cleanup
  br label %49

38:                                               ; preds = %25
  %39 = load ptr, ptr %4, align 8, !tbaa !6
  %40 = icmp eq ptr %39, %4
  br i1 %40, label %45, label %41

41:                                               ; preds = %38, %41
  %42 = phi ptr [ %43, %41 ], [ %39, %38 ]
  %43 = load ptr, ptr %42, align 8, !tbaa !6
  call void @_ZdlPvm(ptr noundef nonnull %42, i64 noundef 24) #13
  %44 = icmp eq ptr %43, %4
  br i1 %44, label %45, label %41, !llvm.loop !32

45:                                               ; preds = %41, %38
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #12
  %46 = icmp eq i64 %24, 0
  br i1 %46, label %58, label %22, !llvm.loop !33

47:                                               ; preds = %25
  %48 = landingpad { ptr, i32 }
          cleanup
  br label %49

49:                                               ; preds = %47, %36
  %50 = phi { ptr, i32 } [ %37, %36 ], [ %48, %47 ]
  %51 = load ptr, ptr %4, align 8, !tbaa !6
  %52 = icmp eq ptr %51, %4
  br i1 %52, label %57, label %53

53:                                               ; preds = %49, %53
  %54 = phi ptr [ %55, %53 ], [ %51, %49 ]
  %55 = load ptr, ptr %54, align 8, !tbaa !6
  call void @_ZdlPvm(ptr noundef nonnull %54, i64 noundef 24) #13
  %56 = icmp eq ptr %55, %4
  br i1 %56, label %57, label %53, !llvm.loop !32

57:                                               ; preds = %53, %49
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #12
  br label %107

58:                                               ; preds = %45, %14
  %59 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str, i64 noundef 7)
          to label %60 unwind label %105

60:                                               ; preds = %58
  %61 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %62 = load ptr, ptr %61, align 8, !tbaa !21
  %63 = load ptr, ptr %3, align 8, !tbaa !25
  %64 = ptrtoint ptr %62 to i64
  %65 = ptrtoint ptr %63 to i64
  %66 = sub i64 %64, %65
  %67 = ashr exact i64 %66, 2
  %68 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i64 noundef %67)
          to label %69 unwind label %105

69:                                               ; preds = %60
  %70 = load ptr, ptr %68, align 8, !tbaa !34
  %71 = getelementptr i8, ptr %70, i64 -24
  %72 = load i64, ptr %71, align 8
  %73 = getelementptr inbounds i8, ptr %68, i64 %72
  %74 = getelementptr inbounds nuw i8, ptr %73, i64 240
  %75 = load ptr, ptr %74, align 8, !tbaa !36
  %76 = icmp eq ptr %75, null
  br i1 %76, label %77, label %79

77:                                               ; preds = %69
  invoke void @_ZSt16__throw_bad_castv() #14
          to label %78 unwind label %105

78:                                               ; preds = %77
  unreachable

79:                                               ; preds = %69
  %80 = getelementptr inbounds nuw i8, ptr %75, i64 56
  %81 = load i8, ptr %80, align 8, !tbaa !52
  %82 = icmp eq i8 %81, 0
  br i1 %82, label %86, label %83

83:                                               ; preds = %79
  %84 = getelementptr inbounds nuw i8, ptr %75, i64 67
  %85 = load i8, ptr %84, align 1, !tbaa !57
  br label %92

86:                                               ; preds = %79
  invoke void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %75)
          to label %87 unwind label %105

87:                                               ; preds = %86
  %88 = load ptr, ptr %75, align 8, !tbaa !34
  %89 = getelementptr inbounds nuw i8, ptr %88, i64 48
  %90 = load ptr, ptr %89, align 8
  %91 = invoke noundef i8 %90(ptr noundef nonnull align 8 dereferenceable(570) %75, i8 noundef 10)
          to label %92 unwind label %105

92:                                               ; preds = %87, %83
  %93 = phi i8 [ %85, %83 ], [ %91, %87 ]
  %94 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) %68, i8 noundef %93)
          to label %95 unwind label %105

95:                                               ; preds = %92
  %96 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %94)
          to label %97 unwind label %105

97:                                               ; preds = %95
  %98 = icmp eq ptr %63, null
  br i1 %98, label %104, label %99

99:                                               ; preds = %97
  %100 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %101 = load ptr, ptr %100, align 8, !tbaa !24
  %102 = ptrtoint ptr %101 to i64
  %103 = sub i64 %102, %65
  call void @_ZdlPvm(ptr noundef nonnull %63, i64 noundef %103) #13
  br label %104

104:                                              ; preds = %97, %99
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #12
  ret i32 0

105:                                              ; preds = %95, %92, %87, %86, %77, %60, %58
  %106 = landingpad { ptr, i32 }
          cleanup
  br label %107

107:                                              ; preds = %105, %57
  %108 = phi { ptr, i32 } [ %50, %57 ], [ %106, %105 ]
  %109 = load ptr, ptr %3, align 8, !tbaa !25
  %110 = icmp eq ptr %109, null
  br i1 %110, label %117, label %111

111:                                              ; preds = %107
  %112 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %113 = load ptr, ptr %112, align 8, !tbaa !24
  %114 = ptrtoint ptr %113 to i64
  %115 = ptrtoint ptr %109 to i64
  %116 = sub i64 %114, %115
  call void @_ZdlPvm(ptr noundef nonnull %109, i64 noundef %116) #13
  br label %117

117:                                              ; preds = %107, %111
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #12
  resume { ptr, i32 } %108
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #4

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #5

; Function Attrs: nounwind
declare void @_ZNSt8__detail15_List_node_base9_M_unhookEv(ptr noundef nonnull align 8 dereferenceable(16)) local_unnamed_addr #4

; Function Attrs: cold noreturn
declare void @_ZSt20__throw_length_errorPKc(ptr noundef) local_unnamed_addr #6

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #7

; Function Attrs: nounwind
declare void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16), ptr noundef) local_unnamed_addr #4

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #8

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8), i64 noundef) local_unnamed_addr #8

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #8

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #8

; Function Attrs: cold noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #6

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570)) local_unnamed_addr #8

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #9

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #10

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #10

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #11

attributes #0 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #10 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #11 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #12 = { nounwind }
attributes #13 = { builtin nounwind }
attributes #14 = { cold noreturn }
attributes #15 = { builtin allocsize(0) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"_ZTSNSt8__detail15_List_node_baseE", !8, i64 0, !8, i64 8}
!8 = !{!"p1 _ZTSNSt8__detail15_List_node_baseE", !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
!12 = !{!13, !13, i64 0}
!13 = !{!"int", !10, i64 0}
!14 = !{!15, !18, i64 16}
!15 = !{!"_ZTSNSt7__cxx1110_List_baseIiSaIiEEE", !16, i64 0}
!16 = !{!"_ZTSNSt7__cxx1110_List_baseIiSaIiEE10_List_implE", !17, i64 0}
!17 = !{!"_ZTSNSt8__detail17_List_node_headerE", !7, i64 0, !18, i64 16}
!18 = !{!"long", !10, i64 0}
!19 = distinct !{!19, !20}
!20 = !{!"llvm.loop.mustprogress"}
!21 = !{!22, !23, i64 8}
!22 = !{!"_ZTSNSt12_Vector_baseIiSaIiEE17_Vector_impl_dataE", !23, i64 0, !23, i64 8, !23, i64 16}
!23 = !{!"p1 int", !9, i64 0}
!24 = !{!22, !23, i64 16}
!25 = !{!22, !23, i64 0}
!26 = distinct !{!26, !20}
!27 = !{!28, !28, i64 0}
!28 = !{!"p1 omnipotent char", !9, i64 0}
!29 = !{!7, !8, i64 8}
!30 = !{!17, !18, i64 16}
!31 = distinct !{!31, !20}
!32 = distinct !{!32, !20}
!33 = distinct !{!33, !20}
!34 = !{!35, !35, i64 0}
!35 = !{!"vtable pointer", !11, i64 0}
!36 = !{!37, !49, i64 240}
!37 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !38, i64 0, !46, i64 216, !10, i64 224, !47, i64 225, !48, i64 232, !49, i64 240, !50, i64 248, !51, i64 256}
!38 = !{!"_ZTSSt8ios_base", !18, i64 8, !18, i64 16, !39, i64 24, !40, i64 28, !40, i64 32, !41, i64 40, !42, i64 48, !10, i64 64, !13, i64 192, !43, i64 200, !44, i64 208}
!39 = !{!"_ZTSSt13_Ios_Fmtflags", !10, i64 0}
!40 = !{!"_ZTSSt12_Ios_Iostate", !10, i64 0}
!41 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !9, i64 0}
!42 = !{!"_ZTSNSt8ios_base6_WordsE", !9, i64 0, !18, i64 8}
!43 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !9, i64 0}
!44 = !{!"_ZTSSt6locale", !45, i64 0}
!45 = !{!"p1 _ZTSNSt6locale5_ImplE", !9, i64 0}
!46 = !{!"p1 _ZTSSo", !9, i64 0}
!47 = !{!"bool", !10, i64 0}
!48 = !{!"p1 _ZTSSt15basic_streambufIcSt11char_traitsIcEE", !9, i64 0}
!49 = !{!"p1 _ZTSSt5ctypeIcE", !9, i64 0}
!50 = !{!"p1 _ZTSSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE", !9, i64 0}
!51 = !{!"p1 _ZTSSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE", !9, i64 0}
!52 = !{!53, !10, i64 56}
!53 = !{!"_ZTSSt5ctypeIcE", !54, i64 0, !55, i64 16, !47, i64 24, !23, i64 32, !23, i64 40, !56, i64 48, !10, i64 56, !10, i64 57, !10, i64 313, !10, i64 569}
!54 = !{!"_ZTSNSt6locale5facetE", !13, i64 8}
!55 = !{!"p1 _ZTS15__locale_struct", !9, i64 0}
!56 = !{!"p1 short", !9, i64 0}
!57 = !{!10, !10, i64 0}
