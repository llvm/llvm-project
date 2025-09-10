; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/methcall.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/methcall.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }

$_ZN6ToggleD0Ev = comdat any

$_ZN6Toggle8activateEv = comdat any

$_ZN6ToggleD2Ev = comdat any

$_ZN9NthToggleD0Ev = comdat any

$_ZN9NthToggle8activateEv = comdat any

$_ZTV6Toggle = comdat any

$_ZTI6Toggle = comdat any

$_ZTS6Toggle = comdat any

$_ZTV9NthToggle = comdat any

$_ZTI9NthToggle = comdat any

$_ZTS9NthToggle = comdat any

@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [5 x i8] c"true\00", align 1
@.str.1 = private unnamed_addr constant [6 x i8] c"false\00", align 1
@_ZTV6Toggle = linkonce_odr dso_local unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI6Toggle, ptr @_ZN6ToggleD2Ev, ptr @_ZN6ToggleD0Ev, ptr @_ZN6Toggle8activateEv] }, comdat, align 8
@_ZTI6Toggle = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS6Toggle }, comdat, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@_ZTS6Toggle = linkonce_odr dso_local constant [8 x i8] c"6Toggle\00", comdat, align 1
@_ZTV9NthToggle = linkonce_odr dso_local unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI9NthToggle, ptr @_ZN6ToggleD2Ev, ptr @_ZN9NthToggleD0Ev, ptr @_ZN9NthToggle8activateEv] }, comdat, align 8
@_ZTI9NthToggle = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS9NthToggle, ptr @_ZTI6Toggle }, comdat, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
@_ZTS9NthToggle = linkonce_odr dso_local constant [11 x i8] c"9NthToggle\00", comdat, align 1

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %3 = icmp eq i32 %0, 2
  br i1 %3, label %4, label %9

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !6
  %7 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %6, ptr noundef null, i32 noundef 10) #8
  %8 = trunc i64 %7 to i32
  br label %9

9:                                                ; preds = %2, %4
  %10 = phi i32 [ %8, %4 ], [ 1000000000, %2 ]
  %11 = tail call noalias noundef nonnull dereferenceable(16) ptr @_Znwm(i64 noundef 16) #9
  store ptr getelementptr inbounds nuw inrange(-16, 24) (i8, ptr @_ZTV6Toggle, i64 16), ptr %11, align 8, !tbaa !11
  %12 = getelementptr inbounds nuw i8, ptr %11, i64 8
  store i8 1, ptr %12, align 8, !tbaa !13
  %13 = icmp sgt i32 %10, 0
  br i1 %13, label %54, label %20

14:                                               ; preds = %54
  %15 = getelementptr inbounds nuw i8, ptr %59, i64 8
  %16 = load i8, ptr %15, align 8, !tbaa !13, !range !16, !noundef !17
  %17 = trunc nuw i8 %16 to i1
  %18 = select i1 %17, ptr @.str, ptr @.str.1
  %19 = select i1 %17, i64 4, i64 5
  br label %20

20:                                               ; preds = %14, %9
  %21 = phi ptr [ @.str, %9 ], [ %18, %14 ]
  %22 = phi i64 [ 4, %9 ], [ %19, %14 ]
  %23 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %21, i64 noundef %22)
  %24 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %25 = getelementptr i8, ptr %24, i64 -24
  %26 = load i64, ptr %25, align 8
  %27 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %26
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 240
  %29 = load ptr, ptr %28, align 8, !tbaa !18
  %30 = icmp eq ptr %29, null
  br i1 %30, label %31, label %32

31:                                               ; preds = %20
  tail call void @_ZSt16__throw_bad_castv() #10
  unreachable

32:                                               ; preds = %20
  %33 = getelementptr inbounds nuw i8, ptr %29, i64 56
  %34 = load i8, ptr %33, align 8, !tbaa !35
  %35 = icmp eq i8 %34, 0
  br i1 %35, label %39, label %36

36:                                               ; preds = %32
  %37 = getelementptr inbounds nuw i8, ptr %29, i64 67
  %38 = load i8, ptr %37, align 1, !tbaa !41
  br label %44

39:                                               ; preds = %32
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %29)
  %40 = load ptr, ptr %29, align 8, !tbaa !11
  %41 = getelementptr inbounds nuw i8, ptr %40, i64 48
  %42 = load ptr, ptr %41, align 8
  %43 = tail call noundef i8 %42(ptr noundef nonnull align 8 dereferenceable(570) %29, i8 noundef 10)
  br label %44

44:                                               ; preds = %36, %39
  %45 = phi i8 [ %38, %36 ], [ %43, %39 ]
  %46 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %45)
  %47 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %46)
  %48 = load ptr, ptr %11, align 8, !tbaa !11
  %49 = getelementptr inbounds nuw i8, ptr %48, i64 8
  %50 = load ptr, ptr %49, align 8
  tail call void %50(ptr noundef nonnull align 8 dereferenceable(9) %11) #8
  %51 = tail call noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #9
  %52 = getelementptr inbounds nuw i8, ptr %51, i64 8
  store i8 1, ptr %52, align 8, !tbaa !13
  store ptr getelementptr inbounds nuw inrange(-16, 24) (i8, ptr @_ZTV9NthToggle, i64 16), ptr %51, align 8, !tbaa !11
  %53 = getelementptr inbounds nuw i8, ptr %51, i64 12
  store <2 x i32> <i32 3, i32 0>, ptr %53, align 4, !tbaa !42
  br i1 %13, label %99, label %68

54:                                               ; preds = %9, %54
  %55 = phi i32 [ %60, %54 ], [ 0, %9 ]
  %56 = load ptr, ptr %11, align 8, !tbaa !11
  %57 = getelementptr inbounds nuw i8, ptr %56, i64 16
  %58 = load ptr, ptr %57, align 8
  %59 = tail call noundef nonnull align 8 dereferenceable(9) ptr %58(ptr noundef nonnull align 8 dereferenceable(9) %11)
  %60 = add nuw nsw i32 %55, 1
  %61 = icmp eq i32 %60, %10
  br i1 %61, label %14, label %54, !llvm.loop !43

62:                                               ; preds = %99
  %63 = getelementptr inbounds nuw i8, ptr %104, i64 8
  %64 = load i8, ptr %63, align 8, !tbaa !13, !range !16, !noundef !17
  %65 = trunc nuw i8 %64 to i1
  %66 = select i1 %65, ptr @.str, ptr @.str.1
  %67 = select i1 %65, i64 4, i64 5
  br label %68

68:                                               ; preds = %62, %44
  %69 = phi ptr [ @.str, %44 ], [ %66, %62 ]
  %70 = phi i64 [ 4, %44 ], [ %67, %62 ]
  %71 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %69, i64 noundef %70)
  %72 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %73 = getelementptr i8, ptr %72, i64 -24
  %74 = load i64, ptr %73, align 8
  %75 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %74
  %76 = getelementptr inbounds nuw i8, ptr %75, i64 240
  %77 = load ptr, ptr %76, align 8, !tbaa !18
  %78 = icmp eq ptr %77, null
  br i1 %78, label %79, label %80

79:                                               ; preds = %68
  tail call void @_ZSt16__throw_bad_castv() #10
  unreachable

80:                                               ; preds = %68
  %81 = getelementptr inbounds nuw i8, ptr %77, i64 56
  %82 = load i8, ptr %81, align 8, !tbaa !35
  %83 = icmp eq i8 %82, 0
  br i1 %83, label %87, label %84

84:                                               ; preds = %80
  %85 = getelementptr inbounds nuw i8, ptr %77, i64 67
  %86 = load i8, ptr %85, align 1, !tbaa !41
  br label %92

87:                                               ; preds = %80
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %77)
  %88 = load ptr, ptr %77, align 8, !tbaa !11
  %89 = getelementptr inbounds nuw i8, ptr %88, i64 48
  %90 = load ptr, ptr %89, align 8
  %91 = tail call noundef i8 %90(ptr noundef nonnull align 8 dereferenceable(570) %77, i8 noundef 10)
  br label %92

92:                                               ; preds = %84, %87
  %93 = phi i8 [ %86, %84 ], [ %91, %87 ]
  %94 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %93)
  %95 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %94)
  %96 = load ptr, ptr %51, align 8, !tbaa !11
  %97 = getelementptr inbounds nuw i8, ptr %96, i64 8
  %98 = load ptr, ptr %97, align 8
  tail call void %98(ptr noundef nonnull align 8 dereferenceable(20) %51) #8
  ret i32 0

99:                                               ; preds = %44, %99
  %100 = phi i32 [ %105, %99 ], [ 0, %44 ]
  %101 = load ptr, ptr %51, align 8, !tbaa !11
  %102 = getelementptr inbounds nuw i8, ptr %101, i64 16
  %103 = load ptr, ptr %102, align 8
  %104 = tail call noundef nonnull align 8 dereferenceable(9) ptr %103(ptr noundef nonnull align 8 dereferenceable(20) %51)
  %105 = add nuw nsw i32 %100, 1
  %106 = icmp eq i32 %105, %10
  br i1 %106, label %62, label %99, !llvm.loop !45
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #1

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #2

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN6ToggleD0Ev(ptr noundef nonnull align 8 dereferenceable(9) %0) unnamed_addr #4 comdat {
  tail call void @_ZdlPvm(ptr noundef nonnull %0, i64 noundef 16) #11
  ret void
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(9) ptr @_ZN6Toggle8activateEv(ptr noundef nonnull align 8 dereferenceable(9) %0) unnamed_addr #4 comdat {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %3 = load i8, ptr %2, align 8, !tbaa !13, !range !16, !noundef !17
  %4 = trunc nuw i8 %3 to i1
  %5 = xor i1 %4, true
  %6 = zext i1 %5 to i8
  store i8 %6, ptr %2, align 8, !tbaa !13
  ret ptr %0
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN6ToggleD2Ev(ptr noundef nonnull align 8 dereferenceable(9) %0) unnamed_addr #4 comdat {
  ret void
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN9NthToggleD0Ev(ptr noundef nonnull align 8 dereferenceable(20) %0) unnamed_addr #5 comdat {
  tail call void @_ZdlPvm(ptr noundef nonnull %0, i64 noundef 24) #11
  ret void
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(9) ptr @_ZN9NthToggle8activateEv(ptr noundef nonnull align 8 dereferenceable(20) %0) unnamed_addr #4 comdat {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %3 = load i32, ptr %2, align 8, !tbaa !46
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr %2, align 8, !tbaa !46
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 12
  %6 = load i32, ptr %5, align 4, !tbaa !48
  %7 = icmp slt i32 %4, %6
  br i1 %7, label %14, label %8

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %10 = load i8, ptr %9, align 8, !tbaa !13, !range !16, !noundef !17
  %11 = trunc nuw i8 %10 to i1
  %12 = xor i1 %11, true
  %13 = zext i1 %12 to i8
  store i8 %13, ptr %9, align 8, !tbaa !13
  store i32 0, ptr %2, align 8, !tbaa !46
  br label %14

14:                                               ; preds = %8, %1
  ret ptr %0
}

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #6

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #6

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #6

; Function Attrs: cold noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #7

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570)) local_unnamed_addr #6

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { inlinehint mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nounwind }
attributes #9 = { builtin allocsize(0) }
attributes #10 = { cold noreturn }
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
!12 = !{!"vtable pointer", !10, i64 0}
!13 = !{!14, !15, i64 8}
!14 = !{!"_ZTS6Toggle", !15, i64 8}
!15 = !{!"bool", !9, i64 0}
!16 = !{i8 0, i8 2}
!17 = !{}
!18 = !{!19, !32, i64 240}
!19 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !20, i64 0, !30, i64 216, !9, i64 224, !15, i64 225, !31, i64 232, !32, i64 240, !33, i64 248, !34, i64 256}
!20 = !{!"_ZTSSt8ios_base", !21, i64 8, !21, i64 16, !22, i64 24, !23, i64 28, !23, i64 32, !24, i64 40, !25, i64 48, !9, i64 64, !26, i64 192, !27, i64 200, !28, i64 208}
!21 = !{!"long", !9, i64 0}
!22 = !{!"_ZTSSt13_Ios_Fmtflags", !9, i64 0}
!23 = !{!"_ZTSSt12_Ios_Iostate", !9, i64 0}
!24 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !8, i64 0}
!25 = !{!"_ZTSNSt8ios_base6_WordsE", !8, i64 0, !21, i64 8}
!26 = !{!"int", !9, i64 0}
!27 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !8, i64 0}
!28 = !{!"_ZTSSt6locale", !29, i64 0}
!29 = !{!"p1 _ZTSNSt6locale5_ImplE", !8, i64 0}
!30 = !{!"p1 _ZTSSo", !8, i64 0}
!31 = !{!"p1 _ZTSSt15basic_streambufIcSt11char_traitsIcEE", !8, i64 0}
!32 = !{!"p1 _ZTSSt5ctypeIcE", !8, i64 0}
!33 = !{!"p1 _ZTSSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!34 = !{!"p1 _ZTSSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!35 = !{!36, !9, i64 56}
!36 = !{!"_ZTSSt5ctypeIcE", !37, i64 0, !38, i64 16, !15, i64 24, !39, i64 32, !39, i64 40, !40, i64 48, !9, i64 56, !9, i64 57, !9, i64 313, !9, i64 569}
!37 = !{!"_ZTSNSt6locale5facetE", !26, i64 8}
!38 = !{!"p1 _ZTS15__locale_struct", !8, i64 0}
!39 = !{!"p1 int", !8, i64 0}
!40 = !{!"p1 short", !8, i64 0}
!41 = !{!9, !9, i64 0}
!42 = !{!26, !26, i64 0}
!43 = distinct !{!43, !44}
!44 = !{!"llvm.loop.mustprogress"}
!45 = distinct !{!45, !44}
!46 = !{!47, !26, i64 16}
!47 = !{!"_ZTS9NthToggle", !14, i64 0, !26, i64 12, !26, i64 16}
!48 = !{!47, !26, i64 12}
