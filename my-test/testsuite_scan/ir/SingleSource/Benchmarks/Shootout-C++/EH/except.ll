; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/EH/except.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/EH/except.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }

$__clang_call_terminate = comdat any

$_ZTI12Lo_exception = comdat any

$_ZTS12Lo_exception = comdat any

$_ZTI12Hi_exception = comdat any

$_ZTS12Hi_exception = comdat any

@HI = dso_local local_unnamed_addr global i64 0, align 8
@LO = dso_local local_unnamed_addr global i64 0, align 8
@_ZTI12Lo_exception = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS12Lo_exception }, comdat, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@_ZTS12Lo_exception = linkonce_odr dso_local constant [15 x i8] c"12Lo_exception\00", comdat, align 1
@_ZTI12Hi_exception = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS12Hi_exception }, comdat, align 8
@_ZTS12Hi_exception = linkonce_odr dso_local constant [15 x i8] c"12Hi_exception\00", comdat, align 1
@_ZSt4cerr = external global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [23 x i8] c"We shouldn't get here\0A\00", align 1
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str.1 = private unnamed_addr constant [16 x i8] c"Exceptions: HI=\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c" / \00", align 1
@.str.3 = private unnamed_addr constant [4 x i8] c"LO=\00", align 1

; Function Attrs: cold mustprogress noreturn uwtable
define dso_local void @_Z6blowupm(i64 noundef %0) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %2 = and i64 %0, 1
  %3 = icmp eq i64 %2, 0
  %4 = tail call ptr @__cxa_allocate_exception(i64 16) #12
  store i64 %0, ptr %4, align 8, !tbaa !6
  br i1 %3, label %6, label %5

5:                                                ; preds = %1
  tail call void @__cxa_throw(ptr nonnull %4, ptr nonnull @_ZTI12Lo_exception, ptr null) #13
  unreachable

6:                                                ; preds = %1
  tail call void @__cxa_throw(ptr nonnull %4, ptr nonnull @_ZTI12Hi_exception, ptr null) #13
  unreachable
}

declare ptr @__cxa_allocate_exception(i64) local_unnamed_addr

declare i32 @__gxx_personality_v0(...)

; Function Attrs: cold noreturn
declare void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr #1

; Function Attrs: cold mustprogress uwtable
define dso_local void @_Z11lo_functionm(i64 noundef %0) local_unnamed_addr #2 personality ptr @__gxx_personality_v0 {
  invoke void @_Z6blowupm(i64 noundef %0)
          to label %12 unwind label %2

2:                                                ; preds = %1
  %3 = landingpad { ptr, i32 }
          catch ptr @_ZTI12Lo_exception
  %4 = extractvalue { ptr, i32 } %3, 1
  %5 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI12Lo_exception) #12
  %6 = icmp eq i32 %4, %5
  br i1 %6, label %7, label %13

7:                                                ; preds = %2
  %8 = extractvalue { ptr, i32 } %3, 0
  %9 = tail call ptr @__cxa_begin_catch(ptr %8) #12
  %10 = load i64, ptr @LO, align 8, !tbaa !6
  %11 = add i64 %10, 1
  store i64 %11, ptr @LO, align 8, !tbaa !6
  tail call void @__cxa_end_catch()
  ret void

12:                                               ; preds = %1
  unreachable

13:                                               ; preds = %2
  resume { ptr, i32 } %3
}

; Function Attrs: nofree nosync nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #3

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: cold mustprogress uwtable
define dso_local void @_Z11hi_functionm(i64 noundef %0) local_unnamed_addr #2 personality ptr @__gxx_personality_v0 {
  invoke void @_Z11lo_functionm(i64 noundef %0)
          to label %12 unwind label %2

2:                                                ; preds = %1
  %3 = landingpad { ptr, i32 }
          catch ptr @_ZTI12Hi_exception
  %4 = extractvalue { ptr, i32 } %3, 1
  %5 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI12Hi_exception) #12
  %6 = icmp eq i32 %4, %5
  br i1 %6, label %7, label %13

7:                                                ; preds = %2
  %8 = extractvalue { ptr, i32 } %3, 0
  %9 = tail call ptr @__cxa_begin_catch(ptr %8) #12
  %10 = load i64, ptr @HI, align 8, !tbaa !6
  %11 = add i64 %10, 1
  store i64 %11, ptr @HI, align 8, !tbaa !6
  tail call void @__cxa_end_catch()
  br label %12

12:                                               ; preds = %1, %7
  ret void

13:                                               ; preds = %2
  resume { ptr, i32 } %3
}

; Function Attrs: cold mustprogress uwtable
define dso_local void @_Z13some_functionm(i64 noundef %0) local_unnamed_addr #2 personality ptr @__gxx_personality_v0 {
  invoke void @_Z11hi_functionm(i64 noundef %0)
          to label %10 unwind label %2

2:                                                ; preds = %1
  %3 = landingpad { ptr, i32 }
          catch ptr null
  %4 = extractvalue { ptr, i32 } %3, 0
  %5 = tail call ptr @__cxa_begin_catch(ptr %4) #12
  %6 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str)
          to label %7 unwind label %8

7:                                                ; preds = %2
  tail call void @exit(i32 noundef 1) #14
  unreachable

8:                                                ; preds = %2
  %9 = landingpad { ptr, i32 }
          cleanup
  invoke void @__cxa_end_catch()
          to label %11 unwind label %12

10:                                               ; preds = %1
  ret void

11:                                               ; preds = %8
  resume { ptr, i32 } %9

12:                                               ; preds = %8
  %13 = landingpad { ptr, i32 }
          catch ptr null
  %14 = extractvalue { ptr, i32 } %13, 0
  tail call void @__clang_call_terminate(ptr %14) #15
  unreachable
}

; Function Attrs: inlinehint mustprogress uwtable
declare noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef) local_unnamed_addr #4

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #5

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) local_unnamed_addr #6 comdat {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #12
  tail call void @_ZSt9terminatev() #15
  unreachable
}

; Function Attrs: cold nofree noreturn
declare void @_ZSt9terminatev() local_unnamed_addr #7

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #8 {
  %3 = icmp eq i32 %0, 2
  br i1 %3, label %4, label %13

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !10
  %7 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %6, ptr noundef null, i32 noundef 10) #12
  %8 = trunc i64 %7 to i32
  %9 = icmp slt i32 %8, 1
  br i1 %9, label %13, label %10

10:                                               ; preds = %4
  %11 = and i64 %7, 2147483647
  %12 = icmp eq i64 %11, 0
  br i1 %12, label %19, label %13

13:                                               ; preds = %4, %2, %10
  %14 = phi i64 [ 1, %4 ], [ 100000, %2 ], [ %11, %10 ]
  br label %15

15:                                               ; preds = %13, %15
  %16 = phi i64 [ %17, %15 ], [ %14, %13 ]
  %17 = add nsw i64 %16, -1
  tail call void @_Z13some_functionm(i64 noundef %17)
  %18 = icmp eq i64 %17, 0
  br i1 %18, label %19, label %15, !llvm.loop !13

19:                                               ; preds = %15, %10
  %20 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.1, i64 noundef 15)
  %21 = load i64, ptr @HI, align 8, !tbaa !6
  %22 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i64 noundef %21)
  %23 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %22, ptr noundef nonnull @.str.2, i64 noundef 3)
  %24 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %22, ptr noundef nonnull @.str.3, i64 noundef 3)
  %25 = load i64, ptr @LO, align 8, !tbaa !6
  %26 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) %22, i64 noundef %25)
  %27 = load ptr, ptr %26, align 8, !tbaa !15
  %28 = getelementptr i8, ptr %27, i64 -24
  %29 = load i64, ptr %28, align 8
  %30 = getelementptr inbounds i8, ptr %26, i64 %29
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 240
  %32 = load ptr, ptr %31, align 8, !tbaa !17
  %33 = icmp eq ptr %32, null
  br i1 %33, label %34, label %35

34:                                               ; preds = %19
  tail call void @_ZSt16__throw_bad_castv() #1
  unreachable

35:                                               ; preds = %19
  %36 = getelementptr inbounds nuw i8, ptr %32, i64 56
  %37 = load i8, ptr %36, align 8, !tbaa !34
  %38 = icmp eq i8 %37, 0
  br i1 %38, label %42, label %39

39:                                               ; preds = %35
  %40 = getelementptr inbounds nuw i8, ptr %32, i64 67
  %41 = load i8, ptr %40, align 1, !tbaa !40
  br label %47

42:                                               ; preds = %35
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %32)
  %43 = load ptr, ptr %32, align 8, !tbaa !15
  %44 = getelementptr inbounds nuw i8, ptr %43, i64 48
  %45 = load ptr, ptr %44, align 8
  %46 = tail call noundef i8 %45(ptr noundef nonnull align 8 dereferenceable(570) %32, i8 noundef 10)
  br label %47

47:                                               ; preds = %39, %42
  %48 = phi i8 [ %41, %39 ], [ %46, %42 ]
  %49 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) %26, i8 noundef %48)
  %50 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %49)
  ret i32 0
}

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #9

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #10

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8), i64 noundef) local_unnamed_addr #10

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #10

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #10

; Function Attrs: cold noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #11

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570)) local_unnamed_addr #10

attributes #0 = { cold mustprogress noreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold noreturn }
attributes #2 = { cold mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nosync nounwind memory(none) }
attributes #4 = { inlinehint mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { cold nofree noreturn }
attributes #8 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #12 = { nounwind }
attributes #13 = { noreturn }
attributes #14 = { cold noreturn nounwind }
attributes #15 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"p1 omnipotent char", !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
!13 = distinct !{!13, !14}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!16, !16, i64 0}
!16 = !{!"vtable pointer", !9, i64 0}
!17 = !{!18, !31, i64 240}
!18 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !19, i64 0, !28, i64 216, !8, i64 224, !29, i64 225, !30, i64 232, !31, i64 240, !32, i64 248, !33, i64 256}
!19 = !{!"_ZTSSt8ios_base", !7, i64 8, !7, i64 16, !20, i64 24, !21, i64 28, !21, i64 32, !22, i64 40, !23, i64 48, !8, i64 64, !24, i64 192, !25, i64 200, !26, i64 208}
!20 = !{!"_ZTSSt13_Ios_Fmtflags", !8, i64 0}
!21 = !{!"_ZTSSt12_Ios_Iostate", !8, i64 0}
!22 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !12, i64 0}
!23 = !{!"_ZTSNSt8ios_base6_WordsE", !12, i64 0, !7, i64 8}
!24 = !{!"int", !8, i64 0}
!25 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !12, i64 0}
!26 = !{!"_ZTSSt6locale", !27, i64 0}
!27 = !{!"p1 _ZTSNSt6locale5_ImplE", !12, i64 0}
!28 = !{!"p1 _ZTSSo", !12, i64 0}
!29 = !{!"bool", !8, i64 0}
!30 = !{!"p1 _ZTSSt15basic_streambufIcSt11char_traitsIcEE", !12, i64 0}
!31 = !{!"p1 _ZTSSt5ctypeIcE", !12, i64 0}
!32 = !{!"p1 _ZTSSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE", !12, i64 0}
!33 = !{!"p1 _ZTSSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE", !12, i64 0}
!34 = !{!35, !8, i64 56}
!35 = !{!"_ZTSSt5ctypeIcE", !36, i64 0, !37, i64 16, !29, i64 24, !38, i64 32, !38, i64 40, !39, i64 48, !8, i64 56, !8, i64 57, !8, i64 313, !8, i64 569}
!36 = !{!"_ZTSNSt6locale5facetE", !24, i64 8}
!37 = !{!"p1 _ZTS15__locale_struct", !12, i64 0}
!38 = !{!"p1 int", !12, i64 0}
!39 = !{!"p1 short", !12, i64 0}
!40 = !{!8, !8, i64 0}
