; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/EH/cleanup-1.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/EH/cleanup-1.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }
%struct.C = type { i8 }
%struct.B = type { i8 }
%struct.A = type { i8 }

$_ZN1CD2Ev = comdat any

$_ZN1BD2Ev = comdat any

$_ZN1AD2Ev = comdat any

$__clang_call_terminate = comdat any

@_ZTIc = external constant ptr
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [14 x i8] c"Caught char: \00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.2 = private unnamed_addr constant [14 x i8] c"In C's d'tor\0A\00", align 1
@_ZTIi = external constant ptr
@.str.3 = private unnamed_addr constant [21 x i8] c"(C::~C) Caught int: \00", align 1
@.str.4 = private unnamed_addr constant [14 x i8] c"In B's d'tor\0A\00", align 1
@.str.5 = private unnamed_addr constant [14 x i8] c"In A's d'tor\0A\00", align 1

; Function Attrs: cold mustprogress noinline noreturn uwtable
define dso_local void @_Z10throw_charv() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %1 = alloca %struct.C, align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #11
  %2 = tail call ptr @__cxa_allocate_exception(i64 1) #11
  store i8 99, ptr %2, align 16, !tbaa !6
  invoke void @__cxa_throw(ptr nonnull %2, ptr nonnull @_ZTIc, ptr null) #12
          to label %5 unwind label %3

3:                                                ; preds = %0
  %4 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN1CD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %1) #11
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #11
  resume { ptr, i32 } %4

5:                                                ; preds = %0
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

declare ptr @__cxa_allocate_exception(i64) local_unnamed_addr

; Function Attrs: cold noreturn
declare void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr #2

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN1CD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #3 comdat personality ptr @__gxx_personality_v0 {
  %2 = alloca %struct.B, align 1
  %3 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.2, i64 noundef 13)
          to label %4 unwind label %22

4:                                                ; preds = %1
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #11
  %5 = tail call ptr @__cxa_allocate_exception(i64 4) #11
  store i32 42, ptr %5, align 16, !tbaa !9
  invoke void @__cxa_throw(ptr nonnull %5, ptr nonnull @_ZTIi, ptr null) #12
          to label %6 unwind label %7

6:                                                ; preds = %4
  unreachable

7:                                                ; preds = %4
  %8 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  %9 = extractvalue { ptr, i32 } %8, 0
  %10 = extractvalue { ptr, i32 } %8, 1
  call void @_ZN1BD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %2) #11
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #11
  %11 = call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi) #11
  %12 = icmp eq i32 %10, %11
  br i1 %12, label %13, label %25

13:                                               ; preds = %7
  %14 = call ptr @__cxa_begin_catch(ptr %9) #11
  %15 = load i32, ptr %14, align 4, !tbaa !9
  %16 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.3, i64 noundef 20)
          to label %17 unwind label %22

17:                                               ; preds = %13
  %18 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i32 noundef %15)
          to label %19 unwind label %22

19:                                               ; preds = %17
  %20 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %18, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %21 unwind label %22

21:                                               ; preds = %19
  call void @__cxa_end_catch() #11
  ret void

22:                                               ; preds = %19, %13, %1, %17
  %23 = landingpad { ptr, i32 }
          catch ptr null
  %24 = extractvalue { ptr, i32 } %23, 0
  call void @__clang_call_terminate(ptr %24) #13
  unreachable

25:                                               ; preds = %7
  call void @__clang_call_terminate(ptr %9) #13
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: cold mustprogress noinline uwtable
define dso_local void @_Z7cleanupv() local_unnamed_addr #4 personality ptr @__gxx_personality_v0 {
  %1 = alloca %struct.A, align 1
  %2 = alloca %struct.B, align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #11
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #11
  invoke void @_Z10throw_charv()
          to label %3 unwind label %4

3:                                                ; preds = %0
  unreachable

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIc
  %6 = extractvalue { ptr, i32 } %5, 1
  call void @_ZN1BD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %2) #11
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #11
  %7 = call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIc) #11
  %8 = icmp eq i32 %6, %7
  br i1 %8, label %9, label %26

9:                                                ; preds = %4
  %10 = extractvalue { ptr, i32 } %5, 0
  %11 = call ptr @__cxa_begin_catch(ptr %10) #11
  %12 = load i8, ptr %11, align 1, !tbaa !6
  %13 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str, i64 noundef 13)
          to label %14 unwind label %24

14:                                               ; preds = %9
  %15 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %12)
          to label %16 unwind label %24

16:                                               ; preds = %14
  %17 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %15, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %18 unwind label %24

18:                                               ; preds = %16
  call void @__cxa_end_catch() #11
  %19 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.5, i64 noundef 13)
          to label %23 unwind label %20

20:                                               ; preds = %18
  %21 = landingpad { ptr, i32 }
          catch ptr null
  %22 = extractvalue { ptr, i32 } %21, 0
  call void @__clang_call_terminate(ptr %22) #13
  unreachable

23:                                               ; preds = %18
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #11
  ret void

24:                                               ; preds = %16, %9, %14
  %25 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch() #11
  br label %26

26:                                               ; preds = %24, %4
  %27 = phi { ptr, i32 } [ %25, %24 ], [ %5, %4 ]
  call void @_ZN1AD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %1) #11
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #11
  resume { ptr, i32 } %27
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN1BD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #3 comdat personality ptr @__gxx_personality_v0 {
  %2 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.4, i64 noundef 13)
          to label %3 unwind label %4

3:                                                ; preds = %1
  ret void

4:                                                ; preds = %1
  %5 = landingpad { ptr, i32 }
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 0
  tail call void @__clang_call_terminate(ptr %6) #13
  unreachable
}

; Function Attrs: nofree nosync nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #5

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

; Function Attrs: inlinehint mustprogress uwtable
declare noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #6

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN1AD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #3 comdat personality ptr @__gxx_personality_v0 {
  %2 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.5, i64 noundef 13)
          to label %3 unwind label %4

3:                                                ; preds = %1
  ret void

4:                                                ; preds = %1
  %5 = landingpad { ptr, i32 }
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 0
  tail call void @__clang_call_terminate(ptr %6) #13
  unreachable
}

; Function Attrs: cold mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #7 personality ptr @__gxx_personality_v0 {
  %1 = alloca %struct.A, align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #11
  invoke void @_Z7cleanupv()
          to label %2 unwind label %8

2:                                                ; preds = %0
  %3 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.5, i64 noundef 13)
          to label %7 unwind label %4

4:                                                ; preds = %2
  %5 = landingpad { ptr, i32 }
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 0
  tail call void @__clang_call_terminate(ptr %6) #13
  unreachable

7:                                                ; preds = %2
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #11
  ret i32 0

8:                                                ; preds = %0
  %9 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN1AD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %1) #11
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #11
  resume { ptr, i32 } %9
}

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) local_unnamed_addr #8 comdat {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #11
  tail call void @_ZSt9terminatev() #13
  unreachable
}

; Function Attrs: cold nofree noreturn
declare void @_ZSt9terminatev() local_unnamed_addr #9

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8), i32 noundef) local_unnamed_addr #10

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #10

attributes #0 = { cold mustprogress noinline noreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { cold noreturn }
attributes #3 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { cold mustprogress noinline uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree nosync nounwind memory(none) }
attributes #6 = { inlinehint mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { cold mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { cold nofree noreturn }
attributes #10 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { nounwind }
attributes #12 = { noreturn }
attributes #13 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !7, i64 0}
