; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/EH/cleanup-2.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/EH/cleanup-2.cpp"
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

$_ZN1CD2Ev = comdat any

$_ZN1BD2Ev = comdat any

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

; Function Attrs: alwaysinline cold mustprogress noreturn uwtable
define dso_local void @_Z10throw_charv() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %1 = alloca %struct.C, align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #10
  %2 = tail call ptr @__cxa_allocate_exception(i64 1) #10
  store i8 99, ptr %2, align 16, !tbaa !6
  invoke void @__cxa_throw(ptr nonnull %2, ptr nonnull @_ZTIc, ptr null) #11
          to label %5 unwind label %3

3:                                                ; preds = %0
  %4 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN1CD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %1) #10
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #10
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
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #10
  %5 = tail call ptr @__cxa_allocate_exception(i64 4) #10
  store i32 42, ptr %5, align 16, !tbaa !9
  invoke void @__cxa_throw(ptr nonnull %5, ptr nonnull @_ZTIi, ptr null) #11
          to label %6 unwind label %7

6:                                                ; preds = %4
  unreachable

7:                                                ; preds = %4
  %8 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  %9 = extractvalue { ptr, i32 } %8, 0
  %10 = extractvalue { ptr, i32 } %8, 1
  call void @_ZN1BD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %2) #10
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #10
  %11 = call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi) #10
  %12 = icmp eq i32 %10, %11
  br i1 %12, label %13, label %25

13:                                               ; preds = %7
  %14 = call ptr @__cxa_begin_catch(ptr %9) #10
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
  call void @__cxa_end_catch() #10
  ret void

22:                                               ; preds = %19, %13, %1, %17
  %23 = landingpad { ptr, i32 }
          catch ptr null
  %24 = extractvalue { ptr, i32 } %23, 0
  call void @__clang_call_terminate(ptr %24) #12
  unreachable

25:                                               ; preds = %7
  call void @__clang_call_terminate(ptr %9) #12
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: alwaysinline cold mustprogress uwtable
define dso_local void @_Z7cleanupv() local_unnamed_addr #4 personality ptr @__gxx_personality_v0 {
  %1 = alloca i8, align 4
  %2 = alloca %struct.C, align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #10
  %3 = tail call ptr @__cxa_allocate_exception(i64 1) #10
  store i8 99, ptr %3, align 16, !tbaa !6
  invoke void @__cxa_throw(ptr nonnull %3, ptr nonnull @_ZTIc, ptr null) #11
          to label %7 unwind label %4

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIc
  call void @_ZN1CD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %2) #10
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #10
  %6 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.4, i64 noundef 13)
          to label %11 unwind label %8

7:                                                ; preds = %0
  unreachable

8:                                                ; preds = %4
  %9 = landingpad { ptr, i32 }
          catch ptr null
  %10 = extractvalue { ptr, i32 } %9, 0
  call void @__clang_call_terminate(ptr %10) #12
  unreachable

11:                                               ; preds = %4
  %12 = extractvalue { ptr, i32 } %5, 1
  %13 = call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIc) #10
  %14 = icmp eq i32 %12, %13
  br i1 %14, label %15, label %43

15:                                               ; preds = %11
  %16 = extractvalue { ptr, i32 } %5, 0
  %17 = call ptr @__cxa_begin_catch(ptr %16) #10
  %18 = load i8, ptr %17, align 1, !tbaa !6
  %19 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str, i64 noundef 13)
          to label %20 unwind label %41

20:                                               ; preds = %15
  call void @llvm.lifetime.start.p0(ptr nonnull %1)
  store i8 %18, ptr %1, align 4, !tbaa !6
  %21 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %22 = getelementptr i8, ptr %21, i64 -24
  %23 = load i64, ptr %22, align 8
  %24 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %23
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %26 = load i64, ptr %25, align 8, !tbaa !13
  %27 = icmp eq i64 %26, 0
  br i1 %27, label %30, label %28

28:                                               ; preds = %20
  %29 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %1, i64 noundef 1)
          to label %32 unwind label %41

30:                                               ; preds = %20
  %31 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %18)
          to label %32 unwind label %41

32:                                               ; preds = %28, %30
  %33 = phi ptr [ %29, %28 ], [ @_ZSt4cout, %30 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %1)
  %34 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %33, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %35 unwind label %41

35:                                               ; preds = %32
  call void @__cxa_end_catch() #10
  %36 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.5, i64 noundef 13)
          to label %40 unwind label %37

37:                                               ; preds = %35
  %38 = landingpad { ptr, i32 }
          catch ptr null
  %39 = extractvalue { ptr, i32 } %38, 0
  call void @__clang_call_terminate(ptr %39) #12
  unreachable

40:                                               ; preds = %35
  ret void

41:                                               ; preds = %32, %30, %28, %15
  %42 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch() #10
  br label %43

43:                                               ; preds = %41, %11
  %44 = phi { ptr, i32 } [ %42, %41 ], [ %5, %11 ]
  %45 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.5, i64 noundef 13)
          to label %49 unwind label %46

46:                                               ; preds = %43
  %47 = landingpad { ptr, i32 }
          catch ptr null
  %48 = extractvalue { ptr, i32 } %47, 0
  call void @__clang_call_terminate(ptr %48) #12
  unreachable

49:                                               ; preds = %43
  resume { ptr, i32 } %44
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
  tail call void @__clang_call_terminate(ptr %6) #12
  unreachable
}

; Function Attrs: nofree nosync nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #5

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: cold mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #6 personality ptr @__gxx_personality_v0 {
  %1 = alloca i8, align 4
  %2 = alloca %struct.C, align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #10
  %3 = tail call ptr @__cxa_allocate_exception(i64 1) #10
  store i8 99, ptr %3, align 16, !tbaa !6
  invoke void @__cxa_throw(ptr nonnull %3, ptr nonnull @_ZTIc, ptr null) #11
          to label %14 unwind label %4

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIc
  call void @_ZN1CD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %2) #10
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #10
  %6 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.4, i64 noundef 13)
          to label %10 unwind label %7

7:                                                ; preds = %4
  %8 = landingpad { ptr, i32 }
          catch ptr null
  %9 = extractvalue { ptr, i32 } %8, 0
  call void @__clang_call_terminate(ptr %9) #12
  unreachable

10:                                               ; preds = %4
  %11 = extractvalue { ptr, i32 } %5, 1
  %12 = call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIc) #10
  %13 = icmp eq i32 %11, %12
  br i1 %13, label %15, label %37

14:                                               ; preds = %0
  unreachable

15:                                               ; preds = %10
  %16 = extractvalue { ptr, i32 } %5, 0
  %17 = call ptr @__cxa_begin_catch(ptr %16) #10
  %18 = load i8, ptr %17, align 1, !tbaa !6
  %19 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str, i64 noundef 13)
          to label %20 unwind label %35

20:                                               ; preds = %15
  call void @llvm.lifetime.start.p0(ptr nonnull %1)
  store i8 %18, ptr %1, align 4, !tbaa !6
  %21 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %22 = getelementptr i8, ptr %21, i64 -24
  %23 = load i64, ptr %22, align 8
  %24 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %23
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %26 = load i64, ptr %25, align 8, !tbaa !13
  %27 = icmp eq i64 %26, 0
  br i1 %27, label %30, label %28

28:                                               ; preds = %20
  %29 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %1, i64 noundef 1)
          to label %32 unwind label %35

30:                                               ; preds = %20
  %31 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %18)
          to label %32 unwind label %35

32:                                               ; preds = %28, %30
  %33 = phi ptr [ %29, %28 ], [ @_ZSt4cout, %30 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %1)
  %34 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %33, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %43 unwind label %35

35:                                               ; preds = %32, %30, %28, %15
  %36 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch() #10
  br label %37

37:                                               ; preds = %35, %10
  %38 = phi { ptr, i32 } [ %36, %35 ], [ %5, %10 ]
  %39 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.5, i64 noundef 13)
          to label %54 unwind label %40

40:                                               ; preds = %37
  %41 = landingpad { ptr, i32 }
          catch ptr null
  %42 = extractvalue { ptr, i32 } %41, 0
  call void @__clang_call_terminate(ptr %42) #12
  unreachable

43:                                               ; preds = %32
  call void @__cxa_end_catch() #10
  %44 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.5, i64 noundef 13)
          to label %48 unwind label %45

45:                                               ; preds = %43
  %46 = landingpad { ptr, i32 }
          catch ptr null
  %47 = extractvalue { ptr, i32 } %46, 0
  call void @__clang_call_terminate(ptr %47) #12
  unreachable

48:                                               ; preds = %43
  %49 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.5, i64 noundef 13)
          to label %53 unwind label %50

50:                                               ; preds = %48
  %51 = landingpad { ptr, i32 }
          catch ptr null
  %52 = extractvalue { ptr, i32 } %51, 0
  call void @__clang_call_terminate(ptr %52) #12
  unreachable

53:                                               ; preds = %48
  ret i32 0

54:                                               ; preds = %37
  %55 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.5, i64 noundef 13)
          to label %59 unwind label %56

56:                                               ; preds = %54
  %57 = landingpad { ptr, i32 }
          catch ptr null
  %58 = extractvalue { ptr, i32 } %57, 0
  call void @__clang_call_terminate(ptr %58) #12
  unreachable

59:                                               ; preds = %54
  resume { ptr, i32 } %38
}

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) local_unnamed_addr #7 comdat {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #10
  tail call void @_ZSt9terminatev() #12
  unreachable
}

; Function Attrs: cold nofree noreturn
declare void @_ZSt9terminatev() local_unnamed_addr #8

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8), i32 noundef) local_unnamed_addr #9

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #9

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #9

attributes #0 = { alwaysinline cold mustprogress noreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { cold noreturn }
attributes #3 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { alwaysinline cold mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree nosync nounwind memory(none) }
attributes #6 = { cold mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { cold nofree noreturn }
attributes #9 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { nounwind }
attributes #11 = { noreturn }
attributes #12 = { noreturn nounwind }

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
!11 = !{!12, !12, i64 0}
!12 = !{!"vtable pointer", !8, i64 0}
!13 = !{!14, !15, i64 16}
!14 = !{!"_ZTSSt8ios_base", !15, i64 8, !15, i64 16, !16, i64 24, !17, i64 28, !17, i64 32, !18, i64 40, !20, i64 48, !7, i64 64, !10, i64 192, !21, i64 200, !22, i64 208}
!15 = !{!"long", !7, i64 0}
!16 = !{!"_ZTSSt13_Ios_Fmtflags", !7, i64 0}
!17 = !{!"_ZTSSt12_Ios_Iostate", !7, i64 0}
!18 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !19, i64 0}
!19 = !{!"any pointer", !7, i64 0}
!20 = !{!"_ZTSNSt8ios_base6_WordsE", !19, i64 0, !15, i64 8}
!21 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !19, i64 0}
!22 = !{!"_ZTSSt6locale", !23, i64 0}
!23 = !{!"p1 _ZTSNSt6locale5_ImplE", !19, i64 0}
