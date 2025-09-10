; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/EH/cleanup-3.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/EH/cleanup-3.cpp"
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
@.str.3 = private unnamed_addr constant [14 x i8] c"In C's d'tor\0A\00", align 1
@.str.4 = private unnamed_addr constant [22 x i8] c"(C::~C) Caught char: \00", align 1
@_ZTIi = external constant ptr
@.str.5 = private unnamed_addr constant [14 x i8] c"In B's d'tor\0A\00", align 1
@.str.6 = private unnamed_addr constant [14 x i8] c"In A's d'tor\0A\00", align 1
@str = private unnamed_addr constant [22 x i8] c"Inside the terminator\00", align 4

; Function Attrs: alwaysinline cold mustprogress noreturn uwtable
define dso_local void @_Z10throw_charv() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %1 = alloca %struct.C, align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #15
  %2 = tail call ptr @__cxa_allocate_exception(i64 1) #15
  store i8 99, ptr %2, align 16, !tbaa !6
  invoke void @__cxa_throw(ptr nonnull %2, ptr nonnull @_ZTIc, ptr null) #16
          to label %5 unwind label %3

3:                                                ; preds = %0
  %4 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN1CD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %1) #15
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #15
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
  %3 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.3, i64 noundef 13)
          to label %4 unwind label %22

4:                                                ; preds = %1
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #15
  %5 = tail call ptr @__cxa_allocate_exception(i64 4) #15
  store i32 42, ptr %5, align 16, !tbaa !9
  invoke void @__cxa_throw(ptr nonnull %5, ptr nonnull @_ZTIi, ptr null) #16
          to label %6 unwind label %7

6:                                                ; preds = %4
  unreachable

7:                                                ; preds = %4
  %8 = landingpad { ptr, i32 }
          catch ptr @_ZTIc
          catch ptr null
  %9 = extractvalue { ptr, i32 } %8, 0
  %10 = extractvalue { ptr, i32 } %8, 1
  call void @_ZN1BD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %2) #15
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #15
  %11 = call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIc) #15
  %12 = icmp eq i32 %10, %11
  br i1 %12, label %13, label %25

13:                                               ; preds = %7
  %14 = call ptr @__cxa_begin_catch(ptr %9) #15
  %15 = load i8, ptr %14, align 1, !tbaa !6
  %16 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.4, i64 noundef 21)
          to label %17 unwind label %22

17:                                               ; preds = %13
  %18 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %15)
          to label %19 unwind label %22

19:                                               ; preds = %17
  %20 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %18, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %21 unwind label %22

21:                                               ; preds = %19
  call void @__cxa_end_catch() #15
  ret void

22:                                               ; preds = %19, %13, %1, %17
  %23 = landingpad { ptr, i32 }
          catch ptr null
  %24 = extractvalue { ptr, i32 } %23, 0
  call void @__clang_call_terminate(ptr %24) #17
  unreachable

25:                                               ; preds = %7
  call void @__clang_call_terminate(ptr %9) #17
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: alwaysinline cold mustprogress uwtable
define dso_local void @_Z7cleanupv() local_unnamed_addr #4 personality ptr @__gxx_personality_v0 {
  %1 = alloca i8, align 4
  %2 = alloca %struct.C, align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #15
  %3 = tail call ptr @__cxa_allocate_exception(i64 1) #15
  store i8 99, ptr %3, align 16, !tbaa !6
  invoke void @__cxa_throw(ptr nonnull %3, ptr nonnull @_ZTIc, ptr null) #16
          to label %7 unwind label %4

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIc
  call void @_ZN1CD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %2) #15
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #15
  %6 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.5, i64 noundef 13)
          to label %11 unwind label %8

7:                                                ; preds = %0
  unreachable

8:                                                ; preds = %4
  %9 = landingpad { ptr, i32 }
          catch ptr null
  %10 = extractvalue { ptr, i32 } %9, 0
  call void @__clang_call_terminate(ptr %10) #17
  unreachable

11:                                               ; preds = %4
  %12 = extractvalue { ptr, i32 } %5, 1
  %13 = call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIc) #15
  %14 = icmp eq i32 %12, %13
  br i1 %14, label %15, label %43

15:                                               ; preds = %11
  %16 = extractvalue { ptr, i32 } %5, 0
  %17 = call ptr @__cxa_begin_catch(ptr %16) #15
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
  call void @__cxa_end_catch() #15
  %36 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.6, i64 noundef 13)
          to label %40 unwind label %37

37:                                               ; preds = %35
  %38 = landingpad { ptr, i32 }
          catch ptr null
  %39 = extractvalue { ptr, i32 } %38, 0
  call void @__clang_call_terminate(ptr %39) #17
  unreachable

40:                                               ; preds = %35
  ret void

41:                                               ; preds = %32, %30, %28, %15
  %42 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch() #15
  br label %43

43:                                               ; preds = %41, %11
  %44 = phi { ptr, i32 } [ %42, %41 ], [ %5, %11 ]
  %45 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.6, i64 noundef 13)
          to label %49 unwind label %46

46:                                               ; preds = %43
  %47 = landingpad { ptr, i32 }
          catch ptr null
  %48 = extractvalue { ptr, i32 } %47, 0
  call void @__clang_call_terminate(ptr %48) #17
  unreachable

49:                                               ; preds = %43
  resume { ptr, i32 } %44
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN1BD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #3 comdat personality ptr @__gxx_personality_v0 {
  %2 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.5, i64 noundef 13)
          to label %3 unwind label %4

3:                                                ; preds = %1
  ret void

4:                                                ; preds = %1
  %5 = landingpad { ptr, i32 }
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 0
  tail call void @__clang_call_terminate(ptr %6) #17
  unreachable
}

; Function Attrs: nofree nosync nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #5

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

; Function Attrs: inlinehint mustprogress uwtable
declare noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #6

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: mustprogress nofree noreturn nounwind uwtable
define dso_local void @_Z4termv() #7 {
  %1 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @exit(i32 noundef 0) #17
  unreachable
}

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #8

; Function Attrs: cold mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #9 personality ptr @__gxx_personality_v0 {
  %1 = alloca i8, align 4
  %2 = alloca %struct.C, align 1
  %3 = tail call noundef ptr @_ZSt13set_terminatePFvvE(ptr noundef nonnull @_Z4termv) #15
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #15
  %4 = tail call ptr @__cxa_allocate_exception(i64 1) #15
  store i8 99, ptr %4, align 16, !tbaa !6
  invoke void @__cxa_throw(ptr nonnull %4, ptr nonnull @_ZTIc, ptr null) #16
          to label %15 unwind label %5

5:                                                ; preds = %0
  %6 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIc
  call void @_ZN1CD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %2) #15
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #15
  %7 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.5, i64 noundef 13)
          to label %11 unwind label %8

8:                                                ; preds = %5
  %9 = landingpad { ptr, i32 }
          catch ptr null
  %10 = extractvalue { ptr, i32 } %9, 0
  call void @__clang_call_terminate(ptr %10) #17
  unreachable

11:                                               ; preds = %5
  %12 = extractvalue { ptr, i32 } %6, 1
  %13 = call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIc) #15
  %14 = icmp eq i32 %12, %13
  br i1 %14, label %16, label %38

15:                                               ; preds = %0
  unreachable

16:                                               ; preds = %11
  %17 = extractvalue { ptr, i32 } %6, 0
  %18 = call ptr @__cxa_begin_catch(ptr %17) #15
  %19 = load i8, ptr %18, align 1, !tbaa !6
  %20 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str, i64 noundef 13)
          to label %21 unwind label %36

21:                                               ; preds = %16
  call void @llvm.lifetime.start.p0(ptr nonnull %1)
  store i8 %19, ptr %1, align 4, !tbaa !6
  %22 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !11
  %23 = getelementptr i8, ptr %22, i64 -24
  %24 = load i64, ptr %23, align 8
  %25 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %24
  %26 = getelementptr inbounds nuw i8, ptr %25, i64 16
  %27 = load i64, ptr %26, align 8, !tbaa !13
  %28 = icmp eq i64 %27, 0
  br i1 %28, label %31, label %29

29:                                               ; preds = %21
  %30 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %1, i64 noundef 1)
          to label %33 unwind label %36

31:                                               ; preds = %21
  %32 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %19)
          to label %33 unwind label %36

33:                                               ; preds = %29, %31
  %34 = phi ptr [ %30, %29 ], [ @_ZSt4cout, %31 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %1)
  %35 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %34, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %44 unwind label %36

36:                                               ; preds = %33, %31, %29, %16
  %37 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch() #15
  br label %38

38:                                               ; preds = %36, %11
  %39 = phi { ptr, i32 } [ %37, %36 ], [ %6, %11 ]
  %40 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.6, i64 noundef 13)
          to label %55 unwind label %41

41:                                               ; preds = %38
  %42 = landingpad { ptr, i32 }
          catch ptr null
  %43 = extractvalue { ptr, i32 } %42, 0
  call void @__clang_call_terminate(ptr %43) #17
  unreachable

44:                                               ; preds = %33
  call void @__cxa_end_catch() #15
  %45 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.6, i64 noundef 13)
          to label %49 unwind label %46

46:                                               ; preds = %44
  %47 = landingpad { ptr, i32 }
          catch ptr null
  %48 = extractvalue { ptr, i32 } %47, 0
  call void @__clang_call_terminate(ptr %48) #17
  unreachable

49:                                               ; preds = %44
  %50 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.6, i64 noundef 13)
          to label %54 unwind label %51

51:                                               ; preds = %49
  %52 = landingpad { ptr, i32 }
          catch ptr null
  %53 = extractvalue { ptr, i32 } %52, 0
  call void @__clang_call_terminate(ptr %53) #17
  unreachable

54:                                               ; preds = %49
  ret i32 0

55:                                               ; preds = %38
  %56 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.6, i64 noundef 13)
          to label %60 unwind label %57

57:                                               ; preds = %55
  %58 = landingpad { ptr, i32 }
          catch ptr null
  %59 = extractvalue { ptr, i32 } %58, 0
  call void @__clang_call_terminate(ptr %59) #17
  unreachable

60:                                               ; preds = %55
  resume { ptr, i32 } %39
}

; Function Attrs: nounwind
declare noundef ptr @_ZSt13set_terminatePFvvE(ptr noundef) local_unnamed_addr #10

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) local_unnamed_addr #11 comdat {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #15
  tail call void @_ZSt9terminatev() #17
  unreachable
}

; Function Attrs: cold nofree noreturn
declare void @_ZSt9terminatev() local_unnamed_addr #12

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #13

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #13

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #14

attributes #0 = { alwaysinline cold mustprogress noreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { cold noreturn }
attributes #3 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { alwaysinline cold mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree nosync nounwind memory(none) }
attributes #6 = { inlinehint mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { mustprogress nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { cold mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #12 = { cold nofree noreturn }
attributes #13 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #14 = { nofree nounwind }
attributes #15 = { nounwind }
attributes #16 = { noreturn }
attributes #17 = { noreturn nounwind }

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
