; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/EH/simple-1.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/EH/simple-1.cpp"
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

$__clang_call_terminate = comdat any

$_ZTINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE = comdat any

$_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE = comdat any

$_ZTI1A = comdat any

$_ZTS1A = comdat any

@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [16 x i8] c"Throwing char: \00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@_ZTIc = external constant ptr
@.str.2 = private unnamed_addr constant [15 x i8] c"Throwing int: \00", align 1
@_ZTIi = external constant ptr
@.str.3 = private unnamed_addr constant [17 x i8] c"Throwing float: \00", align 1
@_ZTIf = external constant ptr
@.str.4 = private unnamed_addr constant [18 x i8] c"Throwing double: \00", align 1
@_ZTId = external constant ptr
@.str.5 = private unnamed_addr constant [12 x i8] c"hello world\00", align 1
@.str.6 = private unnamed_addr constant [23 x i8] c"Throwing std::string: \00", align 1
@_ZTINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE }, comdat, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE = linkonce_odr dso_local constant [53 x i8] c"NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE\00", comdat, align 1
@.str.7 = private unnamed_addr constant [12 x i8] c"Throwing A\0A\00", align 1
@_ZTI1A = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1A }, comdat, align 8
@_ZTS1A = linkonce_odr dso_local constant [3 x i8] c"1A\00", comdat, align 1
@.str.8 = private unnamed_addr constant [14 x i8] c"Caught char: \00", align 1
@.str.9 = private unnamed_addr constant [13 x i8] c"Caught int: \00", align 1
@.str.10 = private unnamed_addr constant [15 x i8] c"Caught float: \00", align 1
@.str.11 = private unnamed_addr constant [16 x i8] c"Caught double: \00", align 1
@.str.12 = private unnamed_addr constant [21 x i8] c"Caught std::string: \00", align 1
@.str.13 = private unnamed_addr constant [10 x i8] c"Caught A\0A\00", align 1

; Function Attrs: cold mustprogress noinline noreturn uwtable
define dso_local void @_Z10throw_charv() local_unnamed_addr #0 {
  %1 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str)
  %2 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(ptr noundef nonnull align 8 dereferenceable(8) %1, i8 noundef 97)
  %3 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull @.str.1)
  %4 = tail call ptr @__cxa_allocate_exception(i64 1) #15
  store i8 97, ptr %4, align 16, !tbaa !6
  tail call void @__cxa_throw(ptr nonnull %4, ptr nonnull @_ZTIc, ptr null) #16
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: inlinehint mustprogress uwtable
declare noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef) local_unnamed_addr #2

; Function Attrs: inlinehint mustprogress uwtable
declare noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #2

declare ptr @__cxa_allocate_exception(i64) local_unnamed_addr

; Function Attrs: cold noreturn
declare void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr #3

; Function Attrs: cold mustprogress noinline noreturn uwtable
define dso_local void @_Z9throw_intv() local_unnamed_addr #0 {
  %1 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.2)
  %2 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) %1, i32 noundef 37)
  %3 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull @.str.1)
  %4 = tail call ptr @__cxa_allocate_exception(i64 4) #15
  store i32 37, ptr %4, align 16, !tbaa !9
  tail call void @__cxa_throw(ptr nonnull %4, ptr nonnull @_ZTIi, ptr null) #16
  unreachable
}

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8), i32 noundef) local_unnamed_addr #4

; Function Attrs: cold mustprogress noinline noreturn uwtable
define dso_local void @_Z11throw_floatv() local_unnamed_addr #0 {
  %1 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.3)
  %2 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEf(ptr noundef nonnull align 8 dereferenceable(8) %1, float noundef 0x4042F6A7E0000000)
  %3 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull @.str.1)
  %4 = tail call ptr @__cxa_allocate_exception(i64 4) #15
  store float 0x4042F6A7E0000000, ptr %4, align 16, !tbaa !11
  tail call void @__cxa_throw(ptr nonnull %4, ptr nonnull @_ZTIf, ptr null) #16
  unreachable
}

; Function Attrs: mustprogress uwtable
declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEf(ptr noundef nonnull align 8 dereferenceable(8), float noundef) local_unnamed_addr #5

; Function Attrs: cold mustprogress noinline noreturn uwtable
define dso_local void @_Z12throw_doublev() local_unnamed_addr #0 {
  %1 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.4)
  %2 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEd(ptr noundef nonnull align 8 dereferenceable(8) %1, double noundef 3.792700e+01)
  %3 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull @.str.1)
  %4 = tail call ptr @__cxa_allocate_exception(i64 8) #15
  store double 3.792700e+01, ptr %4, align 16, !tbaa !13
  tail call void @__cxa_throw(ptr nonnull %4, ptr nonnull @_ZTId, ptr null) #16
  unreachable
}

; Function Attrs: mustprogress uwtable
declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEd(ptr noundef nonnull align 8 dereferenceable(8), double noundef) local_unnamed_addr #5

; Function Attrs: mustprogress noinline noreturn uwtable
define dso_local void @_Z12throw_stringv() local_unnamed_addr #6 personality ptr @__gxx_personality_v0 {
  %1 = alloca %"class.std::__cxx11::basic_string", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #15
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store ptr %2, ptr %1, align 8, !tbaa !15
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(11) %2, ptr noundef nonnull align 1 dereferenceable(11) @.str.5, i64 11, i1 false)
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i64 11, ptr %3, align 8, !tbaa !19
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 27
  store i8 0, ptr %4, align 1, !tbaa !6
  %5 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.6, i64 noundef 22)
          to label %6 unwind label %27

6:                                                ; preds = %0
  %7 = load ptr, ptr %1, align 8, !tbaa !22
  %8 = load i64, ptr %3, align 8, !tbaa !19
  %9 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef %7, i64 noundef %8)
          to label %10 unwind label %27

10:                                               ; preds = %6
  %11 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %9, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %12 unwind label %27

12:                                               ; preds = %10
  %13 = call ptr @__cxa_allocate_exception(i64 32) #15
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 16
  store ptr %14, ptr %13, align 8, !tbaa !15
  %15 = load ptr, ptr %1, align 8, !tbaa !22
  %16 = icmp eq ptr %15, %2
  br i1 %16, label %17, label %21

17:                                               ; preds = %12
  %18 = load i64, ptr %3, align 8, !tbaa !19
  %19 = icmp ult i64 %18, 16
  call void @llvm.assume(i1 %19)
  %20 = add nuw nsw i64 %18, 1
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %14, ptr noundef nonnull align 8 dereferenceable(1) %2, i64 %20, i1 false)
  br label %24

21:                                               ; preds = %12
  store ptr %15, ptr %13, align 8, !tbaa !22
  %22 = load i64, ptr %2, align 8, !tbaa !6
  store i64 %22, ptr %14, align 8, !tbaa !6
  %23 = load i64, ptr %3, align 8, !tbaa !19
  br label %24

24:                                               ; preds = %17, %21
  %25 = phi i64 [ %18, %17 ], [ %23, %21 ]
  %26 = getelementptr inbounds nuw i8, ptr %13, i64 8
  store i64 %25, ptr %26, align 8, !tbaa !19
  store ptr %2, ptr %1, align 8, !tbaa !22
  store i64 0, ptr %3, align 8, !tbaa !19
  store i8 0, ptr %2, align 8, !tbaa !6
  invoke void @__cxa_throw(ptr nonnull %13, ptr nonnull @_ZTINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE, ptr nonnull @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev) #16
          to label %38 unwind label %27

27:                                               ; preds = %10, %6, %0, %24
  %28 = landingpad { ptr, i32 }
          cleanup
  %29 = load ptr, ptr %1, align 8, !tbaa !22
  %30 = icmp eq ptr %29, %2
  br i1 %30, label %31, label %34

31:                                               ; preds = %27
  %32 = load i64, ptr %3, align 8, !tbaa !19
  %33 = icmp ult i64 %32, 16
  call void @llvm.assume(i1 %33)
  br label %37

34:                                               ; preds = %27
  %35 = load i64, ptr %2, align 8, !tbaa !6
  %36 = add i64 %35, 1
  call void @_ZdlPvm(ptr noundef %29, i64 noundef %36) #17
  br label %37

37:                                               ; preds = %34, %31
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #15
  resume { ptr, i32 } %28

38:                                               ; preds = %24
  unreachable
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nounwind uwtable
declare void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev(ptr noundef nonnull align 8 dereferenceable(32)) unnamed_addr #7

; Function Attrs: cold mustprogress noinline noreturn uwtable
define dso_local void @_Z7throw_Av() local_unnamed_addr #0 {
  %1 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.7)
  %2 = tail call ptr @__cxa_allocate_exception(i64 1) #15
  tail call void @__cxa_throw(ptr %2, ptr nonnull @_ZTI1A, ptr null) #16
  unreachable
}

; Function Attrs: cold mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #8 personality ptr @__gxx_personality_v0 {
  invoke void @_Z10throw_charv()
          to label %16 unwind label %1

1:                                                ; preds = %0
  %2 = landingpad { ptr, i32 }
          catch ptr @_ZTIc
  %3 = extractvalue { ptr, i32 } %2, 1
  %4 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIc) #15
  %5 = icmp eq i32 %3, %4
  br i1 %5, label %6, label %109

6:                                                ; preds = %1
  %7 = extractvalue { ptr, i32 } %2, 0
  %8 = tail call ptr @__cxa_begin_catch(ptr %7) #15
  %9 = load i8, ptr %8, align 1, !tbaa !6
  %10 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.8, i64 noundef 13)
          to label %11 unwind label %17

11:                                               ; preds = %6
  %12 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %9)
          to label %13 unwind label %17

13:                                               ; preds = %11
  %14 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %12, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %15 unwind label %17

15:                                               ; preds = %13
  tail call void @__cxa_end_catch() #15
  invoke void @_Z9throw_intv()
          to label %34 unwind label %19

16:                                               ; preds = %0
  unreachable

17:                                               ; preds = %13, %6, %11
  %18 = landingpad { ptr, i32 }
          cleanup
  br label %107

19:                                               ; preds = %15
  %20 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %21 = extractvalue { ptr, i32 } %20, 1
  %22 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi) #15
  %23 = icmp eq i32 %21, %22
  br i1 %23, label %24, label %109

24:                                               ; preds = %19
  %25 = extractvalue { ptr, i32 } %20, 0
  %26 = tail call ptr @__cxa_begin_catch(ptr %25) #15
  %27 = load i32, ptr %26, align 4, !tbaa !9
  %28 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.9, i64 noundef 12)
          to label %29 unwind label %35

29:                                               ; preds = %24
  %30 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i32 noundef %27)
          to label %31 unwind label %35

31:                                               ; preds = %29
  %32 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %30, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %33 unwind label %35

33:                                               ; preds = %31
  tail call void @__cxa_end_catch() #15
  invoke void @_Z11throw_floatv()
          to label %53 unwind label %37

34:                                               ; preds = %15
  unreachable

35:                                               ; preds = %31, %24, %29
  %36 = landingpad { ptr, i32 }
          cleanup
  br label %107

37:                                               ; preds = %33
  %38 = landingpad { ptr, i32 }
          catch ptr @_ZTIf
  %39 = extractvalue { ptr, i32 } %38, 1
  %40 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIf) #15
  %41 = icmp eq i32 %39, %40
  br i1 %41, label %42, label %109

42:                                               ; preds = %37
  %43 = extractvalue { ptr, i32 } %38, 0
  %44 = tail call ptr @__cxa_begin_catch(ptr %43) #15
  %45 = load float, ptr %44, align 4, !tbaa !11
  %46 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.10, i64 noundef 14)
          to label %47 unwind label %54

47:                                               ; preds = %42
  %48 = fpext float %45 to double
  %49 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, double noundef %48)
          to label %50 unwind label %54

50:                                               ; preds = %47
  %51 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %49, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %52 unwind label %54

52:                                               ; preds = %50
  tail call void @__cxa_end_catch() #15
  invoke void @_Z12throw_doublev()
          to label %71 unwind label %56

53:                                               ; preds = %33
  unreachable

54:                                               ; preds = %50, %47, %42
  %55 = landingpad { ptr, i32 }
          cleanup
  br label %107

56:                                               ; preds = %52
  %57 = landingpad { ptr, i32 }
          catch ptr @_ZTId
  %58 = extractvalue { ptr, i32 } %57, 1
  %59 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTId) #15
  %60 = icmp eq i32 %58, %59
  br i1 %60, label %61, label %109

61:                                               ; preds = %56
  %62 = extractvalue { ptr, i32 } %57, 0
  %63 = tail call ptr @__cxa_begin_catch(ptr %62) #15
  %64 = load double, ptr %63, align 8, !tbaa !13
  %65 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.11, i64 noundef 15)
          to label %66 unwind label %72

66:                                               ; preds = %61
  %67 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, double noundef %64)
          to label %68 unwind label %72

68:                                               ; preds = %66
  %69 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %67, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %70 unwind label %72

70:                                               ; preds = %68
  tail call void @__cxa_end_catch() #15
  invoke void @_Z12throw_stringv()
          to label %91 unwind label %74

71:                                               ; preds = %52
  unreachable

72:                                               ; preds = %68, %66, %61
  %73 = landingpad { ptr, i32 }
          cleanup
  br label %107

74:                                               ; preds = %70
  %75 = landingpad { ptr, i32 }
          catch ptr @_ZTINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
  %76 = extractvalue { ptr, i32 } %75, 1
  %77 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE) #15
  %78 = icmp eq i32 %76, %77
  br i1 %78, label %79, label %109

79:                                               ; preds = %74
  %80 = extractvalue { ptr, i32 } %75, 0
  %81 = tail call ptr @__cxa_begin_catch(ptr %80) #15
  %82 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.12, i64 noundef 20)
          to label %83 unwind label %92

83:                                               ; preds = %79
  %84 = load ptr, ptr %81, align 8, !tbaa !22
  %85 = getelementptr inbounds nuw i8, ptr %81, i64 8
  %86 = load i64, ptr %85, align 8, !tbaa !19
  %87 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef %84, i64 noundef %86)
          to label %88 unwind label %92

88:                                               ; preds = %83
  %89 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %87, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %90 unwind label %92

90:                                               ; preds = %88
  tail call void @__cxa_end_catch()
  invoke void @_Z7throw_Av()
          to label %104 unwind label %94

91:                                               ; preds = %70
  unreachable

92:                                               ; preds = %88, %83, %79
  %93 = landingpad { ptr, i32 }
          cleanup
  invoke void @__cxa_end_catch()
          to label %109 unwind label %111

94:                                               ; preds = %90
  %95 = landingpad { ptr, i32 }
          catch ptr @_ZTI1A
  %96 = extractvalue { ptr, i32 } %95, 1
  %97 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI1A) #15
  %98 = icmp eq i32 %96, %97
  br i1 %98, label %99, label %109

99:                                               ; preds = %94
  %100 = extractvalue { ptr, i32 } %95, 0
  %101 = tail call ptr @__cxa_begin_catch(ptr %100) #15
  %102 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.13, i64 noundef 9)
          to label %103 unwind label %105

103:                                              ; preds = %99
  tail call void @__cxa_end_catch()
  ret i32 0

104:                                              ; preds = %90
  unreachable

105:                                              ; preds = %99
  %106 = landingpad { ptr, i32 }
          cleanup
  invoke void @__cxa_end_catch()
          to label %109 unwind label %111

107:                                              ; preds = %17, %35, %54, %72
  %108 = phi { ptr, i32 } [ %18, %17 ], [ %36, %35 ], [ %55, %54 ], [ %73, %72 ]
  tail call void @__cxa_end_catch() #15
  br label %109

109:                                              ; preds = %107, %105, %92, %94, %74, %56, %37, %19, %1
  %110 = phi { ptr, i32 } [ %95, %94 ], [ %75, %74 ], [ %57, %56 ], [ %38, %37 ], [ %20, %19 ], [ %2, %1 ], [ %93, %92 ], [ %106, %105 ], [ %108, %107 ]
  resume { ptr, i32 } %110

111:                                              ; preds = %105, %92
  %112 = landingpad { ptr, i32 }
          catch ptr null
  %113 = extractvalue { ptr, i32 } %112, 0
  tail call void @__clang_call_terminate(ptr %113) #18
  unreachable
}

; Function Attrs: nofree nosync nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #9

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) local_unnamed_addr #10 comdat {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #15
  tail call void @_ZSt9terminatev() #18
  unreachable
}

; Function Attrs: cold nofree noreturn
declare void @_ZSt9terminatev() local_unnamed_addr #11

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #12

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #13

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #4

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8), double noundef) local_unnamed_addr #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #14

attributes #0 = { cold mustprogress noinline noreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { inlinehint mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold noreturn }
attributes #4 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress noinline noreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { cold mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nofree nosync nounwind memory(none) }
attributes #10 = { noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { cold nofree noreturn }
attributes #12 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #13 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #14 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #15 = { nounwind }
attributes #16 = { noreturn }
attributes #17 = { builtin nounwind }
attributes #18 = { noreturn nounwind }

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
!12 = !{!"float", !7, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"double", !7, i64 0}
!15 = !{!16, !17, i64 0}
!16 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderE", !17, i64 0}
!17 = !{!"p1 omnipotent char", !18, i64 0}
!18 = !{!"any pointer", !7, i64 0}
!19 = !{!20, !21, i64 8}
!20 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", !16, i64 0, !21, i64 8, !7, i64 16}
!21 = !{!"long", !7, i64 0}
!22 = !{!20, !17, i64 0}
