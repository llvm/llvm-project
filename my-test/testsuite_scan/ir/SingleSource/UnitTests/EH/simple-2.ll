; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/EH/simple-2.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/EH/simple-2.cpp"
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
%"class.std::allocator" = type { i8 }

$_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_ = comdat any

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
@.str.14 = private unnamed_addr constant [50 x i8] c"basic_string: construction from null is not valid\00", align 1

; Function Attrs: alwaysinline cold mustprogress noreturn uwtable
define dso_local void @_Z10throw_charv() local_unnamed_addr #0 {
  %1 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str)
  %2 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(ptr noundef nonnull align 8 dereferenceable(8) %1, i8 noundef 97)
  %3 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull @.str.1)
  %4 = tail call ptr @__cxa_allocate_exception(i64 1) #17
  store i8 97, ptr %4, align 16, !tbaa !6
  tail call void @__cxa_throw(ptr nonnull %4, ptr nonnull @_ZTIc, ptr null) #18
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

; Function Attrs: alwaysinline cold mustprogress noreturn uwtable
define dso_local void @_Z9throw_intv() local_unnamed_addr #0 {
  %1 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.2)
  %2 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) %1, i32 noundef 37)
  %3 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull @.str.1)
  %4 = tail call ptr @__cxa_allocate_exception(i64 4) #17
  store i32 37, ptr %4, align 16, !tbaa !9
  tail call void @__cxa_throw(ptr nonnull %4, ptr nonnull @_ZTIi, ptr null) #18
  unreachable
}

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8), i32 noundef) local_unnamed_addr #4

; Function Attrs: alwaysinline cold mustprogress noreturn uwtable
define dso_local void @_Z11throw_floatv() local_unnamed_addr #0 {
  %1 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.3)
  %2 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEf(ptr noundef nonnull align 8 dereferenceable(8) %1, float noundef 0x4042F6A7E0000000)
  %3 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull @.str.1)
  %4 = tail call ptr @__cxa_allocate_exception(i64 4) #17
  store float 0x4042F6A7E0000000, ptr %4, align 16, !tbaa !11
  tail call void @__cxa_throw(ptr nonnull %4, ptr nonnull @_ZTIf, ptr null) #18
  unreachable
}

; Function Attrs: mustprogress uwtable
declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEf(ptr noundef nonnull align 8 dereferenceable(8), float noundef) local_unnamed_addr #5

; Function Attrs: alwaysinline cold mustprogress noreturn uwtable
define dso_local void @_Z12throw_doublev() local_unnamed_addr #0 {
  %1 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.4)
  %2 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEd(ptr noundef nonnull align 8 dereferenceable(8) %1, double noundef 3.792700e+01)
  %3 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull @.str.1)
  %4 = tail call ptr @__cxa_allocate_exception(i64 8) #17
  store double 3.792700e+01, ptr %4, align 16, !tbaa !13
  tail call void @__cxa_throw(ptr nonnull %4, ptr nonnull @_ZTId, ptr null) #18
  unreachable
}

; Function Attrs: mustprogress uwtable
declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEd(ptr noundef nonnull align 8 dereferenceable(8), double noundef) local_unnamed_addr #5

; Function Attrs: alwaysinline mustprogress noreturn uwtable
define dso_local void @_Z12throw_stringv() local_unnamed_addr #6 personality ptr @__gxx_personality_v0 {
  %1 = alloca %"class.std::__cxx11::basic_string", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #17
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
  %13 = call ptr @__cxa_allocate_exception(i64 32) #17
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
  invoke void @__cxa_throw(ptr nonnull %13, ptr nonnull @_ZTINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE, ptr nonnull @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev) #18
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
  call void @_ZdlPvm(ptr noundef %29, i64 noundef %36) #19
  br label %37

37:                                               ; preds = %34, %31
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #17
  resume { ptr, i32 } %28

38:                                               ; preds = %24
  unreachable
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef %1, ptr noundef nonnull align 1 dereferenceable(1) %2) unnamed_addr #5 comdat personality ptr @__gxx_personality_v0 {
  %4 = alloca i64, align 8
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %5, ptr %0, align 8, !tbaa !15
  %6 = icmp eq ptr %1, null
  br i1 %6, label %7, label %8

7:                                                ; preds = %3
  tail call void @_ZSt19__throw_logic_errorPKc(ptr noundef nonnull @.str.14) #3
  unreachable

8:                                                ; preds = %3
  %9 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %1) #17
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #17
  store i64 %9, ptr %4, align 8, !tbaa !23
  %10 = icmp ugt i64 %9, 15
  br i1 %10, label %11, label %14

11:                                               ; preds = %8
  %12 = call noundef ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %4, i64 noundef 0)
  store ptr %12, ptr %0, align 8, !tbaa !22
  %13 = load i64, ptr %4, align 8, !tbaa !23
  store i64 %13, ptr %5, align 8, !tbaa !6
  br label %14

14:                                               ; preds = %8, %11
  %15 = phi ptr [ %12, %11 ], [ %5, %8 ]
  switch i64 %9, label %18 [
    i64 1, label %16
    i64 0, label %19
  ]

16:                                               ; preds = %14
  %17 = load i8, ptr %1, align 1, !tbaa !6
  store i8 %17, ptr %15, align 1, !tbaa !6
  br label %19

18:                                               ; preds = %14
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %15, ptr nonnull align 1 %1, i64 %9, i1 false)
  br label %19

19:                                               ; preds = %18, %16, %14
  %20 = load i64, ptr %4, align 8, !tbaa !23
  %21 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %20, ptr %21, align 8, !tbaa !19
  %22 = load ptr, ptr %0, align 8, !tbaa !22
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 %20
  store i8 0, ptr %23, align 1, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #17
  ret void
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nounwind uwtable
declare void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev(ptr noundef nonnull align 8 dereferenceable(32)) unnamed_addr #7

; Function Attrs: alwaysinline cold mustprogress noreturn uwtable
define dso_local void @_Z7throw_Av() local_unnamed_addr #0 {
  %1 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.7)
  %2 = tail call ptr @__cxa_allocate_exception(i64 1) #17
  tail call void @__cxa_throw(ptr %2, ptr nonnull @_ZTI1A, ptr null) #18
  unreachable
}

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #8 personality ptr @__gxx_personality_v0 {
  %1 = alloca i8, align 4
  %2 = alloca %"class.std::__cxx11::basic_string", align 8
  %3 = alloca %"class.std::allocator", align 1
  %4 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str, i64 noundef 15)
          to label %5 unwind label %23

5:                                                ; preds = %0
  call void @llvm.lifetime.start.p0(ptr nonnull %1)
  store i8 97, ptr %1, align 4, !tbaa !6
  %6 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !24
  %7 = getelementptr i8, ptr %6, i64 -24
  %8 = load i64, ptr %7, align 8
  %9 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %8
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %11 = load i64, ptr %10, align 8, !tbaa !26
  %12 = icmp eq i64 %11, 0
  br i1 %12, label %15, label %13

13:                                               ; preds = %5
  %14 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %1, i64 noundef 1)
          to label %17 unwind label %23

15:                                               ; preds = %5
  %16 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef 97)
          to label %17 unwind label %23

17:                                               ; preds = %13, %15
  %18 = phi ptr [ %14, %13 ], [ @_ZSt4cout, %15 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %1)
  %19 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %18, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %20 unwind label %23

20:                                               ; preds = %17
  %21 = call ptr @__cxa_allocate_exception(i64 1) #17
  store i8 97, ptr %21, align 16, !tbaa !6
  invoke void @__cxa_throw(ptr nonnull %21, ptr nonnull @_ZTIc, ptr null) #18
          to label %22 unwind label %23

22:                                               ; preds = %20
  unreachable

23:                                               ; preds = %17, %15, %13, %0, %20
  %24 = landingpad { ptr, i32 }
          catch ptr @_ZTIc
  %25 = extractvalue { ptr, i32 } %24, 1
  %26 = call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIc) #17
  %27 = icmp eq i32 %25, %26
  br i1 %27, label %28, label %193

28:                                               ; preds = %23
  %29 = extractvalue { ptr, i32 } %24, 0
  %30 = call ptr @__cxa_begin_catch(ptr %29) #17
  %31 = load i8, ptr %30, align 1, !tbaa !6
  %32 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.8, i64 noundef 13)
          to label %33 unwind label %46

33:                                               ; preds = %28
  %34 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %31)
          to label %35 unwind label %46

35:                                               ; preds = %33
  %36 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %34, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %37 unwind label %46

37:                                               ; preds = %35
  call void @__cxa_end_catch() #17
  %38 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.2, i64 noundef 14)
          to label %39 unwind label %48

39:                                               ; preds = %37
  %40 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i32 noundef 37)
          to label %41 unwind label %48

41:                                               ; preds = %39
  %42 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %40, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %43 unwind label %48

43:                                               ; preds = %41
  %44 = call ptr @__cxa_allocate_exception(i64 4) #17
  store i32 37, ptr %44, align 16, !tbaa !9
  invoke void @__cxa_throw(ptr nonnull %44, ptr nonnull @_ZTIi, ptr null) #18
          to label %45 unwind label %48

45:                                               ; preds = %43
  unreachable

46:                                               ; preds = %35, %28, %33
  %47 = landingpad { ptr, i32 }
          cleanup
  br label %191

48:                                               ; preds = %41, %37, %43, %39
  %49 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %50 = extractvalue { ptr, i32 } %49, 1
  %51 = call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi) #17
  %52 = icmp eq i32 %50, %51
  br i1 %52, label %53, label %193

53:                                               ; preds = %48
  %54 = extractvalue { ptr, i32 } %49, 0
  %55 = call ptr @__cxa_begin_catch(ptr %54) #17
  %56 = load i32, ptr %55, align 4, !tbaa !9
  %57 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.9, i64 noundef 12)
          to label %58 unwind label %71

58:                                               ; preds = %53
  %59 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i32 noundef %56)
          to label %60 unwind label %71

60:                                               ; preds = %58
  %61 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %59, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %62 unwind label %71

62:                                               ; preds = %60
  call void @__cxa_end_catch() #17
  %63 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.3, i64 noundef 16)
          to label %64 unwind label %73

64:                                               ; preds = %62
  %65 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, double noundef 0x4042F6A7E0000000)
          to label %66 unwind label %73

66:                                               ; preds = %64
  %67 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %65, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %68 unwind label %73

68:                                               ; preds = %66
  %69 = call ptr @__cxa_allocate_exception(i64 4) #17
  store float 0x4042F6A7E0000000, ptr %69, align 16, !tbaa !11
  invoke void @__cxa_throw(ptr nonnull %69, ptr nonnull @_ZTIf, ptr null) #18
          to label %70 unwind label %73

70:                                               ; preds = %68
  unreachable

71:                                               ; preds = %60, %53, %58
  %72 = landingpad { ptr, i32 }
          cleanup
  br label %191

73:                                               ; preds = %66, %64, %62, %68
  %74 = landingpad { ptr, i32 }
          catch ptr @_ZTIf
  %75 = extractvalue { ptr, i32 } %74, 1
  %76 = call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIf) #17
  %77 = icmp eq i32 %75, %76
  br i1 %77, label %78, label %193

78:                                               ; preds = %73
  %79 = extractvalue { ptr, i32 } %74, 0
  %80 = call ptr @__cxa_begin_catch(ptr %79) #17
  %81 = load float, ptr %80, align 4, !tbaa !11
  %82 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.10, i64 noundef 14)
          to label %83 unwind label %97

83:                                               ; preds = %78
  %84 = fpext float %81 to double
  %85 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, double noundef %84)
          to label %86 unwind label %97

86:                                               ; preds = %83
  %87 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %85, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %88 unwind label %97

88:                                               ; preds = %86
  call void @__cxa_end_catch() #17
  %89 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.4, i64 noundef 17)
          to label %90 unwind label %99

90:                                               ; preds = %88
  %91 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, double noundef 3.792700e+01)
          to label %92 unwind label %99

92:                                               ; preds = %90
  %93 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %91, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %94 unwind label %99

94:                                               ; preds = %92
  %95 = call ptr @__cxa_allocate_exception(i64 8) #17
  store double 3.792700e+01, ptr %95, align 16, !tbaa !13
  invoke void @__cxa_throw(ptr nonnull %95, ptr nonnull @_ZTId, ptr null) #18
          to label %96 unwind label %99

96:                                               ; preds = %94
  unreachable

97:                                               ; preds = %86, %83, %78
  %98 = landingpad { ptr, i32 }
          cleanup
  br label %191

99:                                               ; preds = %92, %90, %88, %94
  %100 = landingpad { ptr, i32 }
          catch ptr @_ZTId
  %101 = extractvalue { ptr, i32 } %100, 1
  %102 = call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTId) #17
  %103 = icmp eq i32 %101, %102
  br i1 %103, label %104, label %193

104:                                              ; preds = %99
  %105 = extractvalue { ptr, i32 } %100, 0
  %106 = call ptr @__cxa_begin_catch(ptr %105) #17
  %107 = load double, ptr %106, align 8, !tbaa !13
  %108 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.11, i64 noundef 15)
          to label %109 unwind label %159

109:                                              ; preds = %104
  %110 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, double noundef %107)
          to label %111 unwind label %159

111:                                              ; preds = %109
  %112 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %110, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %113 unwind label %159

113:                                              ; preds = %111
  call void @__cxa_end_catch() #17
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #17
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #17
  invoke void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_(ptr noundef nonnull align 8 dereferenceable(32) %2, ptr noundef nonnull @.str.5, ptr noundef nonnull align 1 dereferenceable(1) %3)
          to label %114 unwind label %139

114:                                              ; preds = %113
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #17
  %115 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.6, i64 noundef 22)
          to label %116 unwind label %141

116:                                              ; preds = %114
  %117 = load ptr, ptr %2, align 8, !tbaa !22
  %118 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %119 = load i64, ptr %118, align 8, !tbaa !19
  %120 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef %117, i64 noundef %119)
          to label %121 unwind label %141

121:                                              ; preds = %116
  %122 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %120, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %123 unwind label %141

123:                                              ; preds = %121
  %124 = call ptr @__cxa_allocate_exception(i64 32) #17
  %125 = getelementptr inbounds nuw i8, ptr %124, i64 16
  store ptr %125, ptr %124, align 8, !tbaa !15
  %126 = load ptr, ptr %2, align 8, !tbaa !22
  %127 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %128 = icmp eq ptr %126, %127
  br i1 %128, label %129, label %133

129:                                              ; preds = %123
  %130 = load i64, ptr %118, align 8, !tbaa !19
  %131 = icmp ult i64 %130, 16
  call void @llvm.assume(i1 %131)
  %132 = add nuw nsw i64 %130, 1
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %125, ptr noundef nonnull align 8 dereferenceable(1) %127, i64 %132, i1 false)
  br label %136

133:                                              ; preds = %123
  store ptr %126, ptr %124, align 8, !tbaa !22
  %134 = load i64, ptr %127, align 8, !tbaa !6
  store i64 %134, ptr %125, align 8, !tbaa !6
  %135 = load i64, ptr %118, align 8, !tbaa !19
  br label %136

136:                                              ; preds = %129, %133
  %137 = phi i64 [ %130, %129 ], [ %135, %133 ]
  %138 = getelementptr inbounds nuw i8, ptr %124, i64 8
  store i64 %137, ptr %138, align 8, !tbaa !19
  store ptr %127, ptr %2, align 8, !tbaa !22
  store i64 0, ptr %118, align 8, !tbaa !19
  store i8 0, ptr %127, align 8, !tbaa !6
  invoke void @__cxa_throw(ptr nonnull %124, ptr nonnull @_ZTINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE, ptr nonnull @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev) #18
          to label %158 unwind label %141

139:                                              ; preds = %113
  %140 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #17
  br label %153

141:                                              ; preds = %121, %116, %114, %136
  %142 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
  %143 = load ptr, ptr %2, align 8, !tbaa !22
  %144 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %145 = icmp eq ptr %143, %144
  br i1 %145, label %146, label %150

146:                                              ; preds = %141
  %147 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %148 = load i64, ptr %147, align 8, !tbaa !19
  %149 = icmp ult i64 %148, 16
  call void @llvm.assume(i1 %149)
  br label %153

150:                                              ; preds = %141
  %151 = load i64, ptr %144, align 8, !tbaa !6
  %152 = add i64 %151, 1
  call void @_ZdlPvm(ptr noundef %143, i64 noundef %152) #19
  br label %153

153:                                              ; preds = %150, %146, %139
  %154 = phi { ptr, i32 } [ %140, %139 ], [ %142, %146 ], [ %142, %150 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #17
  %155 = extractvalue { ptr, i32 } %154, 1
  %156 = call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE) #17
  %157 = icmp eq i32 %155, %156
  br i1 %157, label %161, label %193

158:                                              ; preds = %136
  unreachable

159:                                              ; preds = %111, %109, %104
  %160 = landingpad { ptr, i32 }
          cleanup
  br label %191

161:                                              ; preds = %153
  %162 = extractvalue { ptr, i32 } %154, 0
  %163 = call ptr @__cxa_begin_catch(ptr %162) #17
  %164 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.12, i64 noundef 20)
          to label %165 unwind label %177

165:                                              ; preds = %161
  %166 = load ptr, ptr %163, align 8, !tbaa !22
  %167 = getelementptr inbounds nuw i8, ptr %163, i64 8
  %168 = load i64, ptr %167, align 8, !tbaa !19
  %169 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef %166, i64 noundef %168)
          to label %170 unwind label %177

170:                                              ; preds = %165
  %171 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %169, ptr noundef nonnull @.str.1, i64 noundef 1)
          to label %172 unwind label %177

172:                                              ; preds = %170
  call void @__cxa_end_catch()
  %173 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.7, i64 noundef 11)
          to label %174 unwind label %179

174:                                              ; preds = %172
  %175 = call ptr @__cxa_allocate_exception(i64 1) #17
  invoke void @__cxa_throw(ptr %175, ptr nonnull @_ZTI1A, ptr null) #18
          to label %176 unwind label %179

176:                                              ; preds = %174
  unreachable

177:                                              ; preds = %170, %165, %161
  %178 = landingpad { ptr, i32 }
          cleanup
  invoke void @__cxa_end_catch()
          to label %193 unwind label %195

179:                                              ; preds = %172, %174
  %180 = landingpad { ptr, i32 }
          catch ptr @_ZTI1A
  %181 = extractvalue { ptr, i32 } %180, 1
  %182 = call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI1A) #17
  %183 = icmp eq i32 %181, %182
  br i1 %183, label %184, label %193

184:                                              ; preds = %179
  %185 = extractvalue { ptr, i32 } %180, 0
  %186 = call ptr @__cxa_begin_catch(ptr %185) #17
  %187 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.13, i64 noundef 9)
          to label %188 unwind label %189

188:                                              ; preds = %184
  call void @__cxa_end_catch()
  ret i32 0

189:                                              ; preds = %184
  %190 = landingpad { ptr, i32 }
          cleanup
  invoke void @__cxa_end_catch()
          to label %193 unwind label %195

191:                                              ; preds = %46, %71, %97, %159
  %192 = phi { ptr, i32 } [ %47, %46 ], [ %72, %71 ], [ %98, %97 ], [ %160, %159 ]
  call void @__cxa_end_catch() #17
  br label %193

193:                                              ; preds = %191, %189, %177, %179, %153, %99, %73, %48, %23
  %194 = phi { ptr, i32 } [ %180, %179 ], [ %154, %153 ], [ %100, %99 ], [ %74, %73 ], [ %49, %48 ], [ %24, %23 ], [ %178, %177 ], [ %190, %189 ], [ %192, %191 ]
  resume { ptr, i32 } %194

195:                                              ; preds = %189, %177
  %196 = landingpad { ptr, i32 }
          catch ptr null
  %197 = extractvalue { ptr, i32 } %196, 0
  call void @__clang_call_terminate(ptr %197) #20
  unreachable
}

; Function Attrs: nofree nosync nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #9

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) local_unnamed_addr #10 comdat {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #17
  tail call void @_ZSt9terminatev() #20
  unreachable
}

; Function Attrs: cold nofree noreturn
declare void @_ZSt9terminatev() local_unnamed_addr #11

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #12

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #13

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #14

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #4

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8), double noundef) local_unnamed_addr #4

; Function Attrs: cold noreturn
declare void @_ZSt19__throw_logic_errorPKc(ptr noundef) local_unnamed_addr #15

declare noundef ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(ptr noundef nonnull align 8 dereferenceable(32), ptr noundef nonnull align 8 dereferenceable(8), i64 noundef) local_unnamed_addr #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #16

attributes #0 = { alwaysinline cold mustprogress noreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { inlinehint mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold noreturn }
attributes #4 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { alwaysinline mustprogress noreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nofree nosync nounwind memory(none) }
attributes #10 = { noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { cold nofree noreturn }
attributes #12 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #13 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #14 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #15 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #16 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #17 = { nounwind }
attributes #18 = { noreturn }
attributes #19 = { builtin nounwind }
attributes #20 = { noreturn nounwind }

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
!23 = !{!21, !21, i64 0}
!24 = !{!25, !25, i64 0}
!25 = !{!"vtable pointer", !8, i64 0}
!26 = !{!27, !21, i64 16}
!27 = !{!"_ZTSSt8ios_base", !21, i64 8, !21, i64 16, !28, i64 24, !29, i64 28, !29, i64 32, !30, i64 40, !31, i64 48, !7, i64 64, !10, i64 192, !32, i64 200, !33, i64 208}
!28 = !{!"_ZTSSt13_Ios_Fmtflags", !7, i64 0}
!29 = !{!"_ZTSSt12_Ios_Iostate", !7, i64 0}
!30 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !18, i64 0}
!31 = !{!"_ZTSNSt8ios_base6_WordsE", !18, i64 0, !21, i64 8}
!32 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !18, i64 0}
!33 = !{!"_ZTSSt6locale", !34, i64 0}
!34 = !{!"p1 _ZTSNSt6locale5_ImplE", !18, i64 0}
