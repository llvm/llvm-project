; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc-C++/stepanov_container.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc-C++/stepanov_container.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::vector" = type { %"struct.std::_Vector_base" }
%"struct.std::_Vector_base" = type { %"struct.std::_Vector_base<double, std::allocator<double>>::_Vector_impl" }
%"struct.std::_Vector_base<double, std::allocator<double>>::_Vector_impl" = type { %"struct.std::_Vector_base<double, std::allocator<double>>::_Vector_impl_data" }
%"struct.std::_Vector_base<double, std::allocator<double>>::_Vector_impl_data" = type { ptr, ptr, ptr }
%"struct.std::_Deque_iterator" = type { ptr, ptr, ptr, ptr }
%"class.std::deque" = type { %"class.std::_Deque_base" }
%"class.std::_Deque_base" = type { %"struct.std::_Deque_base<double, std::allocator<double>>::_Deque_impl" }
%"struct.std::_Deque_base<double, std::allocator<double>>::_Deque_impl" = type { %"struct.std::_Deque_base<double, std::allocator<double>>::_Deque_impl_data" }
%"struct.std::_Deque_base<double, std::allocator<double>>::_Deque_impl_data" = type { ptr, i64, %"struct.std::_Deque_iterator", %"struct.std::_Deque_iterator" }
%"class.std::__cxx11::list" = type { %"class.std::__cxx11::_List_base" }
%"class.std::__cxx11::_List_base" = type { %"struct.std::__cxx11::_List_base<double, std::allocator<double>>::_List_impl" }
%"struct.std::__cxx11::_List_base<double, std::allocator<double>>::_List_impl" = type { %"struct.std::__detail::_List_node_header" }
%"struct.std::__detail::_List_node_header" = type { %"struct.std::__detail::_List_node_base", i64 }
%"struct.std::__detail::_List_node_base" = type { ptr, ptr }
%"struct.std::__detail::_Scratch_list" = type { %"struct.std::__detail::_List_node_base" }
%"class.std::set" = type { %"class.std::_Rb_tree" }
%"class.std::_Rb_tree" = type { %"struct.std::_Rb_tree<double, double, std::_Identity<double>, std::less<double>>::_Rb_tree_impl" }
%"struct.std::_Rb_tree<double, double, std::_Identity<double>, std::less<double>>::_Rb_tree_impl" = type { [8 x i8], %"struct.std::_Rb_tree_header" }
%"struct.std::_Rb_tree_header" = type { %"struct.std::_Rb_tree_node_base", i64 }
%"struct.std::_Rb_tree_node_base" = type { i32, ptr, ptr, ptr }
%"struct.std::_Rb_tree<double, double, std::_Identity<double>, std::less<double>>::_Alloc_node" = type { ptr }
%"class.std::multiset" = type { %"class.std::_Rb_tree" }
%"struct.__gnu_cxx::__ops::_Iter_less_iter" = type { i8 }

$_ZNSt6vectorIdSaIdEED2Ev = comdat any

$_ZNSt5dequeIdSaIdEED2Ev = comdat any

$_ZNSt7__cxx114listIdSaIdEE4sortEv = comdat any

$_ZNSt3setIdSt4lessIdESaIdEEC2IPdEET_S6_ = comdat any

$__clang_call_terminate = comdat any

$_ZNSt8_Rb_treeIddSt9_IdentityIdESt4lessIdESaIdEED2Ev = comdat any

$_ZNSt8_Rb_treeIddSt9_IdentityIdESt4lessIdESaIdEE8_M_eraseEPSt13_Rb_tree_nodeIdE = comdat any

$_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_ = comdat any

$_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_ = comdat any

$_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_ = comdat any

$_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_ = comdat any

$_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_ = comdat any

$_ZSt13__heap_selectIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_T0_ = comdat any

$_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_ = comdat any

$_ZNSt5dequeIdSaIdEE18_M_fill_initializeERKd = comdat any

$_ZNSt11_Deque_baseIdSaIdEED2Ev = comdat any

$_ZNSt11_Deque_baseIdSaIdEE17_M_initialize_mapEm = comdat any

$_ZSt16__introsort_loopISt15_Deque_iteratorIdRdPdElN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S7_T0_T1_ = comdat any

$_ZSt22__final_insertion_sortISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S7_T0_ = comdat any

$_ZSt14__partial_sortISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S7_S7_T0_ = comdat any

$_ZSt27__unguarded_partition_pivotISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops15_Iter_less_iterEET_S7_S7_T0_ = comdat any

$_ZSt13__heap_selectISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S7_S7_T0_ = comdat any

$_ZSt13__adjust_heapISt15_Deque_iteratorIdRdPdEldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S8_T1_T2_ = comdat any

$_ZSt16__insertion_sortISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S7_T0_ = comdat any

$_ZSt24__copy_move_backward_ditILb1EdRdPdSt15_Deque_iteratorIdS0_S1_EET3_S2_IT0_T1_T2_ES8_S4_ = comdat any

$_ZNSt8_Rb_treeIddSt9_IdentityIdESt4lessIdESaIdEE16_M_insert_equal_IRdNS5_11_Alloc_nodeEEESt17_Rb_tree_iteratorIdESt23_Rb_tree_const_iteratorIdEOT_RT0_ = comdat any

$_ZNSt8_Rb_treeIddSt9_IdentityIdESt4lessIdESaIdEE28_M_get_insert_hint_equal_posESt23_Rb_tree_const_iteratorIdERKd = comdat any

@result_times = dso_local global %"class.std::vector" zeroinitializer, align 8
@__dso_handle = external hidden global i8
@.str = private unnamed_addr constant [49 x i8] c"cannot create std::vector larger than max_size()\00", align 1
@.str.1 = private unnamed_addr constant [48 x i8] c"cannot create std::deque larger than max_size()\00", align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_stepanov_container.cpp, ptr null }]

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZNSt6vectorIdSaIdEED2Ev(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #0 comdat personality ptr @__gxx_personality_v0 {
  %2 = load ptr, ptr %0, align 8, !tbaa !6
  %3 = icmp eq ptr %2, null
  br i1 %3, label %10, label %4

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %6 = load ptr, ptr %5, align 8, !tbaa !12
  %7 = ptrtoint ptr %6 to i64
  %8 = ptrtoint ptr %2 to i64
  %9 = sub i64 %7, %8
  tail call void @_ZdlPvm(ptr noundef nonnull %2, i64 noundef %9) #22
  br label %10

10:                                               ; preds = %1, %4
  ret void
}

; Function Attrs: nofree nounwind
declare i32 @__cxa_atexit(ptr, ptr, ptr) local_unnamed_addr #1

; Function Attrs: mustprogress uwtable
define dso_local void @_Z3runPFvPdS_iES_S_i(ptr noundef readonly captures(none) %0, ptr noundef %1, ptr noundef %2, i32 noundef %3) local_unnamed_addr #2 {
  %5 = icmp sgt i32 %3, 0
  br i1 %5, label %6, label %10

6:                                                ; preds = %4, %6
  %7 = phi i32 [ %8, %6 ], [ %3, %4 ]
  %8 = add nsw i32 %7, -1
  tail call void %0(ptr noundef %1, ptr noundef %2, i32 noundef %8)
  %9 = icmp samesign ugt i32 %7, 1
  br i1 %9, label %6, label %10, !llvm.loop !13

10:                                               ; preds = %6, %4
  ret void
}

; Function Attrs: mustprogress uwtable
define dso_local void @_Z10array_testPdS_i(ptr noundef %0, ptr noundef %1, i32 %2) local_unnamed_addr #2 {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = tail call i64 @llvm.smax.i64(i64 %6, i64 -1)
  %8 = tail call noalias noundef nonnull ptr @_Znam(i64 noundef %7) #23
  %9 = icmp sgt i64 %6, 8
  br i1 %9, label %10, label %11, !prof !15

10:                                               ; preds = %3
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %8, ptr align 8 %0, i64 %6, i1 false)
  br label %15

11:                                               ; preds = %3
  %12 = icmp eq i64 %6, 8
  br i1 %12, label %13, label %15

13:                                               ; preds = %11
  %14 = load double, ptr %0, align 8, !tbaa !16
  store double %14, ptr %8, align 8, !tbaa !16
  br label %15

15:                                               ; preds = %10, %11, %13
  %16 = getelementptr inbounds i8, ptr %8, i64 %6
  %17 = icmp eq ptr %1, %0
  br i1 %17, label %47, label %18

18:                                               ; preds = %15
  %19 = ashr exact i64 %6, 3
  %20 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %19, i1 true)
  %21 = shl nuw nsw i64 %20, 1
  %22 = xor i64 %21, 126
  tail call void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_(ptr noundef nonnull %8, ptr noundef nonnull %16, i64 noundef %22, i8 undef)
  tail call void @_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_(ptr noundef nonnull %8, ptr noundef nonnull %16, i8 undef)
  br label %23

23:                                               ; preds = %18, %27
  %24 = phi ptr [ %25, %27 ], [ %8, %18 ]
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 8
  %26 = icmp eq ptr %25, %16
  br i1 %26, label %47, label %27

27:                                               ; preds = %23
  %28 = load double, ptr %24, align 8, !tbaa !16
  %29 = load double, ptr %25, align 8, !tbaa !16
  %30 = fcmp oeq double %28, %29
  br i1 %30, label %31, label %23, !llvm.loop !18

31:                                               ; preds = %27
  %32 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %33 = icmp eq ptr %32, %16
  br i1 %33, label %47, label %34

34:                                               ; preds = %31, %42
  %35 = phi double [ %43, %42 ], [ %28, %31 ]
  %36 = phi ptr [ %45, %42 ], [ %32, %31 ]
  %37 = phi ptr [ %44, %42 ], [ %24, %31 ]
  %38 = load double, ptr %36, align 8, !tbaa !16
  %39 = fcmp oeq double %35, %38
  br i1 %39, label %42, label %40

40:                                               ; preds = %34
  %41 = getelementptr inbounds nuw i8, ptr %37, i64 8
  store double %38, ptr %41, align 8, !tbaa !16
  br label %42

42:                                               ; preds = %40, %34
  %43 = phi double [ %35, %34 ], [ %38, %40 ]
  %44 = phi ptr [ %37, %34 ], [ %41, %40 ]
  %45 = getelementptr inbounds nuw i8, ptr %36, i64 8
  %46 = icmp eq ptr %45, %16
  br i1 %46, label %47, label %34, !llvm.loop !19

47:                                               ; preds = %23, %42, %15, %31
  tail call void @_ZdaPv(ptr noundef nonnull %8) #22
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) local_unnamed_addr #4

; Function Attrs: nobuiltin nounwind
declare void @_ZdaPv(ptr noundef) local_unnamed_addr #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

; Function Attrs: mustprogress uwtable
define dso_local void @_Z19vector_pointer_testPdS_i(ptr noundef %0, ptr noundef %1, i32 %2) local_unnamed_addr #2 personality ptr @__gxx_personality_v0 {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = icmp ugt i64 %6, 9223372036854775800
  br i1 %7, label %8, label %9

8:                                                ; preds = %3
  tail call void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str) #24
  unreachable

9:                                                ; preds = %3
  %10 = icmp eq ptr %1, %0
  br i1 %10, label %11, label %13

11:                                               ; preds = %9
  %12 = getelementptr inbounds nuw i8, ptr null, i64 %6
  br label %22

13:                                               ; preds = %9
  %14 = tail call noalias noundef nonnull ptr @_Znwm(i64 noundef %6) #23
  %15 = getelementptr inbounds nuw i8, ptr %14, i64 %6
  %16 = icmp samesign ugt i64 %6, 8
  br i1 %16, label %17, label %18, !prof !20

17:                                               ; preds = %13
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %14, ptr align 8 %0, i64 %6, i1 false)
  br label %22

18:                                               ; preds = %13
  %19 = icmp eq i64 %6, 8
  br i1 %19, label %20, label %22

20:                                               ; preds = %18
  %21 = load double, ptr %0, align 8, !tbaa !16
  store double %21, ptr %14, align 8, !tbaa !16
  br label %22

22:                                               ; preds = %20, %18, %17, %11
  %23 = phi ptr [ %12, %11 ], [ %15, %17 ], [ %15, %20 ], [ %15, %18 ]
  %24 = phi ptr [ null, %11 ], [ %14, %17 ], [ %14, %20 ], [ %14, %18 ]
  %25 = ptrtoint ptr %24 to i64
  %26 = icmp eq ptr %24, %23
  br i1 %26, label %59, label %27

27:                                               ; preds = %22
  %28 = ptrtoint ptr %23 to i64
  %29 = sub i64 %28, %25
  %30 = ashr exact i64 %29, 3
  %31 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %30, i1 true)
  %32 = shl nuw nsw i64 %31, 1
  %33 = xor i64 %32, 126
  invoke void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_(ptr noundef nonnull %24, ptr noundef nonnull %23, i64 noundef %33, i8 undef)
          to label %34 unwind label %65

34:                                               ; preds = %27
  invoke void @_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_(ptr noundef nonnull %24, ptr noundef nonnull %23, i8 undef)
          to label %35 unwind label %65

35:                                               ; preds = %34, %39
  %36 = phi ptr [ %37, %39 ], [ %24, %34 ]
  %37 = getelementptr inbounds nuw i8, ptr %36, i64 8
  %38 = icmp eq ptr %37, %23
  br i1 %38, label %59, label %39

39:                                               ; preds = %35
  %40 = load double, ptr %36, align 8, !tbaa !16
  %41 = load double, ptr %37, align 8, !tbaa !16
  %42 = fcmp oeq double %40, %41
  br i1 %42, label %43, label %35, !llvm.loop !18

43:                                               ; preds = %39
  %44 = getelementptr inbounds nuw i8, ptr %36, i64 16
  %45 = icmp eq ptr %44, %23
  br i1 %45, label %59, label %46

46:                                               ; preds = %43, %54
  %47 = phi double [ %55, %54 ], [ %40, %43 ]
  %48 = phi ptr [ %57, %54 ], [ %44, %43 ]
  %49 = phi ptr [ %56, %54 ], [ %36, %43 ]
  %50 = load double, ptr %48, align 8, !tbaa !16
  %51 = fcmp oeq double %47, %50
  br i1 %51, label %54, label %52

52:                                               ; preds = %46
  %53 = getelementptr inbounds nuw i8, ptr %49, i64 8
  store double %50, ptr %53, align 8, !tbaa !16
  br label %54

54:                                               ; preds = %52, %46
  %55 = phi double [ %47, %46 ], [ %50, %52 ]
  %56 = phi ptr [ %49, %46 ], [ %53, %52 ]
  %57 = getelementptr inbounds nuw i8, ptr %48, i64 8
  %58 = icmp eq ptr %57, %23
  br i1 %58, label %59, label %46, !llvm.loop !19

59:                                               ; preds = %35, %54, %22, %43
  %60 = icmp eq ptr %24, null
  br i1 %60, label %64, label %61

61:                                               ; preds = %59
  %62 = ptrtoint ptr %23 to i64
  %63 = sub i64 %62, %25
  tail call void @_ZdlPvm(ptr noundef nonnull %24, i64 noundef %63) #22
  br label %64

64:                                               ; preds = %59, %61
  ret void

65:                                               ; preds = %27, %34
  %66 = landingpad { ptr, i32 }
          cleanup
  tail call void @_ZdlPvm(ptr noundef nonnull %24, i64 noundef %29) #22
  resume { ptr, i32 } %66
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress uwtable
define dso_local void @_Z20vector_iterator_testPdS_i(ptr noundef %0, ptr noundef %1, i32 %2) local_unnamed_addr #2 personality ptr @__gxx_personality_v0 {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = icmp ugt i64 %6, 9223372036854775800
  br i1 %7, label %8, label %9

8:                                                ; preds = %3
  tail call void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str) #24
  unreachable

9:                                                ; preds = %3
  %10 = icmp eq ptr %1, %0
  br i1 %10, label %11, label %13

11:                                               ; preds = %9
  %12 = getelementptr inbounds nuw i8, ptr null, i64 %6
  br label %22

13:                                               ; preds = %9
  %14 = tail call noalias noundef nonnull ptr @_Znwm(i64 noundef %6) #23
  %15 = getelementptr inbounds nuw i8, ptr %14, i64 %6
  %16 = icmp samesign ugt i64 %6, 8
  br i1 %16, label %17, label %18, !prof !20

17:                                               ; preds = %13
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %14, ptr align 8 %0, i64 %6, i1 false)
  br label %22

18:                                               ; preds = %13
  %19 = icmp eq i64 %6, 8
  br i1 %19, label %20, label %22

20:                                               ; preds = %18
  %21 = load double, ptr %0, align 8, !tbaa !16
  store double %21, ptr %14, align 8, !tbaa !16
  br label %22

22:                                               ; preds = %20, %18, %17, %11
  %23 = phi ptr [ %12, %11 ], [ %15, %17 ], [ %15, %20 ], [ %15, %18 ]
  %24 = phi ptr [ null, %11 ], [ %14, %17 ], [ %14, %20 ], [ %14, %18 ]
  %25 = ptrtoint ptr %24 to i64
  %26 = icmp eq ptr %24, %23
  br i1 %26, label %59, label %27

27:                                               ; preds = %22
  %28 = ptrtoint ptr %23 to i64
  %29 = sub i64 %28, %25
  %30 = ashr exact i64 %29, 3
  %31 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %30, i1 true)
  %32 = shl nuw nsw i64 %31, 1
  %33 = xor i64 %32, 126
  invoke void @_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_(ptr %24, ptr %23, i64 noundef %33, i8 undef)
          to label %34 unwind label %65

34:                                               ; preds = %27
  invoke void @_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_(ptr %24, ptr %23, i8 undef)
          to label %35 unwind label %65

35:                                               ; preds = %34, %39
  %36 = phi ptr [ %37, %39 ], [ %24, %34 ]
  %37 = getelementptr inbounds nuw i8, ptr %36, i64 8
  %38 = icmp eq ptr %37, %23
  br i1 %38, label %59, label %39

39:                                               ; preds = %35
  %40 = load double, ptr %36, align 8, !tbaa !16
  %41 = load double, ptr %37, align 8, !tbaa !16
  %42 = fcmp oeq double %40, %41
  br i1 %42, label %43, label %35, !llvm.loop !21

43:                                               ; preds = %39
  %44 = getelementptr inbounds nuw i8, ptr %36, i64 16
  %45 = icmp eq ptr %44, %23
  br i1 %45, label %59, label %46

46:                                               ; preds = %43, %54
  %47 = phi double [ %55, %54 ], [ %40, %43 ]
  %48 = phi ptr [ %57, %54 ], [ %44, %43 ]
  %49 = phi ptr [ %56, %54 ], [ %36, %43 ]
  %50 = load double, ptr %48, align 8, !tbaa !16
  %51 = fcmp oeq double %47, %50
  br i1 %51, label %54, label %52

52:                                               ; preds = %46
  %53 = getelementptr inbounds nuw i8, ptr %49, i64 8
  store double %50, ptr %53, align 8, !tbaa !16
  br label %54

54:                                               ; preds = %52, %46
  %55 = phi double [ %47, %46 ], [ %50, %52 ]
  %56 = phi ptr [ %49, %46 ], [ %53, %52 ]
  %57 = getelementptr inbounds nuw i8, ptr %48, i64 8
  %58 = icmp eq ptr %57, %23
  br i1 %58, label %59, label %46, !llvm.loop !22

59:                                               ; preds = %35, %54, %43, %22
  %60 = icmp eq ptr %24, null
  br i1 %60, label %64, label %61

61:                                               ; preds = %59
  %62 = ptrtoint ptr %23 to i64
  %63 = sub i64 %62, %25
  tail call void @_ZdlPvm(ptr noundef nonnull %24, i64 noundef %63) #22
  br label %64

64:                                               ; preds = %59, %61
  ret void

65:                                               ; preds = %34, %27
  %66 = landingpad { ptr, i32 }
          cleanup
  %67 = icmp eq ptr %24, null
  br i1 %67, label %69, label %68

68:                                               ; preds = %65
  tail call void @_ZdlPvm(ptr noundef nonnull %24, i64 noundef %29) #22
  br label %69

69:                                               ; preds = %68, %65
  resume { ptr, i32 } %66
}

; Function Attrs: mustprogress uwtable
define dso_local void @_Z10deque_testPdS_i(ptr noundef %0, ptr noundef %1, i32 %2) local_unnamed_addr #2 personality ptr @__gxx_personality_v0 {
  %4 = alloca %"struct.std::_Deque_iterator", align 8
  %5 = alloca %"struct.std::_Deque_iterator", align 8
  %6 = alloca %"struct.std::_Deque_iterator", align 8
  %7 = alloca %"struct.std::_Deque_iterator", align 8
  %8 = alloca %"class.std::deque", align 8
  %9 = alloca double, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #25
  %10 = ptrtoint ptr %1 to i64
  %11 = ptrtoint ptr %0 to i64
  %12 = sub i64 %10, %11
  %13 = ashr exact i64 %12, 3
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #25
  store double 0.000000e+00, ptr %9, align 8, !tbaa !16
  %14 = icmp ugt i64 %13, 1152921504606846975
  br i1 %14, label %15, label %17

15:                                               ; preds = %3
  invoke void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.1) #24
          to label %16 unwind label %228

16:                                               ; preds = %15
  unreachable

17:                                               ; preds = %3
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(80) %8, i8 0, i64 80, i1 false)
  invoke void @_ZNSt11_Deque_baseIdSaIdEE17_M_initialize_mapEm(ptr noundef nonnull align 8 dereferenceable(80) %8, i64 noundef %13)
          to label %18 unwind label %228

18:                                               ; preds = %17
  invoke void @_ZNSt5dequeIdSaIdEE18_M_fill_initializeERKd(ptr noundef nonnull align 8 dereferenceable(80) %8, ptr noundef nonnull align 8 dereferenceable(8) %9)
          to label %21 unwind label %19

19:                                               ; preds = %18
  %20 = landingpad { ptr, i32 }
          cleanup
  call void @_ZNSt11_Deque_baseIdSaIdEED2Ev(ptr noundef nonnull align 8 dereferenceable(80) %8) #25
  br label %230

21:                                               ; preds = %18
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #25
  %22 = getelementptr inbounds nuw i8, ptr %8, i64 16
  %23 = load ptr, ptr %22, align 8, !tbaa !23, !noalias !27
  %24 = getelementptr inbounds nuw i8, ptr %8, i64 24
  %25 = load ptr, ptr %24, align 8, !tbaa !30, !noalias !27
  %26 = getelementptr inbounds nuw i8, ptr %8, i64 32
  %27 = load ptr, ptr %26, align 8, !tbaa !31, !noalias !27
  %28 = getelementptr inbounds nuw i8, ptr %8, i64 40
  %29 = load ptr, ptr %28, align 8, !tbaa !32, !noalias !27
  %30 = icmp eq ptr %1, %0
  br i1 %30, label %85, label %31

31:                                               ; preds = %21, %73
  %32 = phi ptr [ %77, %73 ], [ %23, %21 ]
  %33 = phi ptr [ %74, %73 ], [ %25, %21 ]
  %34 = phi ptr [ %75, %73 ], [ %27, %21 ]
  %35 = phi ptr [ %76, %73 ], [ %29, %21 ]
  %36 = phi ptr [ %43, %73 ], [ %0, %21 ]
  %37 = phi i64 [ %78, %73 ], [ %13, %21 ]
  %38 = ptrtoint ptr %34 to i64
  %39 = ptrtoint ptr %32 to i64
  %40 = sub i64 %38, %39
  %41 = ashr exact i64 %40, 3
  %42 = call i64 @llvm.smin.i64(i64 %41, i64 %37)
  %43 = getelementptr inbounds double, ptr %36, i64 %42
  %44 = icmp sgt i64 %42, 1
  br i1 %44, label %45, label %47, !prof !15

45:                                               ; preds = %31
  %46 = shl nuw nsw i64 %42, 3
  call void @llvm.memmove.p0.p0.i64(ptr align 8 %32, ptr align 8 %36, i64 %46, i1 false), !noalias !33
  br label %51

47:                                               ; preds = %31
  %48 = icmp eq i64 %42, 1
  br i1 %48, label %49, label %51

49:                                               ; preds = %47
  %50 = load double, ptr %36, align 8, !tbaa !16, !noalias !33
  store double %50, ptr %32, align 8, !tbaa !16, !noalias !33
  br label %51

51:                                               ; preds = %49, %47, %45
  %52 = ptrtoint ptr %33 to i64
  %53 = sub i64 %39, %52
  %54 = ashr exact i64 %53, 3
  %55 = add nsw i64 %42, %54
  %56 = icmp sgt i64 %55, -1
  br i1 %56, label %57, label %63

57:                                               ; preds = %51
  %58 = icmp samesign ult i64 %55, 64
  br i1 %58, label %59, label %61

59:                                               ; preds = %57
  %60 = getelementptr inbounds double, ptr %32, i64 %42
  br label %73

61:                                               ; preds = %57
  %62 = lshr i64 %55, 6
  br label %65

63:                                               ; preds = %51
  %64 = ashr i64 %55, 6
  br label %65

65:                                               ; preds = %63, %61
  %66 = phi i64 [ %62, %61 ], [ %64, %63 ]
  %67 = getelementptr inbounds ptr, ptr %35, i64 %66
  %68 = load ptr, ptr %67, align 8, !tbaa !40, !noalias !33
  %69 = getelementptr inbounds nuw i8, ptr %68, i64 512
  %70 = shl nsw i64 %66, 6
  %71 = sub nsw i64 %55, %70
  %72 = getelementptr inbounds double, ptr %68, i64 %71
  br label %73

73:                                               ; preds = %65, %59
  %74 = phi ptr [ %33, %59 ], [ %68, %65 ]
  %75 = phi ptr [ %34, %59 ], [ %69, %65 ]
  %76 = phi ptr [ %35, %59 ], [ %67, %65 ]
  %77 = phi ptr [ %60, %59 ], [ %72, %65 ]
  %78 = sub nsw i64 %37, %42
  %79 = icmp sgt i64 %78, 0
  br i1 %79, label %31, label %80, !llvm.loop !41

80:                                               ; preds = %73
  %81 = load ptr, ptr %22, align 8, !tbaa !23, !noalias !42
  %82 = load ptr, ptr %24, align 8, !tbaa !30, !noalias !42
  %83 = load ptr, ptr %26, align 8, !tbaa !31, !noalias !42
  %84 = load ptr, ptr %28, align 8, !tbaa !32, !noalias !42
  br label %85

85:                                               ; preds = %80, %21
  %86 = phi ptr [ %84, %80 ], [ %29, %21 ]
  %87 = phi ptr [ %83, %80 ], [ %27, %21 ]
  %88 = phi ptr [ %82, %80 ], [ %25, %21 ]
  %89 = phi ptr [ %81, %80 ], [ %23, %21 ]
  %90 = getelementptr inbounds nuw i8, ptr %8, i64 48
  %91 = load ptr, ptr %90, align 8, !tbaa !23, !noalias !45
  %92 = getelementptr inbounds nuw i8, ptr %8, i64 56
  %93 = load ptr, ptr %92, align 8, !tbaa !30, !noalias !45
  %94 = getelementptr inbounds nuw i8, ptr %8, i64 64
  %95 = load ptr, ptr %94, align 8, !tbaa !31, !noalias !45
  %96 = getelementptr inbounds nuw i8, ptr %8, i64 72
  %97 = load ptr, ptr %96, align 8, !tbaa !32, !noalias !45
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  %98 = icmp eq ptr %89, %91
  br i1 %98, label %99, label %100

99:                                               ; preds = %85
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  br label %207

100:                                              ; preds = %85
  store ptr %89, ptr %4, align 8, !tbaa !23
  %101 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr %88, ptr %101, align 8, !tbaa !30
  %102 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store ptr %87, ptr %102, align 8, !tbaa !31
  %103 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store ptr %86, ptr %103, align 8, !tbaa !32
  store ptr %91, ptr %5, align 8, !tbaa !23
  %104 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store ptr %93, ptr %104, align 8, !tbaa !30
  %105 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store ptr %95, ptr %105, align 8, !tbaa !31
  %106 = getelementptr inbounds nuw i8, ptr %5, i64 24
  store ptr %97, ptr %106, align 8, !tbaa !32
  %107 = ptrtoint ptr %97 to i64
  %108 = ptrtoint ptr %86 to i64
  %109 = sub i64 %107, %108
  %110 = ashr exact i64 %109, 3
  %111 = icmp ne ptr %97, null
  %112 = sext i1 %111 to i64
  %113 = add nsw i64 %110, %112
  %114 = shl nsw i64 %113, 6
  %115 = ptrtoint ptr %91 to i64
  %116 = ptrtoint ptr %93 to i64
  %117 = sub i64 %115, %116
  %118 = ashr exact i64 %117, 3
  %119 = ptrtoint ptr %87 to i64
  %120 = ptrtoint ptr %89 to i64
  %121 = sub i64 %119, %120
  %122 = ashr exact i64 %121, 3
  %123 = add nsw i64 %118, %122
  %124 = add i64 %123, %114
  %125 = call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %124, i1 false)
  %126 = shl nuw nsw i64 %125, 1
  %127 = sub nsw i64 126, %126
  invoke void @_ZSt16__introsort_loopISt15_Deque_iteratorIdRdPdElN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S7_T0_T1_(ptr dead_on_return noundef nonnull %4, ptr dead_on_return noundef nonnull %5, i64 noundef %127, i8 undef)
          to label %128 unwind label %232

128:                                              ; preds = %100
  store ptr %89, ptr %6, align 8, !tbaa !23
  %129 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store ptr %88, ptr %129, align 8, !tbaa !30
  %130 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store ptr %87, ptr %130, align 8, !tbaa !31
  %131 = getelementptr inbounds nuw i8, ptr %6, i64 24
  store ptr %86, ptr %131, align 8, !tbaa !32
  store ptr %91, ptr %7, align 8, !tbaa !23
  %132 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store ptr %93, ptr %132, align 8, !tbaa !30
  %133 = getelementptr inbounds nuw i8, ptr %7, i64 16
  store ptr %95, ptr %133, align 8, !tbaa !31
  %134 = getelementptr inbounds nuw i8, ptr %7, i64 24
  store ptr %97, ptr %134, align 8, !tbaa !32
  invoke void @_ZSt22__final_insertion_sortISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S7_T0_(ptr dead_on_return noundef nonnull %6, ptr dead_on_return noundef nonnull %7, i8 undef)
          to label %135 unwind label %232

135:                                              ; preds = %128
  %136 = load ptr, ptr %22, align 8, !tbaa !23, !noalias !48
  %137 = load ptr, ptr %26, align 8, !tbaa !31, !noalias !48
  %138 = load ptr, ptr %28, align 8, !tbaa !32, !noalias !48
  %139 = load ptr, ptr %90, align 8, !tbaa !23, !noalias !51
  %140 = load ptr, ptr %96, align 8, !tbaa !32, !noalias !51
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  %141 = icmp eq ptr %136, %139
  br i1 %141, label %207, label %142

142:                                              ; preds = %135, %157
  %143 = phi ptr [ %153, %157 ], [ %136, %135 ]
  %144 = phi ptr [ %154, %157 ], [ %137, %135 ]
  %145 = phi ptr [ %155, %157 ], [ %138, %135 ]
  %146 = getelementptr inbounds nuw i8, ptr %143, i64 8
  %147 = icmp eq ptr %146, %144
  br i1 %147, label %148, label %152

148:                                              ; preds = %142
  %149 = getelementptr inbounds nuw i8, ptr %145, i64 8
  %150 = load ptr, ptr %149, align 8, !tbaa !40, !noalias !54
  %151 = getelementptr inbounds nuw i8, ptr %150, i64 512
  br label %152

152:                                              ; preds = %148, %142
  %153 = phi ptr [ %150, %148 ], [ %146, %142 ]
  %154 = phi ptr [ %151, %148 ], [ %144, %142 ]
  %155 = phi ptr [ %149, %148 ], [ %145, %142 ]
  %156 = icmp eq ptr %153, %139
  br i1 %156, label %207, label %157

157:                                              ; preds = %152
  %158 = load double, ptr %143, align 8, !tbaa !16, !noalias !54
  %159 = load double, ptr %153, align 8, !tbaa !16, !noalias !54
  %160 = fcmp oeq double %158, %159
  br i1 %160, label %161, label %142, !llvm.loop !61

161:                                              ; preds = %157
  br i1 %147, label %162, label %166

162:                                              ; preds = %161
  %163 = getelementptr inbounds nuw i8, ptr %145, i64 8
  %164 = load ptr, ptr %163, align 8, !tbaa !40, !noalias !62
  %165 = getelementptr inbounds nuw i8, ptr %164, i64 512
  br label %166

166:                                              ; preds = %162, %161
  %167 = phi ptr [ %145, %161 ], [ %163, %162 ]
  %168 = phi ptr [ %146, %161 ], [ %164, %162 ]
  %169 = phi ptr [ %144, %161 ], [ %165, %162 ]
  br label %170

170:                                              ; preds = %166, %203
  %171 = phi double [ %158, %166 ], [ %194, %203 ]
  %172 = phi ptr [ %167, %166 ], [ %190, %203 ]
  %173 = phi ptr [ %168, %166 ], [ %189, %203 ]
  %174 = phi ptr [ %169, %166 ], [ %191, %203 ]
  %175 = phi ptr [ %145, %166 ], [ %204, %203 ]
  %176 = phi ptr [ %144, %166 ], [ %205, %203 ]
  %177 = phi ptr [ %143, %166 ], [ %206, %203 ]
  br label %178

178:                                              ; preds = %170, %193
  %179 = phi ptr [ %190, %193 ], [ %172, %170 ]
  %180 = phi ptr [ %189, %193 ], [ %173, %170 ]
  %181 = phi ptr [ %191, %193 ], [ %174, %170 ]
  %182 = getelementptr inbounds nuw i8, ptr %180, i64 8
  %183 = icmp eq ptr %182, %181
  br i1 %183, label %184, label %188

184:                                              ; preds = %178
  %185 = getelementptr inbounds nuw i8, ptr %179, i64 8
  %186 = load ptr, ptr %185, align 8, !tbaa !40, !noalias !62
  %187 = getelementptr inbounds nuw i8, ptr %186, i64 512
  br label %188

188:                                              ; preds = %184, %178
  %189 = phi ptr [ %182, %178 ], [ %186, %184 ]
  %190 = phi ptr [ %179, %178 ], [ %185, %184 ]
  %191 = phi ptr [ %181, %178 ], [ %187, %184 ]
  %192 = icmp eq ptr %189, %139
  br i1 %192, label %207, label %193

193:                                              ; preds = %188
  %194 = load double, ptr %189, align 8, !tbaa !16, !noalias !62
  %195 = fcmp oeq double %171, %194
  br i1 %195, label %178, label %196, !llvm.loop !63

196:                                              ; preds = %193
  %197 = getelementptr inbounds nuw i8, ptr %177, i64 8
  %198 = icmp eq ptr %197, %176
  br i1 %198, label %199, label %203

199:                                              ; preds = %196
  %200 = getelementptr inbounds nuw i8, ptr %175, i64 8
  %201 = load ptr, ptr %200, align 8, !tbaa !40, !noalias !62
  %202 = getelementptr inbounds nuw i8, ptr %201, i64 512
  br label %203

203:                                              ; preds = %199, %196
  %204 = phi ptr [ %200, %199 ], [ %175, %196 ]
  %205 = phi ptr [ %202, %199 ], [ %176, %196 ]
  %206 = phi ptr [ %201, %199 ], [ %197, %196 ]
  store double %194, ptr %206, align 8, !tbaa !16, !noalias !62
  br label %170, !llvm.loop !63

207:                                              ; preds = %152, %188, %99, %135
  %208 = phi ptr [ %86, %99 ], [ %138, %135 ], [ %138, %188 ], [ %138, %152 ]
  %209 = phi ptr [ %97, %99 ], [ %140, %135 ], [ %140, %188 ], [ %140, %152 ]
  %210 = load ptr, ptr %8, align 8, !tbaa !64
  %211 = icmp eq ptr %210, null
  br i1 %211, label %227, label %212

212:                                              ; preds = %207
  %213 = getelementptr inbounds nuw i8, ptr %209, i64 8
  %214 = icmp ult ptr %208, %213
  br i1 %214, label %215, label %222

215:                                              ; preds = %212, %215
  %216 = phi ptr [ %218, %215 ], [ %208, %212 ]
  %217 = load ptr, ptr %216, align 8, !tbaa !40
  call void @_ZdlPvm(ptr noundef %217, i64 noundef 512) #22
  %218 = getelementptr inbounds nuw i8, ptr %216, i64 8
  %219 = icmp ult ptr %216, %209
  br i1 %219, label %215, label %220, !llvm.loop !67

220:                                              ; preds = %215
  %221 = load ptr, ptr %8, align 8, !tbaa !64
  br label %222

222:                                              ; preds = %220, %212
  %223 = phi ptr [ %221, %220 ], [ %210, %212 ]
  %224 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %225 = load i64, ptr %224, align 8, !tbaa !68
  %226 = shl i64 %225, 3
  call void @_ZdlPvm(ptr noundef %223, i64 noundef %226) #22
  br label %227

227:                                              ; preds = %207, %222
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #25
  ret void

228:                                              ; preds = %17, %15
  %229 = landingpad { ptr, i32 }
          cleanup
  br label %230

230:                                              ; preds = %19, %228
  %231 = phi { ptr, i32 } [ %229, %228 ], [ %20, %19 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #25
  br label %234

232:                                              ; preds = %128, %100
  %233 = landingpad { ptr, i32 }
          cleanup
  call void @_ZNSt5dequeIdSaIdEED2Ev(ptr noundef nonnull align 8 dereferenceable(80) %8) #25
  br label %234

234:                                              ; preds = %232, %230
  %235 = phi { ptr, i32 } [ %233, %232 ], [ %231, %230 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #25
  resume { ptr, i32 } %235
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZNSt5dequeIdSaIdEED2Ev(ptr noundef nonnull align 8 dereferenceable(80) %0) unnamed_addr #0 comdat personality ptr @__gxx_personality_v0 {
  %2 = load ptr, ptr %0, align 8, !tbaa !64
  %3 = icmp eq ptr %2, null
  br i1 %3, label %23, label %4

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %7 = load ptr, ptr %6, align 8, !tbaa !69
  %8 = load ptr, ptr %5, align 8, !tbaa !70
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %10 = icmp ult ptr %7, %9
  br i1 %10, label %11, label %18

11:                                               ; preds = %4, %11
  %12 = phi ptr [ %14, %11 ], [ %7, %4 ]
  %13 = load ptr, ptr %12, align 8, !tbaa !40
  tail call void @_ZdlPvm(ptr noundef %13, i64 noundef 512) #22
  %14 = getelementptr inbounds nuw i8, ptr %12, i64 8
  %15 = icmp ult ptr %12, %8
  br i1 %15, label %11, label %16, !llvm.loop !67

16:                                               ; preds = %11
  %17 = load ptr, ptr %0, align 8, !tbaa !64
  br label %18

18:                                               ; preds = %16, %4
  %19 = phi ptr [ %17, %16 ], [ %2, %4 ]
  %20 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %21 = load i64, ptr %20, align 8, !tbaa !68
  %22 = shl i64 %21, 3
  tail call void @_ZdlPvm(ptr noundef %19, i64 noundef %22) #22
  br label %23

23:                                               ; preds = %1, %18
  ret void
}

; Function Attrs: mustprogress uwtable
define dso_local void @_Z9list_testPdS_i(ptr noundef readonly captures(address) %0, ptr noundef readnone captures(address) %1, i32 %2) local_unnamed_addr #2 personality ptr @__gxx_personality_v0 {
  %4 = alloca %"class.std::__cxx11::list", align 8
  %5 = alloca %"class.std::__cxx11::list", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #25
  %6 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store ptr %5, ptr %6, align 8, !tbaa !71
  store ptr %5, ptr %5, align 8, !tbaa !74
  %7 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store i64 0, ptr %7, align 8, !tbaa !75
  %8 = icmp eq ptr %0, %1
  br i1 %8, label %27, label %9

9:                                                ; preds = %3, %12
  %10 = phi ptr [ %17, %12 ], [ %0, %3 ]
  %11 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #23
          to label %12 unwind label %19

12:                                               ; preds = %9
  %13 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %14 = load double, ptr %10, align 8, !tbaa !16
  store double %14, ptr %13, align 8, !tbaa !16
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16) %11, ptr noundef nonnull align 8 dereferenceable(24) %5) #25
  %15 = load i64, ptr %7, align 8, !tbaa !77
  %16 = add i64 %15, 1
  store i64 %16, ptr %7, align 8, !tbaa !77
  %17 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %18 = icmp eq ptr %17, %1
  br i1 %18, label %27, label %9, !llvm.loop !80

19:                                               ; preds = %9
  %20 = landingpad { ptr, i32 }
          cleanup
  %21 = load ptr, ptr %5, align 8, !tbaa !74
  %22 = icmp eq ptr %21, %5
  br i1 %22, label %84, label %23

23:                                               ; preds = %19, %23
  %24 = phi ptr [ %25, %23 ], [ %21, %19 ]
  %25 = load ptr, ptr %24, align 8, !tbaa !74
  call void @_ZdlPvm(ptr noundef nonnull %24, i64 noundef 24) #22
  %26 = icmp eq ptr %25, %5
  br i1 %26, label %84, label %23, !llvm.loop !81

27:                                               ; preds = %12, %3
  invoke void @_ZNSt7__cxx114listIdSaIdEE4sortEv(ptr noundef nonnull align 8 dereferenceable(24) %5)
          to label %28 unwind label %76

28:                                               ; preds = %27
  %29 = load ptr, ptr %5, align 8, !tbaa !74
  %30 = icmp eq ptr %29, %5
  br i1 %30, label %68, label %31

31:                                               ; preds = %28
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #25
  %32 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr %4, ptr %32, align 8, !tbaa !71
  store ptr %4, ptr %4, align 8, !tbaa !74
  %33 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i64 0, ptr %33, align 8, !tbaa !75
  %34 = load ptr, ptr %29, align 8, !tbaa !74
  %35 = icmp eq ptr %34, %5
  br i1 %35, label %66, label %36

36:                                               ; preds = %31, %55
  %37 = phi ptr [ %57, %55 ], [ %34, %31 ]
  %38 = phi ptr [ %56, %55 ], [ %29, %31 ]
  %39 = getelementptr inbounds nuw i8, ptr %38, i64 16
  %40 = load double, ptr %39, align 8, !tbaa !16
  %41 = getelementptr inbounds nuw i8, ptr %37, i64 16
  %42 = load double, ptr %41, align 8, !tbaa !16
  %43 = fcmp oeq double %40, %42
  br i1 %43, label %44, label %55

44:                                               ; preds = %36
  %45 = load ptr, ptr %4, align 8, !tbaa !74
  %46 = load ptr, ptr %37, align 8, !tbaa !74
  %47 = icmp eq ptr %45, %37
  %48 = icmp eq ptr %45, %46
  %49 = select i1 %47, i1 true, i1 %48
  br i1 %49, label %55, label %50

50:                                               ; preds = %44
  call void @_ZNSt8__detail15_List_node_base11_M_transferEPS0_S1_(ptr noundef nonnull align 8 dereferenceable(16) %45, ptr noundef nonnull %37, ptr noundef %46) #25
  %51 = load i64, ptr %33, align 8, !tbaa !77
  %52 = add i64 %51, 1
  store i64 %52, ptr %33, align 8, !tbaa !77
  %53 = load i64, ptr %7, align 8, !tbaa !77
  %54 = add i64 %53, -1
  store i64 %54, ptr %7, align 8, !tbaa !77
  br label %55

55:                                               ; preds = %50, %44, %36
  %56 = phi ptr [ %38, %44 ], [ %38, %50 ], [ %37, %36 ]
  %57 = load ptr, ptr %56, align 8, !tbaa !74
  %58 = icmp eq ptr %57, %5
  br i1 %58, label %59, label %36, !llvm.loop !82

59:                                               ; preds = %55
  %60 = load ptr, ptr %4, align 8, !tbaa !74
  %61 = icmp eq ptr %60, %4
  br i1 %61, label %66, label %62

62:                                               ; preds = %59, %62
  %63 = phi ptr [ %64, %62 ], [ %60, %59 ]
  %64 = load ptr, ptr %63, align 8, !tbaa !74
  call void @_ZdlPvm(ptr noundef nonnull %63, i64 noundef 24) #22
  %65 = icmp eq ptr %64, %4
  br i1 %65, label %66, label %62, !llvm.loop !81

66:                                               ; preds = %62, %59, %31
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #25
  %67 = load ptr, ptr %5, align 8, !tbaa !74
  br label %68

68:                                               ; preds = %28, %66
  %69 = phi ptr [ %29, %28 ], [ %67, %66 ]
  %70 = icmp eq ptr %69, %5
  br i1 %70, label %75, label %71

71:                                               ; preds = %68, %71
  %72 = phi ptr [ %73, %71 ], [ %69, %68 ]
  %73 = load ptr, ptr %72, align 8, !tbaa !74
  call void @_ZdlPvm(ptr noundef nonnull %72, i64 noundef 24) #22
  %74 = icmp eq ptr %73, %5
  br i1 %74, label %75, label %71, !llvm.loop !81

75:                                               ; preds = %71, %68
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #25
  ret void

76:                                               ; preds = %27
  %77 = landingpad { ptr, i32 }
          cleanup
  %78 = load ptr, ptr %5, align 8, !tbaa !74
  %79 = icmp eq ptr %78, %5
  br i1 %79, label %84, label %80

80:                                               ; preds = %76, %80
  %81 = phi ptr [ %82, %80 ], [ %78, %76 ]
  %82 = load ptr, ptr %81, align 8, !tbaa !74
  call void @_ZdlPvm(ptr noundef nonnull %81, i64 noundef 24) #22
  %83 = icmp eq ptr %82, %5
  br i1 %83, label %84, label %80, !llvm.loop !81

84:                                               ; preds = %23, %80, %76, %19
  %85 = phi { ptr, i32 } [ %20, %19 ], [ %77, %76 ], [ %77, %80 ], [ %20, %23 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #25
  resume { ptr, i32 } %85
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNSt7__cxx114listIdSaIdEE4sortEv(ptr noundef nonnull align 8 dereferenceable(24) %0) local_unnamed_addr #2 comdat personality ptr @__gxx_personality_v0 {
  %2 = alloca %"struct.std::__detail::_Scratch_list", align 8
  %3 = alloca [64 x %"struct.std::__detail::_Scratch_list"], align 8
  %4 = load ptr, ptr %0, align 8, !tbaa !74
  %5 = icmp eq ptr %4, %0
  br i1 %5, label %220, label %6

6:                                                ; preds = %1
  %7 = load ptr, ptr %4, align 8, !tbaa !74
  %8 = icmp eq ptr %7, %0
  br i1 %8, label %220, label %9

9:                                                ; preds = %6
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #25
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store ptr %2, ptr %10, align 8, !tbaa !71
  store ptr %2, ptr %2, align 8, !tbaa !74
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #25
  %11 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store ptr %3, ptr %11, align 8, !tbaa !71
  store ptr %3, ptr %3, align 8, !tbaa !74
  %12 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %13 = getelementptr inbounds nuw i8, ptr %3, i64 24
  store ptr %12, ptr %13, align 8, !tbaa !71
  store ptr %12, ptr %12, align 8, !tbaa !74
  %14 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %15 = getelementptr inbounds nuw i8, ptr %3, i64 40
  store ptr %14, ptr %15, align 8, !tbaa !71
  store ptr %14, ptr %14, align 8, !tbaa !74
  %16 = getelementptr inbounds nuw i8, ptr %3, i64 48
  %17 = getelementptr inbounds nuw i8, ptr %3, i64 56
  store ptr %16, ptr %17, align 8, !tbaa !71
  store ptr %16, ptr %16, align 8, !tbaa !74
  %18 = getelementptr inbounds nuw i8, ptr %3, i64 64
  %19 = getelementptr inbounds nuw i8, ptr %3, i64 72
  store ptr %18, ptr %19, align 8, !tbaa !71
  store ptr %18, ptr %18, align 8, !tbaa !74
  %20 = getelementptr inbounds nuw i8, ptr %3, i64 80
  %21 = getelementptr inbounds nuw i8, ptr %3, i64 88
  store ptr %20, ptr %21, align 8, !tbaa !71
  store ptr %20, ptr %20, align 8, !tbaa !74
  %22 = getelementptr inbounds nuw i8, ptr %3, i64 96
  %23 = getelementptr inbounds nuw i8, ptr %3, i64 104
  store ptr %22, ptr %23, align 8, !tbaa !71
  store ptr %22, ptr %22, align 8, !tbaa !74
  %24 = getelementptr inbounds nuw i8, ptr %3, i64 112
  %25 = getelementptr inbounds nuw i8, ptr %3, i64 120
  store ptr %24, ptr %25, align 8, !tbaa !71
  store ptr %24, ptr %24, align 8, !tbaa !74
  %26 = getelementptr inbounds nuw i8, ptr %3, i64 128
  %27 = getelementptr inbounds nuw i8, ptr %3, i64 136
  store ptr %26, ptr %27, align 8, !tbaa !71
  store ptr %26, ptr %26, align 8, !tbaa !74
  %28 = getelementptr inbounds nuw i8, ptr %3, i64 144
  %29 = getelementptr inbounds nuw i8, ptr %3, i64 152
  store ptr %28, ptr %29, align 8, !tbaa !71
  store ptr %28, ptr %28, align 8, !tbaa !74
  %30 = getelementptr inbounds nuw i8, ptr %3, i64 160
  %31 = getelementptr inbounds nuw i8, ptr %3, i64 168
  store ptr %30, ptr %31, align 8, !tbaa !71
  store ptr %30, ptr %30, align 8, !tbaa !74
  %32 = getelementptr inbounds nuw i8, ptr %3, i64 176
  %33 = getelementptr inbounds nuw i8, ptr %3, i64 184
  store ptr %32, ptr %33, align 8, !tbaa !71
  store ptr %32, ptr %32, align 8, !tbaa !74
  %34 = getelementptr inbounds nuw i8, ptr %3, i64 192
  %35 = getelementptr inbounds nuw i8, ptr %3, i64 200
  store ptr %34, ptr %35, align 8, !tbaa !71
  store ptr %34, ptr %34, align 8, !tbaa !74
  %36 = getelementptr inbounds nuw i8, ptr %3, i64 208
  %37 = getelementptr inbounds nuw i8, ptr %3, i64 216
  store ptr %36, ptr %37, align 8, !tbaa !71
  store ptr %36, ptr %36, align 8, !tbaa !74
  %38 = getelementptr inbounds nuw i8, ptr %3, i64 224
  %39 = getelementptr inbounds nuw i8, ptr %3, i64 232
  store ptr %38, ptr %39, align 8, !tbaa !71
  store ptr %38, ptr %38, align 8, !tbaa !74
  %40 = getelementptr inbounds nuw i8, ptr %3, i64 240
  %41 = getelementptr inbounds nuw i8, ptr %3, i64 248
  store ptr %40, ptr %41, align 8, !tbaa !71
  store ptr %40, ptr %40, align 8, !tbaa !74
  %42 = getelementptr inbounds nuw i8, ptr %3, i64 256
  %43 = getelementptr inbounds nuw i8, ptr %3, i64 264
  store ptr %42, ptr %43, align 8, !tbaa !71
  store ptr %42, ptr %42, align 8, !tbaa !74
  %44 = getelementptr inbounds nuw i8, ptr %3, i64 272
  %45 = getelementptr inbounds nuw i8, ptr %3, i64 280
  store ptr %44, ptr %45, align 8, !tbaa !71
  store ptr %44, ptr %44, align 8, !tbaa !74
  %46 = getelementptr inbounds nuw i8, ptr %3, i64 288
  %47 = getelementptr inbounds nuw i8, ptr %3, i64 296
  store ptr %46, ptr %47, align 8, !tbaa !71
  store ptr %46, ptr %46, align 8, !tbaa !74
  %48 = getelementptr inbounds nuw i8, ptr %3, i64 304
  %49 = getelementptr inbounds nuw i8, ptr %3, i64 312
  store ptr %48, ptr %49, align 8, !tbaa !71
  store ptr %48, ptr %48, align 8, !tbaa !74
  %50 = getelementptr inbounds nuw i8, ptr %3, i64 320
  %51 = getelementptr inbounds nuw i8, ptr %3, i64 328
  store ptr %50, ptr %51, align 8, !tbaa !71
  store ptr %50, ptr %50, align 8, !tbaa !74
  %52 = getelementptr inbounds nuw i8, ptr %3, i64 336
  %53 = getelementptr inbounds nuw i8, ptr %3, i64 344
  store ptr %52, ptr %53, align 8, !tbaa !71
  store ptr %52, ptr %52, align 8, !tbaa !74
  %54 = getelementptr inbounds nuw i8, ptr %3, i64 352
  %55 = getelementptr inbounds nuw i8, ptr %3, i64 360
  store ptr %54, ptr %55, align 8, !tbaa !71
  store ptr %54, ptr %54, align 8, !tbaa !74
  %56 = getelementptr inbounds nuw i8, ptr %3, i64 368
  %57 = getelementptr inbounds nuw i8, ptr %3, i64 376
  store ptr %56, ptr %57, align 8, !tbaa !71
  store ptr %56, ptr %56, align 8, !tbaa !74
  %58 = getelementptr inbounds nuw i8, ptr %3, i64 384
  %59 = getelementptr inbounds nuw i8, ptr %3, i64 392
  store ptr %58, ptr %59, align 8, !tbaa !71
  store ptr %58, ptr %58, align 8, !tbaa !74
  %60 = getelementptr inbounds nuw i8, ptr %3, i64 400
  %61 = getelementptr inbounds nuw i8, ptr %3, i64 408
  store ptr %60, ptr %61, align 8, !tbaa !71
  store ptr %60, ptr %60, align 8, !tbaa !74
  %62 = getelementptr inbounds nuw i8, ptr %3, i64 416
  %63 = getelementptr inbounds nuw i8, ptr %3, i64 424
  store ptr %62, ptr %63, align 8, !tbaa !71
  store ptr %62, ptr %62, align 8, !tbaa !74
  %64 = getelementptr inbounds nuw i8, ptr %3, i64 432
  %65 = getelementptr inbounds nuw i8, ptr %3, i64 440
  store ptr %64, ptr %65, align 8, !tbaa !71
  store ptr %64, ptr %64, align 8, !tbaa !74
  %66 = getelementptr inbounds nuw i8, ptr %3, i64 448
  %67 = getelementptr inbounds nuw i8, ptr %3, i64 456
  store ptr %66, ptr %67, align 8, !tbaa !71
  store ptr %66, ptr %66, align 8, !tbaa !74
  %68 = getelementptr inbounds nuw i8, ptr %3, i64 464
  %69 = getelementptr inbounds nuw i8, ptr %3, i64 472
  store ptr %68, ptr %69, align 8, !tbaa !71
  store ptr %68, ptr %68, align 8, !tbaa !74
  %70 = getelementptr inbounds nuw i8, ptr %3, i64 480
  %71 = getelementptr inbounds nuw i8, ptr %3, i64 488
  store ptr %70, ptr %71, align 8, !tbaa !71
  store ptr %70, ptr %70, align 8, !tbaa !74
  %72 = getelementptr inbounds nuw i8, ptr %3, i64 496
  %73 = getelementptr inbounds nuw i8, ptr %3, i64 504
  store ptr %72, ptr %73, align 8, !tbaa !71
  store ptr %72, ptr %72, align 8, !tbaa !74
  %74 = getelementptr inbounds nuw i8, ptr %3, i64 512
  %75 = getelementptr inbounds nuw i8, ptr %3, i64 520
  store ptr %74, ptr %75, align 8, !tbaa !71
  store ptr %74, ptr %74, align 8, !tbaa !74
  %76 = getelementptr inbounds nuw i8, ptr %3, i64 528
  %77 = getelementptr inbounds nuw i8, ptr %3, i64 536
  store ptr %76, ptr %77, align 8, !tbaa !71
  store ptr %76, ptr %76, align 8, !tbaa !74
  %78 = getelementptr inbounds nuw i8, ptr %3, i64 544
  %79 = getelementptr inbounds nuw i8, ptr %3, i64 552
  store ptr %78, ptr %79, align 8, !tbaa !71
  store ptr %78, ptr %78, align 8, !tbaa !74
  %80 = getelementptr inbounds nuw i8, ptr %3, i64 560
  %81 = getelementptr inbounds nuw i8, ptr %3, i64 568
  store ptr %80, ptr %81, align 8, !tbaa !71
  store ptr %80, ptr %80, align 8, !tbaa !74
  %82 = getelementptr inbounds nuw i8, ptr %3, i64 576
  %83 = getelementptr inbounds nuw i8, ptr %3, i64 584
  store ptr %82, ptr %83, align 8, !tbaa !71
  store ptr %82, ptr %82, align 8, !tbaa !74
  %84 = getelementptr inbounds nuw i8, ptr %3, i64 592
  %85 = getelementptr inbounds nuw i8, ptr %3, i64 600
  store ptr %84, ptr %85, align 8, !tbaa !71
  store ptr %84, ptr %84, align 8, !tbaa !74
  %86 = getelementptr inbounds nuw i8, ptr %3, i64 608
  %87 = getelementptr inbounds nuw i8, ptr %3, i64 616
  store ptr %86, ptr %87, align 8, !tbaa !71
  store ptr %86, ptr %86, align 8, !tbaa !74
  %88 = getelementptr inbounds nuw i8, ptr %3, i64 624
  %89 = getelementptr inbounds nuw i8, ptr %3, i64 632
  store ptr %88, ptr %89, align 8, !tbaa !71
  store ptr %88, ptr %88, align 8, !tbaa !74
  %90 = getelementptr inbounds nuw i8, ptr %3, i64 640
  %91 = getelementptr inbounds nuw i8, ptr %3, i64 648
  store ptr %90, ptr %91, align 8, !tbaa !71
  store ptr %90, ptr %90, align 8, !tbaa !74
  %92 = getelementptr inbounds nuw i8, ptr %3, i64 656
  %93 = getelementptr inbounds nuw i8, ptr %3, i64 664
  store ptr %92, ptr %93, align 8, !tbaa !71
  store ptr %92, ptr %92, align 8, !tbaa !74
  %94 = getelementptr inbounds nuw i8, ptr %3, i64 672
  %95 = getelementptr inbounds nuw i8, ptr %3, i64 680
  store ptr %94, ptr %95, align 8, !tbaa !71
  store ptr %94, ptr %94, align 8, !tbaa !74
  %96 = getelementptr inbounds nuw i8, ptr %3, i64 688
  %97 = getelementptr inbounds nuw i8, ptr %3, i64 696
  store ptr %96, ptr %97, align 8, !tbaa !71
  store ptr %96, ptr %96, align 8, !tbaa !74
  %98 = getelementptr inbounds nuw i8, ptr %3, i64 704
  %99 = getelementptr inbounds nuw i8, ptr %3, i64 712
  store ptr %98, ptr %99, align 8, !tbaa !71
  store ptr %98, ptr %98, align 8, !tbaa !74
  %100 = getelementptr inbounds nuw i8, ptr %3, i64 720
  %101 = getelementptr inbounds nuw i8, ptr %3, i64 728
  store ptr %100, ptr %101, align 8, !tbaa !71
  store ptr %100, ptr %100, align 8, !tbaa !74
  %102 = getelementptr inbounds nuw i8, ptr %3, i64 736
  %103 = getelementptr inbounds nuw i8, ptr %3, i64 744
  store ptr %102, ptr %103, align 8, !tbaa !71
  store ptr %102, ptr %102, align 8, !tbaa !74
  %104 = getelementptr inbounds nuw i8, ptr %3, i64 752
  %105 = getelementptr inbounds nuw i8, ptr %3, i64 760
  store ptr %104, ptr %105, align 8, !tbaa !71
  store ptr %104, ptr %104, align 8, !tbaa !74
  %106 = getelementptr inbounds nuw i8, ptr %3, i64 768
  %107 = getelementptr inbounds nuw i8, ptr %3, i64 776
  store ptr %106, ptr %107, align 8, !tbaa !71
  store ptr %106, ptr %106, align 8, !tbaa !74
  %108 = getelementptr inbounds nuw i8, ptr %3, i64 784
  %109 = getelementptr inbounds nuw i8, ptr %3, i64 792
  store ptr %108, ptr %109, align 8, !tbaa !71
  store ptr %108, ptr %108, align 8, !tbaa !74
  %110 = getelementptr inbounds nuw i8, ptr %3, i64 800
  %111 = getelementptr inbounds nuw i8, ptr %3, i64 808
  store ptr %110, ptr %111, align 8, !tbaa !71
  store ptr %110, ptr %110, align 8, !tbaa !74
  %112 = getelementptr inbounds nuw i8, ptr %3, i64 816
  %113 = getelementptr inbounds nuw i8, ptr %3, i64 824
  store ptr %112, ptr %113, align 8, !tbaa !71
  store ptr %112, ptr %112, align 8, !tbaa !74
  %114 = getelementptr inbounds nuw i8, ptr %3, i64 832
  %115 = getelementptr inbounds nuw i8, ptr %3, i64 840
  store ptr %114, ptr %115, align 8, !tbaa !71
  store ptr %114, ptr %114, align 8, !tbaa !74
  %116 = getelementptr inbounds nuw i8, ptr %3, i64 848
  %117 = getelementptr inbounds nuw i8, ptr %3, i64 856
  store ptr %116, ptr %117, align 8, !tbaa !71
  store ptr %116, ptr %116, align 8, !tbaa !74
  %118 = getelementptr inbounds nuw i8, ptr %3, i64 864
  %119 = getelementptr inbounds nuw i8, ptr %3, i64 872
  store ptr %118, ptr %119, align 8, !tbaa !71
  store ptr %118, ptr %118, align 8, !tbaa !74
  %120 = getelementptr inbounds nuw i8, ptr %3, i64 880
  %121 = getelementptr inbounds nuw i8, ptr %3, i64 888
  store ptr %120, ptr %121, align 8, !tbaa !71
  store ptr %120, ptr %120, align 8, !tbaa !74
  %122 = getelementptr inbounds nuw i8, ptr %3, i64 896
  %123 = getelementptr inbounds nuw i8, ptr %3, i64 904
  store ptr %122, ptr %123, align 8, !tbaa !71
  store ptr %122, ptr %122, align 8, !tbaa !74
  %124 = getelementptr inbounds nuw i8, ptr %3, i64 912
  %125 = getelementptr inbounds nuw i8, ptr %3, i64 920
  store ptr %124, ptr %125, align 8, !tbaa !71
  store ptr %124, ptr %124, align 8, !tbaa !74
  %126 = getelementptr inbounds nuw i8, ptr %3, i64 928
  %127 = getelementptr inbounds nuw i8, ptr %3, i64 936
  store ptr %126, ptr %127, align 8, !tbaa !71
  store ptr %126, ptr %126, align 8, !tbaa !74
  %128 = getelementptr inbounds nuw i8, ptr %3, i64 944
  %129 = getelementptr inbounds nuw i8, ptr %3, i64 952
  store ptr %128, ptr %129, align 8, !tbaa !71
  store ptr %128, ptr %128, align 8, !tbaa !74
  %130 = getelementptr inbounds nuw i8, ptr %3, i64 960
  %131 = getelementptr inbounds nuw i8, ptr %3, i64 968
  store ptr %130, ptr %131, align 8, !tbaa !71
  store ptr %130, ptr %130, align 8, !tbaa !74
  %132 = getelementptr inbounds nuw i8, ptr %3, i64 976
  %133 = getelementptr inbounds nuw i8, ptr %3, i64 984
  store ptr %132, ptr %133, align 8, !tbaa !71
  store ptr %132, ptr %132, align 8, !tbaa !74
  %134 = getelementptr inbounds nuw i8, ptr %3, i64 992
  %135 = getelementptr inbounds nuw i8, ptr %3, i64 1000
  store ptr %134, ptr %135, align 8, !tbaa !71
  store ptr %134, ptr %134, align 8, !tbaa !74
  %136 = getelementptr inbounds nuw i8, ptr %3, i64 1008
  %137 = getelementptr inbounds nuw i8, ptr %3, i64 1016
  store ptr %136, ptr %137, align 8, !tbaa !71
  store ptr %136, ptr %136, align 8, !tbaa !74
  %138 = load ptr, ptr %0, align 8, !tbaa !74
  br label %139

139:                                              ; preds = %9, %175
  %140 = phi ptr [ %180, %175 ], [ %138, %9 ]
  %141 = phi i64 [ %179, %175 ], [ 0, %9 ]
  %142 = getelementptr inbounds nuw i8, ptr %3, i64 %141
  %143 = load ptr, ptr %140, align 8, !tbaa !74
  call void @_ZNSt8__detail15_List_node_base11_M_transferEPS0_S1_(ptr noundef nonnull align 8 dereferenceable(16) %2, ptr noundef nonnull %140, ptr noundef %143) #25
  %144 = icmp samesign eq i64 %141, 0
  br i1 %144, label %175, label %145

145:                                              ; preds = %139, %172
  %146 = phi ptr [ %173, %172 ], [ %3, %139 ]
  %147 = load ptr, ptr %146, align 8, !tbaa !74
  %148 = icmp eq ptr %147, %146
  br i1 %148, label %175, label %149

149:                                              ; preds = %145
  %150 = load ptr, ptr %2, align 8, !tbaa !74
  %151 = icmp eq ptr %150, %2
  br i1 %151, label %172, label %152

152:                                              ; preds = %149, %164
  %153 = phi ptr [ %166, %164 ], [ %147, %149 ]
  %154 = phi ptr [ %165, %164 ], [ %150, %149 ]
  %155 = getelementptr inbounds nuw i8, ptr %154, i64 16
  %156 = load double, ptr %155, align 8, !tbaa !16
  %157 = getelementptr inbounds nuw i8, ptr %153, i64 16
  %158 = load double, ptr %157, align 8, !tbaa !16
  %159 = fcmp olt double %156, %158
  br i1 %159, label %160, label %162

160:                                              ; preds = %152
  %161 = load ptr, ptr %154, align 8, !tbaa !74
  call void @_ZNSt8__detail15_List_node_base11_M_transferEPS0_S1_(ptr noundef nonnull align 8 dereferenceable(16) %153, ptr noundef nonnull %154, ptr noundef %161) #25
  br label %164

162:                                              ; preds = %152
  %163 = load ptr, ptr %153, align 8, !tbaa !74
  br label %164

164:                                              ; preds = %162, %160
  %165 = phi ptr [ %161, %160 ], [ %154, %162 ]
  %166 = phi ptr [ %153, %160 ], [ %163, %162 ]
  %167 = icmp ne ptr %166, %146
  %168 = icmp ne ptr %165, %2
  %169 = and i1 %168, %167
  br i1 %169, label %152, label %170, !llvm.loop !83

170:                                              ; preds = %164
  br i1 %168, label %171, label %172

171:                                              ; preds = %170
  call void @_ZNSt8__detail15_List_node_base11_M_transferEPS0_S1_(ptr noundef nonnull align 8 dereferenceable(16) %146, ptr noundef %165, ptr noundef nonnull align 8 dereferenceable(16) %2) #25
  br label %172

172:                                              ; preds = %149, %171, %170
  call void @_ZNSt8__detail15_List_node_base4swapERS0_S1_(ptr noundef nonnull align 8 dereferenceable(16) %2, ptr noundef nonnull align 8 dereferenceable(16) %146) #25
  %173 = getelementptr inbounds nuw i8, ptr %146, i64 16
  %174 = icmp eq ptr %173, %142
  br i1 %174, label %175, label %145, !llvm.loop !84

175:                                              ; preds = %145, %172, %139
  %176 = phi ptr [ %3, %139 ], [ %142, %172 ], [ %146, %145 ]
  %177 = phi i64 [ 0, %139 ], [ 0, %172 ], [ -16, %145 ]
  %178 = phi i64 [ 16, %139 ], [ 16, %172 ], [ 0, %145 ]
  call void @_ZNSt8__detail15_List_node_base4swapERS0_S1_(ptr noundef nonnull align 8 dereferenceable(16) %2, ptr noundef nonnull align 8 dereferenceable(16) %176) #25
  %179 = add nuw nsw i64 %141, %178
  %180 = load ptr, ptr %0, align 8, !tbaa !74
  %181 = icmp eq ptr %180, %0
  br i1 %181, label %182, label %139, !llvm.loop !85

182:                                              ; preds = %175
  %183 = getelementptr inbounds nuw i8, ptr %3, i64 %179
  %184 = icmp eq i64 %179, 16
  br i1 %184, label %218, label %185

185:                                              ; preds = %182, %215
  %186 = phi ptr [ %216, %215 ], [ %12, %182 ]
  %187 = phi ptr [ %186, %215 ], [ %3, %182 ]
  %188 = load ptr, ptr %186, align 8, !tbaa !74
  %189 = load ptr, ptr %187, align 8, !tbaa !74
  %190 = icmp ne ptr %188, %186
  %191 = icmp ne ptr %189, %187
  %192 = select i1 %190, i1 %191, i1 false
  br i1 %192, label %193, label %211

193:                                              ; preds = %185, %205
  %194 = phi ptr [ %207, %205 ], [ %188, %185 ]
  %195 = phi ptr [ %206, %205 ], [ %189, %185 ]
  %196 = getelementptr inbounds nuw i8, ptr %195, i64 16
  %197 = load double, ptr %196, align 8, !tbaa !16
  %198 = getelementptr inbounds nuw i8, ptr %194, i64 16
  %199 = load double, ptr %198, align 8, !tbaa !16
  %200 = fcmp olt double %197, %199
  br i1 %200, label %201, label %203

201:                                              ; preds = %193
  %202 = load ptr, ptr %195, align 8, !tbaa !74
  call void @_ZNSt8__detail15_List_node_base11_M_transferEPS0_S1_(ptr noundef nonnull align 8 dereferenceable(16) %194, ptr noundef nonnull %195, ptr noundef %202) #25
  br label %205

203:                                              ; preds = %193
  %204 = load ptr, ptr %194, align 8, !tbaa !74
  br label %205

205:                                              ; preds = %203, %201
  %206 = phi ptr [ %202, %201 ], [ %195, %203 ]
  %207 = phi ptr [ %194, %201 ], [ %204, %203 ]
  %208 = icmp ne ptr %207, %186
  %209 = icmp ne ptr %206, %187
  %210 = and i1 %209, %208
  br i1 %210, label %193, label %211, !llvm.loop !83

211:                                              ; preds = %205, %185
  %212 = phi ptr [ %189, %185 ], [ %206, %205 ]
  %213 = phi i1 [ %191, %185 ], [ %209, %205 ]
  br i1 %213, label %214, label %215

214:                                              ; preds = %211
  call void @_ZNSt8__detail15_List_node_base11_M_transferEPS0_S1_(ptr noundef nonnull align 8 dereferenceable(16) %186, ptr noundef %212, ptr noundef nonnull align 8 dereferenceable(16) %187) #25
  br label %215

215:                                              ; preds = %211, %214
  %216 = getelementptr inbounds nuw i8, ptr %186, i64 16
  %217 = icmp eq ptr %216, %183
  br i1 %217, label %218, label %185

218:                                              ; preds = %215, %182
  %219 = getelementptr inbounds i8, ptr %142, i64 %177
  call void @_ZNSt8__detail15_List_node_base4swapERS0_S1_(ptr noundef nonnull align 8 dereferenceable(16) %219, ptr noundef nonnull align 8 dereferenceable(16) %0) #25
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #25
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #25
  br label %220

220:                                              ; preds = %218, %6, %1
  ret void
}

; Function Attrs: mustprogress uwtable
define dso_local void @_Z8set_testPdS_i(ptr noundef %0, ptr noundef %1, i32 noundef %2) local_unnamed_addr #2 personality ptr @__gxx_personality_v0 {
  %4 = alloca %"class.std::set", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #25
  call void @_ZNSt3setIdSt4lessIdESaIdEEC2IPdEET_S6_(ptr noundef nonnull align 8 dereferenceable(48) %4, ptr noundef %0, ptr noundef %1)
  %5 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %6 = load ptr, ptr %5, align 8, !tbaa !86
  invoke void @_ZNSt8_Rb_treeIddSt9_IdentityIdESt4lessIdESaIdEE8_M_eraseEPSt13_Rb_tree_nodeIdE(ptr noundef nonnull align 8 dereferenceable(48) %4, ptr noundef %6)
          to label %10 unwind label %7

7:                                                ; preds = %3
  %8 = landingpad { ptr, i32 }
          catch ptr null
  %9 = extractvalue { ptr, i32 } %8, 0
  call void @__clang_call_terminate(ptr %9) #26
  unreachable

10:                                               ; preds = %3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #25
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNSt3setIdSt4lessIdESaIdEEC2IPdEET_S6_(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef %1, ptr noundef %2) unnamed_addr #2 comdat personality ptr @__gxx_personality_v0 {
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 24
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %0, i8 0, i64 24, i1 false)
  store ptr %4, ptr %6, align 8, !tbaa !91
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 32
  store ptr %4, ptr %7, align 8, !tbaa !92
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 40
  store i64 0, ptr %8, align 8, !tbaa !93
  %9 = icmp eq ptr %1, %2
  br i1 %9, label %68, label %10

10:                                               ; preds = %3, %64
  %11 = phi i64 [ %65, %64 ], [ 0, %3 ]
  %12 = phi ptr [ %66, %64 ], [ %1, %3 ]
  %13 = icmp eq i64 %11, 0
  br i1 %13, label %20, label %14

14:                                               ; preds = %10
  %15 = load ptr, ptr %7, align 8, !tbaa !94
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 32
  %17 = load double, ptr %16, align 8, !tbaa !16
  %18 = load double, ptr %12, align 8, !tbaa !16
  %19 = fcmp olt double %17, %18
  br i1 %19, label %49, label %20

20:                                               ; preds = %14, %10
  %21 = load ptr, ptr %5, align 8, !tbaa !94
  %22 = icmp eq ptr %21, null
  br i1 %22, label %35, label %23

23:                                               ; preds = %20
  %24 = load double, ptr %12, align 8, !tbaa !16
  br label %25

25:                                               ; preds = %25, %23
  %26 = phi ptr [ %21, %23 ], [ %32, %25 ]
  %27 = getelementptr inbounds nuw i8, ptr %26, i64 32
  %28 = load double, ptr %27, align 8, !tbaa !16
  %29 = fcmp olt double %24, %28
  %30 = select i1 %29, i64 16, i64 24
  %31 = getelementptr inbounds nuw i8, ptr %26, i64 %30
  %32 = load ptr, ptr %31, align 8, !tbaa !94
  %33 = icmp eq ptr %32, null
  br i1 %33, label %34, label %25, !llvm.loop !95

34:                                               ; preds = %25
  br i1 %29, label %35, label %44

35:                                               ; preds = %34, %20
  %36 = phi ptr [ %26, %34 ], [ %4, %20 ]
  %37 = load ptr, ptr %6, align 8, !tbaa !91
  %38 = icmp eq ptr %36, %37
  br i1 %38, label %49, label %39

39:                                               ; preds = %35
  %40 = tail call noundef ptr @_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base(ptr noundef nonnull %36) #27
  %41 = getelementptr inbounds nuw i8, ptr %40, i64 32
  %42 = load double, ptr %41, align 8, !tbaa !16
  %43 = load double, ptr %12, align 8, !tbaa !16
  br label %44

44:                                               ; preds = %39, %34
  %45 = phi double [ %43, %39 ], [ %24, %34 ]
  %46 = phi double [ %42, %39 ], [ %28, %34 ]
  %47 = phi ptr [ %36, %39 ], [ %26, %34 ]
  %48 = fcmp olt double %46, %45
  br i1 %48, label %49, label %64

49:                                               ; preds = %14, %35, %44
  %50 = phi ptr [ %47, %44 ], [ %36, %35 ], [ %15, %14 ]
  %51 = icmp eq ptr %4, %50
  %52 = load double, ptr %12, align 8, !tbaa !16
  br i1 %51, label %57, label %53

53:                                               ; preds = %49
  %54 = getelementptr inbounds nuw i8, ptr %50, i64 32
  %55 = load double, ptr %54, align 8, !tbaa !16
  %56 = fcmp olt double %52, %55
  br label %57

57:                                               ; preds = %53, %49
  %58 = phi i1 [ true, %49 ], [ %56, %53 ]
  %59 = invoke noalias noundef nonnull dereferenceable(40) ptr @_Znwm(i64 noundef 40) #23
          to label %60 unwind label %69

60:                                               ; preds = %57
  %61 = getelementptr inbounds nuw i8, ptr %59, i64 32
  store double %52, ptr %61, align 8, !tbaa !16
  tail call void @_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_node_baseS0_RS_(i1 noundef %58, ptr noundef nonnull %59, ptr noundef nonnull %50, ptr noundef nonnull align 8 dereferenceable(32) %4) #25
  %62 = load i64, ptr %8, align 8, !tbaa !93
  %63 = add i64 %62, 1
  store i64 %63, ptr %8, align 8, !tbaa !93
  br label %64

64:                                               ; preds = %44, %60
  %65 = phi i64 [ %63, %60 ], [ %11, %44 ]
  %66 = getelementptr inbounds nuw i8, ptr %12, i64 8
  %67 = icmp eq ptr %66, %2
  br i1 %67, label %68, label %10, !llvm.loop !96

68:                                               ; preds = %64, %3
  ret void

69:                                               ; preds = %57
  %70 = landingpad { ptr, i32 }
          cleanup
  tail call void @_ZNSt8_Rb_treeIddSt9_IdentityIdESt4lessIdESaIdEED2Ev(ptr noundef nonnull align 8 dereferenceable(48) %0) #25
  resume { ptr, i32 } %70
}

; Function Attrs: mustprogress uwtable
define dso_local void @_Z13multiset_testPdS_i(ptr noundef %0, ptr noundef readnone captures(address) %1, i32 %2) local_unnamed_addr #2 personality ptr @__gxx_personality_v0 {
  %4 = alloca %"struct.std::_Rb_tree<double, double, std::_Identity<double>, std::less<double>>::_Alloc_node", align 8
  %5 = alloca %"class.std::multiset", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #25
  %6 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %7 = getelementptr inbounds nuw i8, ptr %5, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(48) %5, i8 0, i64 24, i1 false)
  store ptr %6, ptr %7, align 8, !tbaa !91
  %8 = getelementptr inbounds nuw i8, ptr %5, i64 32
  store ptr %6, ptr %8, align 8, !tbaa !92
  %9 = getelementptr inbounds nuw i8, ptr %5, i64 40
  store i64 0, ptr %9, align 8, !tbaa !93
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #25
  store ptr %5, ptr %4, align 8, !tbaa !97
  %10 = icmp eq ptr %0, %1
  br i1 %10, label %11, label %12

11:                                               ; preds = %3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #25
  br label %40

12:                                               ; preds = %3, %15
  %13 = phi ptr [ %16, %15 ], [ %0, %3 ]
  %14 = invoke i64 @_ZNSt8_Rb_treeIddSt9_IdentityIdESt4lessIdESaIdEE16_M_insert_equal_IRdNS5_11_Alloc_nodeEEESt17_Rb_tree_iteratorIdESt23_Rb_tree_const_iteratorIdEOT_RT0_(ptr noundef nonnull align 8 dereferenceable(48) %5, ptr nonnull %6, ptr noundef nonnull align 8 dereferenceable(8) %13, ptr noundef nonnull align 8 dereferenceable(8) %4)
          to label %15 unwind label %18

15:                                               ; preds = %12
  %16 = getelementptr inbounds nuw i8, ptr %13, i64 8
  %17 = icmp eq ptr %16, %1
  br i1 %17, label %20, label %12, !llvm.loop !99

18:                                               ; preds = %12
  %19 = landingpad { ptr, i32 }
          cleanup
  call void @_ZNSt8_Rb_treeIddSt9_IdentityIdESt4lessIdESaIdEED2Ev(ptr noundef nonnull align 8 dereferenceable(48) %5) #25
  resume { ptr, i32 } %19

20:                                               ; preds = %15
  %21 = load ptr, ptr %7, align 8, !tbaa !91
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #25
  %22 = icmp eq ptr %21, %6
  br i1 %22, label %40, label %23

23:                                               ; preds = %20, %37
  %24 = phi ptr [ %38, %37 ], [ %21, %20 ]
  %25 = call noundef ptr @_ZSt18_Rb_tree_incrementPKSt18_Rb_tree_node_base(ptr noundef %24) #27
  %26 = icmp eq ptr %25, %6
  br i1 %26, label %40, label %27

27:                                               ; preds = %23
  %28 = getelementptr inbounds nuw i8, ptr %24, i64 32
  %29 = load double, ptr %28, align 8, !tbaa !16
  %30 = getelementptr inbounds nuw i8, ptr %25, i64 32
  %31 = load double, ptr %30, align 8, !tbaa !16
  %32 = fcmp oeq double %29, %31
  br i1 %32, label %33, label %37

33:                                               ; preds = %27
  %34 = call noundef nonnull ptr @_ZSt28_Rb_tree_rebalance_for_erasePSt18_Rb_tree_node_baseRS_(ptr noundef nonnull %25, ptr noundef nonnull align 8 dereferenceable(32) %6) #25
  call void @_ZdlPvm(ptr noundef nonnull %34, i64 noundef 40) #22
  %35 = load i64, ptr %9, align 8, !tbaa !93
  %36 = add i64 %35, -1
  store i64 %36, ptr %9, align 8, !tbaa !93
  br label %37

37:                                               ; preds = %27, %33
  %38 = phi ptr [ %24, %33 ], [ %25, %27 ]
  %39 = icmp eq ptr %38, %6
  br i1 %39, label %40, label %23

40:                                               ; preds = %37, %23, %11, %20
  %41 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %42 = load ptr, ptr %41, align 8, !tbaa !86
  invoke void @_ZNSt8_Rb_treeIddSt9_IdentityIdESt4lessIdESaIdEE8_M_eraseEPSt13_Rb_tree_nodeIdE(ptr noundef nonnull align 8 dereferenceable(48) %5, ptr noundef %42)
          to label %46 unwind label %43

43:                                               ; preds = %40
  %44 = landingpad { ptr, i32 }
          catch ptr null
  %45 = extractvalue { ptr, i32 } %44, 0
  call void @__clang_call_terminate(ptr %45) #26
  unreachable

46:                                               ; preds = %40
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #25
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #6

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: write) uwtable
define dso_local void @_Z10initializePdS_(ptr noundef writeonly captures(address) %0, ptr noundef readnone captures(address) %1) local_unnamed_addr #7 {
  %3 = icmp eq ptr %0, %1
  br i1 %3, label %10, label %4

4:                                                ; preds = %2, %4
  %5 = phi double [ %8, %4 ], [ 0.000000e+00, %2 ]
  %6 = phi ptr [ %7, %4 ], [ %0, %2 ]
  %7 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store double %5, ptr %6, align 8, !tbaa !16
  %8 = fadd double %5, 1.000000e+00
  %9 = icmp eq ptr %7, %1
  br i1 %9, label %10, label %4, !llvm.loop !100

10:                                               ; preds = %4, %2
  ret void
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(errnomem: write) uwtable
define dso_local noundef double @_Z6logtwod(double noundef %0) local_unnamed_addr #8 {
  %2 = tail call double @log(double noundef %0) #25, !tbaa !101
  %3 = fdiv double %2, 0x3FE62E42FEFA39EF
  ret double %3
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @log(double noundef) local_unnamed_addr #9

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(errnomem: write) uwtable
define dso_local noundef i32 @_Z15number_of_testsi(i32 noundef %0) local_unnamed_addr #8 {
  %2 = sitofp i32 %0 to double
  %3 = tail call double @log(double noundef %2) #25, !tbaa !101
  %4 = fdiv double %3, 0x3FE62E42FEFA39EF
  %5 = fmul double %4, %2
  %6 = fdiv double 0x4173021B091BF3AA, %5
  %7 = tail call double @llvm.floor.f64(double %6)
  %8 = fptosi double %7 to i32
  ret i32 %8
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.floor.f64(double) #10

; Function Attrs: mustprogress uwtable
define dso_local void @_Z9run_testsi(i32 noundef %0) local_unnamed_addr #2 personality ptr @__gxx_personality_v0 {
  %2 = alloca %"class.std::set", align 8
  %3 = sitofp i32 %0 to double
  %4 = tail call double @log(double noundef %3) #25, !tbaa !101
  %5 = fdiv double %4, 0x3FE62E42FEFA39EF
  %6 = fmul double %5, %3
  %7 = fdiv double 0x4173021B091BF3AA, %6
  %8 = tail call double @llvm.floor.f64(double %7)
  %9 = fptosi double %8 to i32
  %10 = shl nsw i32 %0, 1
  %11 = sext i32 %10 to i64
  %12 = load ptr, ptr @result_times, align 8, !tbaa !6
  %13 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @result_times, i64 8), align 8, !tbaa !103
  %14 = icmp eq ptr %13, %12
  br i1 %14, label %16, label %15

15:                                               ; preds = %1
  store ptr %12, ptr getelementptr inbounds nuw (i8, ptr @result_times, i64 8), align 8, !tbaa !103
  br label %16

16:                                               ; preds = %1, %15
  %17 = icmp slt i32 %0, 0
  br i1 %17, label %18, label %19

18:                                               ; preds = %16
  tail call void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str) #24
  unreachable

19:                                               ; preds = %16
  %20 = icmp eq i32 %0, 0
  br i1 %20, label %68, label %21

21:                                               ; preds = %19
  %22 = shl nuw nsw i64 %11, 3
  %23 = tail call noalias noundef nonnull ptr @_Znwm(i64 noundef %22) #23
  %24 = getelementptr inbounds nuw double, ptr %23, i64 %11
  store double 0.000000e+00, ptr %23, align 8, !tbaa !16
  %25 = getelementptr i8, ptr %23, i64 8
  %26 = add nsw i64 %22, -8
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %25, i8 0, i64 %26, i1 false), !tbaa !16
  %27 = ptrtoint ptr %24 to i64
  %28 = getelementptr inbounds nuw i8, ptr %23, i64 %22
  %29 = zext nneg i32 %0 to i64
  %30 = shl nuw nsw i64 %29, 3
  %31 = getelementptr inbounds nuw i8, ptr %23, i64 %30
  br label %32

32:                                               ; preds = %21, %32
  %33 = phi double [ %36, %32 ], [ 0.000000e+00, %21 ]
  %34 = phi ptr [ %35, %32 ], [ %23, %21 ]
  %35 = getelementptr inbounds nuw i8, ptr %34, i64 8
  store double %33, ptr %34, align 8, !tbaa !16
  %36 = fadd double %33, 1.000000e+00
  %37 = icmp eq ptr %35, %31
  br i1 %37, label %38, label %32, !llvm.loop !100

38:                                               ; preds = %32
  %39 = icmp samesign eq i64 %30, %22
  br i1 %39, label %46, label %40

40:                                               ; preds = %38, %40
  %41 = phi double [ %44, %40 ], [ 0.000000e+00, %38 ]
  %42 = phi ptr [ %43, %40 ], [ %31, %38 ]
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 8
  store double %41, ptr %42, align 8, !tbaa !16
  %44 = fadd double %41, 1.000000e+00
  %45 = icmp eq ptr %43, %28
  br i1 %45, label %46, label %40, !llvm.loop !100

46:                                               ; preds = %40, %38
  %47 = getelementptr inbounds nuw i8, ptr %23, i64 8
  %48 = ptrtoint ptr %23 to i64
  br label %49

49:                                               ; preds = %63, %46
  %50 = phi ptr [ %47, %46 ], [ %64, %63 ]
  %51 = tail call i32 @rand() #25
  %52 = sext i32 %51 to i64
  %53 = ptrtoint ptr %50 to i64
  %54 = sub i64 %53, %48
  %55 = ashr exact i64 %54, 3
  %56 = add nsw i64 %55, 1
  %57 = srem i64 %52, %56
  %58 = getelementptr inbounds double, ptr %23, i64 %57
  %59 = icmp eq ptr %50, %58
  br i1 %59, label %63, label %60

60:                                               ; preds = %49
  %61 = load double, ptr %50, align 8, !tbaa !16
  %62 = load double, ptr %58, align 8, !tbaa !16
  store double %62, ptr %50, align 8, !tbaa !16
  store double %61, ptr %58, align 8, !tbaa !16
  br label %63

63:                                               ; preds = %60, %49
  %64 = getelementptr inbounds nuw i8, ptr %50, i64 8
  %65 = icmp eq ptr %64, %28
  br i1 %65, label %66, label %49, !llvm.loop !104

66:                                               ; preds = %63
  %67 = icmp sgt i32 %9, 0
  br i1 %67, label %70, label %116

68:                                               ; preds = %19
  %69 = icmp sgt i32 %9, 0
  br i1 %69, label %70, label %121

70:                                               ; preds = %68, %66
  %71 = phi i64 [ 0, %68 ], [ %27, %66 ]
  %72 = phi ptr [ null, %68 ], [ %23, %66 ]
  %73 = phi ptr [ null, %68 ], [ %28, %66 ]
  br label %74

74:                                               ; preds = %70, %76
  %75 = phi i32 [ %77, %76 ], [ %9, %70 ]
  invoke void @_Z10array_testPdS_i(ptr noundef nonnull %72, ptr noundef nonnull %73, i32 poison)
          to label %76 unwind label %134

76:                                               ; preds = %74
  %77 = add nsw i32 %75, -1
  %78 = icmp samesign ugt i32 %75, 1
  br i1 %78, label %74, label %79, !llvm.loop !13

79:                                               ; preds = %76, %81
  %80 = phi i32 [ %82, %81 ], [ %9, %76 ]
  invoke void @_Z19vector_pointer_testPdS_i(ptr noundef nonnull %72, ptr noundef nonnull %73, i32 poison)
          to label %81 unwind label %132

81:                                               ; preds = %79
  %82 = add nsw i32 %80, -1
  %83 = icmp samesign ugt i32 %80, 1
  br i1 %83, label %79, label %84, !llvm.loop !13

84:                                               ; preds = %81, %86
  %85 = phi i32 [ %87, %86 ], [ %9, %81 ]
  invoke void @_Z20vector_iterator_testPdS_i(ptr noundef nonnull %72, ptr noundef nonnull %73, i32 poison)
          to label %86 unwind label %130

86:                                               ; preds = %84
  %87 = add nsw i32 %85, -1
  %88 = icmp samesign ugt i32 %85, 1
  br i1 %88, label %84, label %89, !llvm.loop !13

89:                                               ; preds = %86, %91
  %90 = phi i32 [ %92, %91 ], [ %9, %86 ]
  invoke void @_Z10deque_testPdS_i(ptr noundef nonnull %72, ptr noundef nonnull %73, i32 poison)
          to label %91 unwind label %128

91:                                               ; preds = %89
  %92 = add nsw i32 %90, -1
  %93 = icmp samesign ugt i32 %90, 1
  br i1 %93, label %89, label %94, !llvm.loop !13

94:                                               ; preds = %91, %96
  %95 = phi i32 [ %97, %96 ], [ %9, %91 ]
  invoke void @_Z9list_testPdS_i(ptr noundef nonnull %72, ptr noundef nonnull %73, i32 poison)
          to label %96 unwind label %126

96:                                               ; preds = %94
  %97 = add nsw i32 %95, -1
  %98 = icmp samesign ugt i32 %95, 1
  br i1 %98, label %94, label %99, !llvm.loop !13

99:                                               ; preds = %96
  %100 = getelementptr inbounds nuw i8, ptr %2, i64 16
  br label %101

101:                                              ; preds = %99, %109
  %102 = phi i32 [ %103, %109 ], [ %9, %99 ]
  %103 = add nsw i32 %102, -1
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #25
  invoke void @_ZNSt3setIdSt4lessIdESaIdEEC2IPdEET_S6_(ptr noundef nonnull align 8 dereferenceable(48) %2, ptr noundef nonnull %72, ptr noundef nonnull %73)
          to label %104 unwind label %124

104:                                              ; preds = %101
  %105 = load ptr, ptr %100, align 8, !tbaa !86
  invoke void @_ZNSt8_Rb_treeIddSt9_IdentityIdESt4lessIdESaIdEE8_M_eraseEPSt13_Rb_tree_nodeIdE(ptr noundef nonnull align 8 dereferenceable(48) %2, ptr noundef %105)
          to label %109 unwind label %106

106:                                              ; preds = %104
  %107 = landingpad { ptr, i32 }
          catch ptr null
  %108 = extractvalue { ptr, i32 } %107, 0
  call void @__clang_call_terminate(ptr %108) #26
  unreachable

109:                                              ; preds = %104
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #25
  %110 = icmp samesign ugt i32 %102, 1
  br i1 %110, label %101, label %111, !llvm.loop !13

111:                                              ; preds = %109, %113
  %112 = phi i32 [ %114, %113 ], [ %9, %109 ]
  invoke void @_Z13multiset_testPdS_i(ptr noundef nonnull %72, ptr noundef nonnull %73, i32 poison)
          to label %113 unwind label %122

113:                                              ; preds = %111
  %114 = add nsw i32 %112, -1
  %115 = icmp samesign ugt i32 %112, 1
  br i1 %115, label %111, label %116, !llvm.loop !13

116:                                              ; preds = %113, %66
  %117 = phi i64 [ %27, %66 ], [ %71, %113 ]
  %118 = phi ptr [ %23, %66 ], [ %72, %113 ]
  %119 = ptrtoint ptr %118 to i64
  %120 = sub i64 %117, %119
  call void @_ZdlPvm(ptr noundef nonnull %118, i64 noundef %120) #22
  br label %121

121:                                              ; preds = %68, %116
  ret void

122:                                              ; preds = %111
  %123 = landingpad { ptr, i32 }
          cleanup
  br label %136

124:                                              ; preds = %101
  %125 = landingpad { ptr, i32 }
          cleanup
  br label %136

126:                                              ; preds = %94
  %127 = landingpad { ptr, i32 }
          cleanup
  br label %136

128:                                              ; preds = %89
  %129 = landingpad { ptr, i32 }
          cleanup
  br label %136

130:                                              ; preds = %84
  %131 = landingpad { ptr, i32 }
          cleanup
  br label %136

132:                                              ; preds = %79
  %133 = landingpad { ptr, i32 }
          cleanup
  br label %136

134:                                              ; preds = %74
  %135 = landingpad { ptr, i32 }
          cleanup
  br label %136

136:                                              ; preds = %124, %128, %132, %134, %130, %126, %122
  %137 = phi { ptr, i32 } [ %123, %122 ], [ %125, %124 ], [ %127, %126 ], [ %129, %128 ], [ %131, %130 ], [ %133, %132 ], [ %135, %134 ]
  %138 = ptrtoint ptr %72 to i64
  %139 = sub i64 %71, %138
  call void @_ZdlPvm(ptr noundef nonnull %72, i64 noundef %139) #22
  resume { ptr, i32 } %137
}

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #11 {
  tail call void @_Z9run_testsi(i32 noundef 100000)
  ret i32 0
}

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) local_unnamed_addr #12 comdat {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #25
  tail call void @_ZSt9terminatev() #26
  unreachable
}

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

; Function Attrs: cold nofree noreturn
declare void @_ZSt9terminatev() local_unnamed_addr #13

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #5

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZNSt8_Rb_treeIddSt9_IdentityIdESt4lessIdESaIdEED2Ev(ptr noundef nonnull align 8 dereferenceable(48) %0) unnamed_addr #0 comdat personality ptr @__gxx_personality_v0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %3 = load ptr, ptr %2, align 8, !tbaa !86
  invoke void @_ZNSt8_Rb_treeIddSt9_IdentityIdESt4lessIdESaIdEE8_M_eraseEPSt13_Rb_tree_nodeIdE(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef %3)
          to label %4 unwind label %5

4:                                                ; preds = %1
  ret void

5:                                                ; preds = %1
  %6 = landingpad { ptr, i32 }
          catch ptr null
  %7 = extractvalue { ptr, i32 } %6, 0
  tail call void @__clang_call_terminate(ptr %7) #26
  unreachable
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNSt8_Rb_treeIddSt9_IdentityIdESt4lessIdESaIdEE8_M_eraseEPSt13_Rb_tree_nodeIdE(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef %1) local_unnamed_addr #2 comdat personality ptr @__gxx_personality_v0 {
  %3 = icmp eq ptr %1, null
  br i1 %3, label %11, label %4

4:                                                ; preds = %2, %4
  %5 = phi ptr [ %9, %4 ], [ %1, %2 ]
  %6 = getelementptr inbounds nuw i8, ptr %5, i64 24
  %7 = load ptr, ptr %6, align 8, !tbaa !105
  tail call void @_ZNSt8_Rb_treeIddSt9_IdentityIdESt4lessIdESaIdEE8_M_eraseEPSt13_Rb_tree_nodeIdE(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef %7)
  %8 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %9 = load ptr, ptr %8, align 8, !tbaa !106
  tail call void @_ZdlPvm(ptr noundef nonnull %5, i64 noundef 40) #22
  %10 = icmp eq ptr %9, null
  br i1 %10, label %11, label %4, !llvm.loop !107

11:                                               ; preds = %4, %2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg) #6

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_(ptr noundef %0, ptr noundef %1, i64 noundef %2, i8 %3) local_unnamed_addr #2 comdat {
  %5 = alloca %"struct.__gnu_cxx::__ops::_Iter_less_iter", align 1
  %6 = ptrtoint ptr %0 to i64
  %7 = ptrtoint ptr %1 to i64
  %8 = sub i64 %7, %6
  %9 = icmp sgt i64 %8, 128
  br i1 %9, label %10, label %126

10:                                               ; preds = %4
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8
  br label %12

12:                                               ; preds = %10, %122
  %13 = phi i64 [ %8, %10 ], [ %124, %122 ]
  %14 = phi ptr [ %1, %10 ], [ %110, %122 ]
  %15 = phi i64 [ %2, %10 ], [ %78, %122 ]
  %16 = icmp eq i64 %15, 0
  br i1 %16, label %17, label %77

17:                                               ; preds = %12
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_(ptr noundef %0, ptr noundef %14, ptr noundef nonnull align 1 dereferenceable(1) %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  br label %18

18:                                               ; preds = %17, %73
  %19 = phi ptr [ %20, %73 ], [ %14, %17 ]
  %20 = getelementptr inbounds i8, ptr %19, i64 -8
  %21 = load double, ptr %20, align 8, !tbaa !16
  %22 = load double, ptr %0, align 8, !tbaa !16
  store double %22, ptr %20, align 8, !tbaa !16
  %23 = ptrtoint ptr %20 to i64
  %24 = sub i64 %23, %6
  %25 = ashr exact i64 %24, 3
  %26 = add nsw i64 %25, -1
  %27 = sdiv i64 %26, 2
  %28 = icmp sgt i64 %25, 2
  br i1 %28, label %29, label %45

29:                                               ; preds = %18, %29
  %30 = phi i64 [ %40, %29 ], [ 0, %18 ]
  %31 = shl i64 %30, 1
  %32 = add i64 %31, 2
  %33 = getelementptr inbounds double, ptr %0, i64 %32
  %34 = getelementptr double, ptr %0, i64 %31
  %35 = getelementptr i8, ptr %34, i64 8
  %36 = load double, ptr %33, align 8, !tbaa !16
  %37 = load double, ptr %35, align 8, !tbaa !16
  %38 = fcmp olt double %36, %37
  %39 = or disjoint i64 %31, 1
  %40 = select i1 %38, i64 %39, i64 %32
  %41 = getelementptr inbounds double, ptr %0, i64 %40
  %42 = load double, ptr %41, align 8, !tbaa !16
  %43 = getelementptr inbounds double, ptr %0, i64 %30
  store double %42, ptr %43, align 8, !tbaa !16
  %44 = icmp slt i64 %40, %27
  br i1 %44, label %29, label %45, !llvm.loop !108

45:                                               ; preds = %29, %18
  %46 = phi i64 [ 0, %18 ], [ %40, %29 ]
  %47 = and i64 %24, 8
  %48 = icmp eq i64 %47, 0
  br i1 %48, label %49, label %59

49:                                               ; preds = %45
  %50 = add nsw i64 %25, -2
  %51 = ashr exact i64 %50, 1
  %52 = icmp eq i64 %46, %51
  br i1 %52, label %53, label %59

53:                                               ; preds = %49
  %54 = shl nuw nsw i64 %46, 1
  %55 = or disjoint i64 %54, 1
  %56 = getelementptr inbounds nuw double, ptr %0, i64 %55
  %57 = load double, ptr %56, align 8, !tbaa !16
  %58 = getelementptr inbounds double, ptr %0, i64 %46
  store double %57, ptr %58, align 8, !tbaa !16
  br label %61

59:                                               ; preds = %49, %45
  %60 = icmp eq i64 %46, 0
  br i1 %60, label %73, label %61

61:                                               ; preds = %59, %53
  %62 = phi i64 [ %46, %59 ], [ %55, %53 ]
  br label %63

63:                                               ; preds = %61, %70
  %64 = phi i64 [ %66, %70 ], [ %62, %61 ]
  %65 = add nsw i64 %64, -1
  %66 = lshr i64 %65, 1
  %67 = getelementptr inbounds nuw double, ptr %0, i64 %66
  %68 = load double, ptr %67, align 8, !tbaa !16
  %69 = fcmp olt double %68, %21
  br i1 %69, label %70, label %73

70:                                               ; preds = %63
  %71 = getelementptr inbounds double, ptr %0, i64 %64
  store double %68, ptr %71, align 8, !tbaa !16
  %72 = icmp ult i64 %65, 2
  br i1 %72, label %73, label %63, !llvm.loop !109

73:                                               ; preds = %70, %63, %59
  %74 = phi i64 [ 0, %59 ], [ %64, %63 ], [ 0, %70 ]
  %75 = getelementptr inbounds double, ptr %0, i64 %74
  store double %21, ptr %75, align 8, !tbaa !16
  %76 = icmp sgt i64 %24, 8
  br i1 %76, label %18, label %126, !llvm.loop !110

77:                                               ; preds = %12
  %78 = add nsw i64 %15, -1
  %79 = lshr i64 %13, 4
  %80 = getelementptr inbounds nuw double, ptr %0, i64 %79
  %81 = getelementptr inbounds i8, ptr %14, i64 -8
  %82 = load double, ptr %11, align 8, !tbaa !16
  %83 = load double, ptr %80, align 8, !tbaa !16
  %84 = fcmp olt double %82, %83
  %85 = load double, ptr %81, align 8, !tbaa !16
  br i1 %84, label %86, label %95

86:                                               ; preds = %77
  %87 = fcmp olt double %83, %85
  br i1 %87, label %88, label %90

88:                                               ; preds = %86
  %89 = load double, ptr %0, align 8, !tbaa !16
  store double %83, ptr %0, align 8, !tbaa !16
  store double %89, ptr %80, align 8, !tbaa !16
  br label %104

90:                                               ; preds = %86
  %91 = fcmp olt double %82, %85
  %92 = load double, ptr %0, align 8, !tbaa !16
  br i1 %91, label %93, label %94

93:                                               ; preds = %90
  store double %85, ptr %0, align 8, !tbaa !16
  store double %92, ptr %81, align 8, !tbaa !16
  br label %104

94:                                               ; preds = %90
  store double %82, ptr %0, align 8, !tbaa !16
  store double %92, ptr %11, align 8, !tbaa !16
  br label %104

95:                                               ; preds = %77
  %96 = fcmp olt double %82, %85
  br i1 %96, label %97, label %99

97:                                               ; preds = %95
  %98 = load double, ptr %0, align 8, !tbaa !16
  store double %82, ptr %0, align 8, !tbaa !16
  store double %98, ptr %11, align 8, !tbaa !16
  br label %104

99:                                               ; preds = %95
  %100 = fcmp olt double %83, %85
  %101 = load double, ptr %0, align 8, !tbaa !16
  br i1 %100, label %102, label %103

102:                                              ; preds = %99
  store double %85, ptr %0, align 8, !tbaa !16
  store double %101, ptr %81, align 8, !tbaa !16
  br label %104

103:                                              ; preds = %99
  store double %83, ptr %0, align 8, !tbaa !16
  store double %101, ptr %80, align 8, !tbaa !16
  br label %104

104:                                              ; preds = %103, %102, %97, %94, %93, %88
  br label %105

105:                                              ; preds = %104, %121
  %106 = phi ptr [ %116, %121 ], [ %14, %104 ]
  %107 = phi ptr [ %113, %121 ], [ %11, %104 ]
  %108 = load double, ptr %0, align 8, !tbaa !16
  br label %109

109:                                              ; preds = %109, %105
  %110 = phi ptr [ %107, %105 ], [ %113, %109 ]
  %111 = load double, ptr %110, align 8, !tbaa !16
  %112 = fcmp olt double %111, %108
  %113 = getelementptr inbounds nuw i8, ptr %110, i64 8
  br i1 %112, label %109, label %114, !llvm.loop !111

114:                                              ; preds = %109, %114
  %115 = phi ptr [ %116, %114 ], [ %106, %109 ]
  %116 = getelementptr inbounds i8, ptr %115, i64 -8
  %117 = load double, ptr %116, align 8, !tbaa !16
  %118 = fcmp olt double %108, %117
  br i1 %118, label %114, label %119, !llvm.loop !112

119:                                              ; preds = %114
  %120 = icmp ult ptr %110, %116
  br i1 %120, label %121, label %122

121:                                              ; preds = %119
  store double %117, ptr %110, align 8, !tbaa !16
  store double %111, ptr %116, align 8, !tbaa !16
  br label %105, !llvm.loop !113

122:                                              ; preds = %119
  tail call void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_(ptr noundef nonnull %110, ptr noundef %14, i64 noundef %78, i8 undef)
  %123 = ptrtoint ptr %110 to i64
  %124 = sub i64 %123, %6
  %125 = icmp sgt i64 %124, 128
  br i1 %125, label %12, label %126, !llvm.loop !114

126:                                              ; preds = %122, %73, %4
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_(ptr noundef %0, ptr noundef %1, i8 %2) local_unnamed_addr #2 comdat {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = icmp sgt i64 %6, 128
  br i1 %7, label %8, label %56

8:                                                ; preds = %3
  %9 = getelementptr i8, ptr %0, i64 8
  br label %10

10:                                               ; preds = %32, %8
  %11 = phi i64 [ 8, %8 ], [ %34, %32 ]
  %12 = phi ptr [ %0, %8 ], [ %13, %32 ]
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 %11
  %14 = load double, ptr %13, align 8, !tbaa !16
  %15 = load double, ptr %0, align 8, !tbaa !16
  %16 = fcmp olt double %14, %15
  br i1 %16, label %17, label %22

17:                                               ; preds = %10
  %18 = icmp samesign ugt i64 %11, 8
  br i1 %18, label %19, label %20, !prof !15

19:                                               ; preds = %17
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %9, ptr noundef nonnull align 8 dereferenceable(1) %0, i64 %11, i1 false)
  br label %32

20:                                               ; preds = %17
  %21 = getelementptr inbounds nuw i8, ptr %12, i64 8
  store double %15, ptr %21, align 8, !tbaa !16
  br label %32

22:                                               ; preds = %10
  %23 = load double, ptr %12, align 8, !tbaa !16
  %24 = fcmp olt double %14, %23
  br i1 %24, label %25, label %32

25:                                               ; preds = %22, %25
  %26 = phi double [ %30, %25 ], [ %23, %22 ]
  %27 = phi ptr [ %29, %25 ], [ %12, %22 ]
  %28 = phi ptr [ %27, %25 ], [ %13, %22 ]
  store double %26, ptr %28, align 8, !tbaa !16
  %29 = getelementptr inbounds i8, ptr %27, i64 -8
  %30 = load double, ptr %29, align 8, !tbaa !16
  %31 = fcmp olt double %14, %30
  br i1 %31, label %25, label %32, !llvm.loop !115

32:                                               ; preds = %25, %22, %20, %19
  %33 = phi ptr [ %0, %19 ], [ %0, %20 ], [ %13, %22 ], [ %27, %25 ]
  store double %14, ptr %33, align 8, !tbaa !16
  %34 = add nuw nsw i64 %11, 8
  %35 = icmp eq i64 %34, 128
  br i1 %35, label %36, label %10, !llvm.loop !116

36:                                               ; preds = %32
  %37 = getelementptr inbounds nuw i8, ptr %0, i64 128
  %38 = icmp eq ptr %37, %1
  br i1 %38, label %94, label %39

39:                                               ; preds = %36, %52
  %40 = phi ptr [ %54, %52 ], [ %37, %36 ]
  %41 = load double, ptr %40, align 8, !tbaa !16
  %42 = getelementptr inbounds i8, ptr %40, i64 -8
  %43 = load double, ptr %42, align 8, !tbaa !16
  %44 = fcmp olt double %41, %43
  br i1 %44, label %45, label %52

45:                                               ; preds = %39, %45
  %46 = phi double [ %50, %45 ], [ %43, %39 ]
  %47 = phi ptr [ %49, %45 ], [ %42, %39 ]
  %48 = phi ptr [ %47, %45 ], [ %40, %39 ]
  store double %46, ptr %48, align 8, !tbaa !16
  %49 = getelementptr inbounds i8, ptr %47, i64 -8
  %50 = load double, ptr %49, align 8, !tbaa !16
  %51 = fcmp olt double %41, %50
  br i1 %51, label %45, label %52, !llvm.loop !115

52:                                               ; preds = %45, %39
  %53 = phi ptr [ %40, %39 ], [ %47, %45 ]
  store double %41, ptr %53, align 8, !tbaa !16
  %54 = getelementptr inbounds nuw i8, ptr %40, i64 8
  %55 = icmp eq ptr %54, %1
  br i1 %55, label %94, label %39, !llvm.loop !117

56:                                               ; preds = %3
  %57 = icmp eq ptr %0, %1
  %58 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %59 = icmp eq ptr %58, %1
  %60 = select i1 %57, i1 true, i1 %59
  br i1 %60, label %94, label %61

61:                                               ; preds = %56, %90
  %62 = phi ptr [ %92, %90 ], [ %58, %56 ]
  %63 = phi ptr [ %62, %90 ], [ %0, %56 ]
  %64 = load double, ptr %62, align 8, !tbaa !16
  %65 = load double, ptr %0, align 8, !tbaa !16
  %66 = fcmp olt double %64, %65
  br i1 %66, label %67, label %80

67:                                               ; preds = %61
  %68 = ptrtoint ptr %62 to i64
  %69 = sub i64 %68, %5
  %70 = ashr exact i64 %69, 3
  %71 = icmp sgt i64 %70, 1
  br i1 %71, label %72, label %76, !prof !15

72:                                               ; preds = %67
  %73 = getelementptr inbounds nuw i8, ptr %63, i64 16
  %74 = sub nsw i64 0, %70
  %75 = getelementptr inbounds double, ptr %73, i64 %74
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %75, ptr noundef nonnull align 8 dereferenceable(1) %0, i64 %69, i1 false)
  br label %90

76:                                               ; preds = %67
  %77 = icmp eq i64 %69, 8
  br i1 %77, label %78, label %90

78:                                               ; preds = %76
  %79 = getelementptr inbounds nuw i8, ptr %63, i64 8
  store double %65, ptr %79, align 8, !tbaa !16
  br label %90

80:                                               ; preds = %61
  %81 = load double, ptr %63, align 8, !tbaa !16
  %82 = fcmp olt double %64, %81
  br i1 %82, label %83, label %90

83:                                               ; preds = %80, %83
  %84 = phi double [ %88, %83 ], [ %81, %80 ]
  %85 = phi ptr [ %87, %83 ], [ %63, %80 ]
  %86 = phi ptr [ %85, %83 ], [ %62, %80 ]
  store double %84, ptr %86, align 8, !tbaa !16
  %87 = getelementptr inbounds i8, ptr %85, i64 -8
  %88 = load double, ptr %87, align 8, !tbaa !16
  %89 = fcmp olt double %64, %88
  br i1 %89, label %83, label %90, !llvm.loop !115

90:                                               ; preds = %83, %80, %78, %76, %72
  %91 = phi ptr [ %0, %72 ], [ %0, %76 ], [ %0, %78 ], [ %62, %80 ], [ %85, %83 ]
  store double %64, ptr %91, align 8, !tbaa !16
  %92 = getelementptr inbounds nuw i8, ptr %62, i64 8
  %93 = icmp eq ptr %92, %1
  br i1 %93, label %94, label %61, !llvm.loop !116

94:                                               ; preds = %90, %52, %56, %36
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_(ptr noundef %0, ptr noundef %1, ptr noundef nonnull align 1 dereferenceable(1) %2) local_unnamed_addr #2 comdat {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = ashr exact i64 %6, 3
  %8 = icmp slt i64 %7, 2
  br i1 %8, label %103, label %9

9:                                                ; preds = %3
  %10 = add nsw i64 %7, -2
  %11 = lshr i64 %10, 1
  %12 = add nsw i64 %7, -1
  %13 = lshr i64 %12, 1
  %14 = and i64 %6, 8
  %15 = icmp eq i64 %14, 0
  %16 = lshr exact i64 %10, 1
  br i1 %15, label %17, label %21

17:                                               ; preds = %9
  %18 = or disjoint i64 %10, 1
  %19 = getelementptr inbounds nuw double, ptr %0, i64 %18
  %20 = getelementptr inbounds nuw double, ptr %0, i64 %16
  br label %59

21:                                               ; preds = %9, %54
  %22 = phi i64 [ %58, %54 ], [ %11, %9 ]
  %23 = getelementptr inbounds nuw double, ptr %0, i64 %22
  %24 = load double, ptr %23, align 8, !tbaa !16
  %25 = icmp slt i64 %22, %13
  br i1 %25, label %26, label %54

26:                                               ; preds = %21, %26
  %27 = phi i64 [ %37, %26 ], [ %22, %21 ]
  %28 = shl i64 %27, 1
  %29 = add i64 %28, 2
  %30 = getelementptr inbounds double, ptr %0, i64 %29
  %31 = getelementptr double, ptr %0, i64 %28
  %32 = getelementptr i8, ptr %31, i64 8
  %33 = load double, ptr %30, align 8, !tbaa !16
  %34 = load double, ptr %32, align 8, !tbaa !16
  %35 = fcmp olt double %33, %34
  %36 = or disjoint i64 %28, 1
  %37 = select i1 %35, i64 %36, i64 %29
  %38 = getelementptr inbounds double, ptr %0, i64 %37
  %39 = load double, ptr %38, align 8, !tbaa !16
  %40 = getelementptr inbounds double, ptr %0, i64 %27
  store double %39, ptr %40, align 8, !tbaa !16
  %41 = icmp slt i64 %37, %13
  br i1 %41, label %26, label %42, !llvm.loop !108

42:                                               ; preds = %26
  %43 = icmp sgt i64 %37, %22
  br i1 %43, label %44, label %54

44:                                               ; preds = %42, %51
  %45 = phi i64 [ %47, %51 ], [ %37, %42 ]
  %46 = add nsw i64 %45, -1
  %47 = sdiv i64 %46, 2
  %48 = getelementptr inbounds nuw double, ptr %0, i64 %47
  %49 = load double, ptr %48, align 8, !tbaa !16
  %50 = fcmp olt double %49, %24
  br i1 %50, label %51, label %54

51:                                               ; preds = %44
  %52 = getelementptr inbounds nuw double, ptr %0, i64 %45
  store double %49, ptr %52, align 8, !tbaa !16
  %53 = icmp sgt i64 %47, %22
  br i1 %53, label %44, label %54, !llvm.loop !109

54:                                               ; preds = %44, %51, %21, %42
  %55 = phi i64 [ %37, %42 ], [ %22, %21 ], [ %47, %51 ], [ %45, %44 ]
  %56 = getelementptr inbounds nuw double, ptr %0, i64 %55
  store double %24, ptr %56, align 8, !tbaa !16
  %57 = icmp eq i64 %22, 0
  %58 = add nsw i64 %22, -1
  br i1 %57, label %103, label %21, !llvm.loop !118

59:                                               ; preds = %17, %98
  %60 = phi i64 [ %102, %98 ], [ %11, %17 ]
  %61 = getelementptr inbounds nuw double, ptr %0, i64 %60
  %62 = load double, ptr %61, align 8, !tbaa !16
  %63 = icmp slt i64 %60, %13
  br i1 %63, label %64, label %80

64:                                               ; preds = %59, %64
  %65 = phi i64 [ %75, %64 ], [ %60, %59 ]
  %66 = shl i64 %65, 1
  %67 = add i64 %66, 2
  %68 = getelementptr inbounds double, ptr %0, i64 %67
  %69 = getelementptr double, ptr %0, i64 %66
  %70 = getelementptr i8, ptr %69, i64 8
  %71 = load double, ptr %68, align 8, !tbaa !16
  %72 = load double, ptr %70, align 8, !tbaa !16
  %73 = fcmp olt double %71, %72
  %74 = or disjoint i64 %66, 1
  %75 = select i1 %73, i64 %74, i64 %67
  %76 = getelementptr inbounds double, ptr %0, i64 %75
  %77 = load double, ptr %76, align 8, !tbaa !16
  %78 = getelementptr inbounds double, ptr %0, i64 %65
  store double %77, ptr %78, align 8, !tbaa !16
  %79 = icmp slt i64 %75, %13
  br i1 %79, label %64, label %80, !llvm.loop !108

80:                                               ; preds = %64, %59
  %81 = phi i64 [ %60, %59 ], [ %75, %64 ]
  %82 = icmp eq i64 %81, %16
  br i1 %82, label %83, label %85

83:                                               ; preds = %80
  %84 = load double, ptr %19, align 8, !tbaa !16
  store double %84, ptr %20, align 8, !tbaa !16
  br label %85

85:                                               ; preds = %83, %80
  %86 = phi i64 [ %18, %83 ], [ %81, %80 ]
  %87 = icmp sgt i64 %86, %60
  br i1 %87, label %88, label %98

88:                                               ; preds = %85, %95
  %89 = phi i64 [ %91, %95 ], [ %86, %85 ]
  %90 = add nsw i64 %89, -1
  %91 = sdiv i64 %90, 2
  %92 = getelementptr inbounds nuw double, ptr %0, i64 %91
  %93 = load double, ptr %92, align 8, !tbaa !16
  %94 = fcmp olt double %93, %62
  br i1 %94, label %95, label %98

95:                                               ; preds = %88
  %96 = getelementptr inbounds nuw double, ptr %0, i64 %89
  store double %93, ptr %96, align 8, !tbaa !16
  %97 = icmp sgt i64 %91, %60
  br i1 %97, label %88, label %98, !llvm.loop !109

98:                                               ; preds = %88, %95, %85
  %99 = phi i64 [ %86, %85 ], [ %91, %95 ], [ %89, %88 ]
  %100 = getelementptr inbounds nuw double, ptr %0, i64 %99
  store double %62, ptr %100, align 8, !tbaa !16
  %101 = icmp eq i64 %60, 0
  %102 = add nsw i64 %60, -1
  br i1 %101, label %103, label %59, !llvm.loop !118

103:                                              ; preds = %54, %98, %3
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.ctlz.i64(i64, i1 immarg) #10

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #4

; Function Attrs: cold noreturn
declare void @_ZSt20__throw_length_errorPKc(ptr noundef) local_unnamed_addr #14

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_(ptr %0, ptr %1, i64 noundef %2, i8 %3) local_unnamed_addr #2 comdat {
  %5 = ptrtoint ptr %0 to i64
  %6 = ptrtoint ptr %1 to i64
  %7 = sub i64 %6, %5
  %8 = ashr exact i64 %7, 3
  %9 = icmp sgt i64 %8, 16
  br i1 %9, label %10, label %126

10:                                               ; preds = %4
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8
  br label %12

12:                                               ; preds = %10, %121
  %13 = phi i64 [ %8, %10 ], [ %124, %121 ]
  %14 = phi i64 [ %2, %10 ], [ %77, %121 ]
  %15 = phi ptr [ %1, %10 ], [ %109, %121 ]
  %16 = icmp eq i64 %14, 0
  br i1 %16, label %17, label %76

17:                                               ; preds = %12
  tail call void @_ZSt13__heap_selectIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_T0_(ptr %0, ptr %15, ptr %15, i8 undef)
  br label %18

18:                                               ; preds = %17, %72
  %19 = phi ptr [ %20, %72 ], [ %15, %17 ]
  %20 = getelementptr inbounds i8, ptr %19, i64 -8
  %21 = load double, ptr %20, align 8, !tbaa !16
  %22 = load double, ptr %0, align 8, !tbaa !16
  store double %22, ptr %20, align 8, !tbaa !16
  %23 = ptrtoint ptr %20 to i64
  %24 = sub i64 %23, %5
  %25 = ashr exact i64 %24, 3
  %26 = add nsw i64 %25, -1
  %27 = sdiv i64 %26, 2
  %28 = icmp sgt i64 %25, 2
  br i1 %28, label %29, label %44

29:                                               ; preds = %18, %29
  %30 = phi i64 [ %39, %29 ], [ 0, %18 ]
  %31 = shl i64 %30, 1
  %32 = add i64 %31, 2
  %33 = getelementptr inbounds double, ptr %0, i64 %32
  %34 = or disjoint i64 %31, 1
  %35 = getelementptr inbounds double, ptr %0, i64 %34
  %36 = load double, ptr %33, align 8, !tbaa !16
  %37 = load double, ptr %35, align 8, !tbaa !16
  %38 = fcmp olt double %36, %37
  %39 = select i1 %38, i64 %34, i64 %32
  %40 = getelementptr inbounds double, ptr %0, i64 %39
  %41 = load double, ptr %40, align 8, !tbaa !16
  %42 = getelementptr inbounds double, ptr %0, i64 %30
  store double %41, ptr %42, align 8, !tbaa !16
  %43 = icmp slt i64 %39, %27
  br i1 %43, label %29, label %44, !llvm.loop !119

44:                                               ; preds = %29, %18
  %45 = phi i64 [ 0, %18 ], [ %39, %29 ]
  %46 = and i64 %24, 8
  %47 = icmp eq i64 %46, 0
  br i1 %47, label %48, label %58

48:                                               ; preds = %44
  %49 = add nsw i64 %25, -2
  %50 = ashr exact i64 %49, 1
  %51 = icmp eq i64 %45, %50
  br i1 %51, label %52, label %58

52:                                               ; preds = %48
  %53 = shl nuw nsw i64 %45, 1
  %54 = or disjoint i64 %53, 1
  %55 = getelementptr inbounds nuw double, ptr %0, i64 %54
  %56 = load double, ptr %55, align 8, !tbaa !16
  %57 = getelementptr inbounds double, ptr %0, i64 %45
  store double %56, ptr %57, align 8, !tbaa !16
  br label %60

58:                                               ; preds = %48, %44
  %59 = icmp eq i64 %45, 0
  br i1 %59, label %72, label %60

60:                                               ; preds = %58, %52
  %61 = phi i64 [ %45, %58 ], [ %54, %52 ]
  br label %62

62:                                               ; preds = %60, %69
  %63 = phi i64 [ %65, %69 ], [ %61, %60 ]
  %64 = add nsw i64 %63, -1
  %65 = lshr i64 %64, 1
  %66 = getelementptr inbounds nuw double, ptr %0, i64 %65
  %67 = load double, ptr %66, align 8, !tbaa !16
  %68 = fcmp olt double %67, %21
  br i1 %68, label %69, label %72

69:                                               ; preds = %62
  %70 = getelementptr inbounds double, ptr %0, i64 %63
  store double %67, ptr %70, align 8, !tbaa !16
  %71 = icmp ult i64 %64, 2
  br i1 %71, label %72, label %62, !llvm.loop !120

72:                                               ; preds = %69, %62, %58
  %73 = phi i64 [ 0, %58 ], [ %63, %62 ], [ 0, %69 ]
  %74 = getelementptr inbounds double, ptr %0, i64 %73
  store double %21, ptr %74, align 8, !tbaa !16
  %75 = icmp sgt i64 %24, 8
  br i1 %75, label %18, label %126, !llvm.loop !121

76:                                               ; preds = %12
  %77 = add nsw i64 %14, -1
  %78 = lshr i64 %13, 1
  %79 = getelementptr inbounds nuw double, ptr %0, i64 %78
  %80 = getelementptr inbounds i8, ptr %15, i64 -8
  %81 = load double, ptr %11, align 8, !tbaa !16
  %82 = load double, ptr %79, align 8, !tbaa !16
  %83 = fcmp olt double %81, %82
  %84 = load double, ptr %80, align 8, !tbaa !16
  br i1 %83, label %85, label %94

85:                                               ; preds = %76
  %86 = fcmp olt double %82, %84
  br i1 %86, label %87, label %89

87:                                               ; preds = %85
  %88 = load double, ptr %0, align 8, !tbaa !16
  store double %82, ptr %0, align 8, !tbaa !16
  store double %88, ptr %79, align 8, !tbaa !16
  br label %103

89:                                               ; preds = %85
  %90 = fcmp olt double %81, %84
  %91 = load double, ptr %0, align 8, !tbaa !16
  br i1 %90, label %92, label %93

92:                                               ; preds = %89
  store double %84, ptr %0, align 8, !tbaa !16
  store double %91, ptr %80, align 8, !tbaa !16
  br label %103

93:                                               ; preds = %89
  store double %81, ptr %0, align 8, !tbaa !16
  store double %91, ptr %11, align 8, !tbaa !16
  br label %103

94:                                               ; preds = %76
  %95 = fcmp olt double %81, %84
  br i1 %95, label %96, label %98

96:                                               ; preds = %94
  %97 = load double, ptr %0, align 8, !tbaa !16
  store double %81, ptr %0, align 8, !tbaa !16
  store double %97, ptr %11, align 8, !tbaa !16
  br label %103

98:                                               ; preds = %94
  %99 = fcmp olt double %82, %84
  %100 = load double, ptr %0, align 8, !tbaa !16
  br i1 %99, label %101, label %102

101:                                              ; preds = %98
  store double %84, ptr %0, align 8, !tbaa !16
  store double %100, ptr %80, align 8, !tbaa !16
  br label %103

102:                                              ; preds = %98
  store double %82, ptr %0, align 8, !tbaa !16
  store double %100, ptr %79, align 8, !tbaa !16
  br label %103

103:                                              ; preds = %102, %101, %96, %93, %92, %87
  br label %104

104:                                              ; preds = %103, %120
  %105 = phi ptr [ %115, %120 ], [ %15, %103 ]
  %106 = phi ptr [ %112, %120 ], [ %11, %103 ]
  %107 = load double, ptr %0, align 8, !tbaa !16
  br label %108

108:                                              ; preds = %108, %104
  %109 = phi ptr [ %106, %104 ], [ %112, %108 ]
  %110 = load double, ptr %109, align 8, !tbaa !16
  %111 = fcmp olt double %110, %107
  %112 = getelementptr inbounds nuw i8, ptr %109, i64 8
  br i1 %111, label %108, label %113, !llvm.loop !122

113:                                              ; preds = %108, %113
  %114 = phi ptr [ %115, %113 ], [ %105, %108 ]
  %115 = getelementptr inbounds i8, ptr %114, i64 -8
  %116 = load double, ptr %115, align 8, !tbaa !16
  %117 = fcmp olt double %107, %116
  br i1 %117, label %113, label %118, !llvm.loop !123

118:                                              ; preds = %113
  %119 = icmp ult ptr %109, %115
  br i1 %119, label %120, label %121

120:                                              ; preds = %118
  store double %116, ptr %109, align 8, !tbaa !16
  store double %110, ptr %115, align 8, !tbaa !16
  br label %104, !llvm.loop !124

121:                                              ; preds = %118
  tail call void @_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_(ptr nonnull %109, ptr %15, i64 noundef %77, i8 undef)
  %122 = ptrtoint ptr %109 to i64
  %123 = sub i64 %122, %5
  %124 = ashr exact i64 %123, 3
  %125 = icmp sgt i64 %124, 16
  br i1 %125, label %12, label %126, !llvm.loop !125

126:                                              ; preds = %121, %72, %4
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_(ptr %0, ptr %1, i8 %2) local_unnamed_addr #2 comdat {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = icmp sgt i64 %6, 128
  br i1 %7, label %8, label %56

8:                                                ; preds = %3
  %9 = getelementptr i8, ptr %0, i64 8
  br label %10

10:                                               ; preds = %32, %8
  %11 = phi i64 [ 8, %8 ], [ %34, %32 ]
  %12 = phi ptr [ %0, %8 ], [ %13, %32 ]
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 %11
  %14 = load double, ptr %13, align 8, !tbaa !16
  %15 = load double, ptr %0, align 8, !tbaa !16
  %16 = fcmp olt double %14, %15
  br i1 %16, label %17, label %22

17:                                               ; preds = %10
  %18 = icmp samesign ugt i64 %11, 8
  br i1 %18, label %19, label %20, !prof !15

19:                                               ; preds = %17
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %9, ptr noundef nonnull align 8 dereferenceable(1) %0, i64 %11, i1 false)
  br label %32

20:                                               ; preds = %17
  %21 = getelementptr inbounds nuw i8, ptr %12, i64 8
  store double %15, ptr %21, align 8, !tbaa !16
  br label %32

22:                                               ; preds = %10
  %23 = load double, ptr %12, align 8, !tbaa !16
  %24 = fcmp olt double %14, %23
  br i1 %24, label %25, label %32

25:                                               ; preds = %22, %25
  %26 = phi double [ %30, %25 ], [ %23, %22 ]
  %27 = phi ptr [ %29, %25 ], [ %12, %22 ]
  %28 = phi ptr [ %27, %25 ], [ %13, %22 ]
  store double %26, ptr %28, align 8, !tbaa !16
  %29 = getelementptr inbounds i8, ptr %27, i64 -8
  %30 = load double, ptr %29, align 8, !tbaa !16
  %31 = fcmp olt double %14, %30
  br i1 %31, label %25, label %32, !llvm.loop !126

32:                                               ; preds = %25, %22, %20, %19
  %33 = phi ptr [ %0, %19 ], [ %0, %20 ], [ %13, %22 ], [ %27, %25 ]
  store double %14, ptr %33, align 8, !tbaa !16
  %34 = add nuw nsw i64 %11, 8
  %35 = icmp eq i64 %34, 128
  br i1 %35, label %36, label %10, !llvm.loop !127

36:                                               ; preds = %32
  %37 = getelementptr inbounds nuw i8, ptr %0, i64 128
  %38 = icmp eq ptr %37, %1
  br i1 %38, label %94, label %39

39:                                               ; preds = %36, %52
  %40 = phi ptr [ %54, %52 ], [ %37, %36 ]
  %41 = load double, ptr %40, align 8, !tbaa !16
  %42 = getelementptr inbounds i8, ptr %40, i64 -8
  %43 = load double, ptr %42, align 8, !tbaa !16
  %44 = fcmp olt double %41, %43
  br i1 %44, label %45, label %52

45:                                               ; preds = %39, %45
  %46 = phi double [ %50, %45 ], [ %43, %39 ]
  %47 = phi ptr [ %49, %45 ], [ %42, %39 ]
  %48 = phi ptr [ %47, %45 ], [ %40, %39 ]
  store double %46, ptr %48, align 8, !tbaa !16
  %49 = getelementptr inbounds i8, ptr %47, i64 -8
  %50 = load double, ptr %49, align 8, !tbaa !16
  %51 = fcmp olt double %41, %50
  br i1 %51, label %45, label %52, !llvm.loop !126

52:                                               ; preds = %45, %39
  %53 = phi ptr [ %40, %39 ], [ %47, %45 ]
  store double %41, ptr %53, align 8, !tbaa !16
  %54 = getelementptr inbounds nuw i8, ptr %40, i64 8
  %55 = icmp eq ptr %54, %1
  br i1 %55, label %94, label %39, !llvm.loop !128

56:                                               ; preds = %3
  %57 = icmp eq ptr %0, %1
  %58 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %59 = icmp eq ptr %58, %1
  %60 = select i1 %57, i1 true, i1 %59
  br i1 %60, label %94, label %61

61:                                               ; preds = %56, %90
  %62 = phi ptr [ %92, %90 ], [ %58, %56 ]
  %63 = phi ptr [ %62, %90 ], [ %0, %56 ]
  %64 = load double, ptr %62, align 8, !tbaa !16
  %65 = load double, ptr %0, align 8, !tbaa !16
  %66 = fcmp olt double %64, %65
  br i1 %66, label %67, label %80

67:                                               ; preds = %61
  %68 = ptrtoint ptr %62 to i64
  %69 = sub i64 %68, %5
  %70 = ashr exact i64 %69, 3
  %71 = icmp sgt i64 %70, 1
  br i1 %71, label %72, label %76, !prof !15

72:                                               ; preds = %67
  %73 = getelementptr inbounds nuw i8, ptr %63, i64 16
  %74 = sub nsw i64 0, %70
  %75 = getelementptr inbounds double, ptr %73, i64 %74
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %75, ptr noundef nonnull align 8 dereferenceable(1) %0, i64 %69, i1 false)
  br label %90

76:                                               ; preds = %67
  %77 = icmp eq i64 %69, 8
  br i1 %77, label %78, label %90

78:                                               ; preds = %76
  %79 = getelementptr inbounds nuw i8, ptr %63, i64 8
  store double %65, ptr %79, align 8, !tbaa !16
  br label %90

80:                                               ; preds = %61
  %81 = load double, ptr %63, align 8, !tbaa !16
  %82 = fcmp olt double %64, %81
  br i1 %82, label %83, label %90

83:                                               ; preds = %80, %83
  %84 = phi double [ %88, %83 ], [ %81, %80 ]
  %85 = phi ptr [ %87, %83 ], [ %63, %80 ]
  %86 = phi ptr [ %85, %83 ], [ %62, %80 ]
  store double %84, ptr %86, align 8, !tbaa !16
  %87 = getelementptr inbounds i8, ptr %85, i64 -8
  %88 = load double, ptr %87, align 8, !tbaa !16
  %89 = fcmp olt double %64, %88
  br i1 %89, label %83, label %90, !llvm.loop !126

90:                                               ; preds = %83, %80, %78, %76, %72
  %91 = phi ptr [ %0, %72 ], [ %0, %76 ], [ %0, %78 ], [ %62, %80 ], [ %85, %83 ]
  store double %64, ptr %91, align 8, !tbaa !16
  %92 = getelementptr inbounds nuw i8, ptr %62, i64 8
  %93 = icmp eq ptr %92, %1
  br i1 %93, label %94, label %61, !llvm.loop !127

94:                                               ; preds = %90, %52, %56, %36
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt13__heap_selectIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_T0_(ptr %0, ptr %1, ptr %2, i8 %3) local_unnamed_addr #2 comdat {
  %5 = alloca %"struct.__gnu_cxx::__ops::_Iter_less_iter", align 4
  %6 = freeze ptr %0
  %7 = freeze ptr %1
  store i8 %3, ptr %5, align 4
  call void @_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_(ptr %6, ptr %7, ptr noundef nonnull align 1 dereferenceable(1) %5)
  %8 = icmp ult ptr %7, %2
  br i1 %8, label %9, label %102

9:                                                ; preds = %4
  %10 = ptrtoint ptr %7 to i64
  %11 = ptrtoint ptr %6 to i64
  %12 = sub i64 %10, %11
  %13 = ashr i64 %12, 3
  %14 = add nsw i64 %13, -1
  %15 = sdiv i64 %14, 2
  %16 = icmp sgt i64 %13, 2
  %17 = and i64 %12, 8
  %18 = icmp eq i64 %17, 0
  %19 = add nsw i64 %13, -2
  %20 = ashr exact i64 %19, 1
  br i1 %16, label %21, label %71

21:                                               ; preds = %9
  %22 = or disjoint i64 %19, 1
  %23 = getelementptr inbounds nuw double, ptr %6, i64 %22
  %24 = getelementptr inbounds double, ptr %6, i64 %20
  br label %25

25:                                               ; preds = %21, %65
  %26 = phi ptr [ %66, %65 ], [ %7, %21 ]
  %27 = load double, ptr %26, align 8, !tbaa !16
  %28 = load double, ptr %6, align 8, !tbaa !16
  %29 = fcmp olt double %27, %28
  br i1 %29, label %30, label %65

30:                                               ; preds = %25
  store double %28, ptr %26, align 8, !tbaa !16
  br label %31

31:                                               ; preds = %30, %31
  %32 = phi i64 [ %41, %31 ], [ 0, %30 ]
  %33 = shl i64 %32, 1
  %34 = add i64 %33, 2
  %35 = getelementptr inbounds double, ptr %6, i64 %34
  %36 = or disjoint i64 %33, 1
  %37 = getelementptr inbounds double, ptr %6, i64 %36
  %38 = load double, ptr %35, align 8, !tbaa !16
  %39 = load double, ptr %37, align 8, !tbaa !16
  %40 = fcmp olt double %38, %39
  %41 = select i1 %40, i64 %36, i64 %34
  %42 = getelementptr inbounds double, ptr %6, i64 %41
  %43 = load double, ptr %42, align 8, !tbaa !16
  %44 = getelementptr inbounds double, ptr %6, i64 %32
  store double %43, ptr %44, align 8, !tbaa !16
  %45 = icmp slt i64 %41, %15
  br i1 %45, label %31, label %68, !llvm.loop !119

46:                                               ; preds = %68
  %47 = icmp eq i64 %41, 0
  br i1 %47, label %62, label %50

48:                                               ; preds = %68
  %49 = load double, ptr %23, align 8, !tbaa !16
  store double %49, ptr %24, align 8, !tbaa !16
  br label %50

50:                                               ; preds = %48, %46
  %51 = phi i64 [ %41, %46 ], [ %22, %48 ]
  br label %52

52:                                               ; preds = %50, %59
  %53 = phi i64 [ %55, %59 ], [ %51, %50 ]
  %54 = add nsw i64 %53, -1
  %55 = lshr i64 %54, 1
  %56 = getelementptr inbounds nuw double, ptr %6, i64 %55
  %57 = load double, ptr %56, align 8, !tbaa !16
  %58 = fcmp olt double %57, %27
  br i1 %58, label %59, label %62

59:                                               ; preds = %52
  %60 = getelementptr inbounds double, ptr %6, i64 %53
  store double %57, ptr %60, align 8, !tbaa !16
  %61 = icmp ult i64 %54, 2
  br i1 %61, label %62, label %52, !llvm.loop !120

62:                                               ; preds = %52, %59, %46
  %63 = phi i64 [ 0, %46 ], [ %53, %52 ], [ 0, %59 ]
  %64 = getelementptr inbounds double, ptr %6, i64 %63
  store double %27, ptr %64, align 8, !tbaa !16
  br label %65

65:                                               ; preds = %62, %25
  %66 = getelementptr inbounds nuw i8, ptr %26, i64 8
  %67 = icmp ult ptr %66, %2
  br i1 %67, label %25, label %102, !llvm.loop !129

68:                                               ; preds = %31
  %69 = icmp eq i64 %41, %20
  %70 = select i1 %18, i1 %69, i1 false
  br i1 %70, label %48, label %46

71:                                               ; preds = %9
  %72 = getelementptr inbounds nuw i8, ptr %6, i64 8
  br i1 %18, label %75, label %73

73:                                               ; preds = %71
  %74 = load double, ptr %6, align 8, !tbaa !16
  br label %103

75:                                               ; preds = %71
  %76 = icmp eq i64 %19, 0
  br i1 %76, label %79, label %77

77:                                               ; preds = %75
  %78 = load double, ptr %6, align 8, !tbaa !16
  br label %92

79:                                               ; preds = %75, %89
  %80 = phi ptr [ %90, %89 ], [ %7, %75 ]
  %81 = load double, ptr %80, align 8, !tbaa !16
  %82 = load double, ptr %6, align 8, !tbaa !16
  %83 = fcmp olt double %81, %82
  br i1 %83, label %84, label %89

84:                                               ; preds = %79
  store double %82, ptr %80, align 8, !tbaa !16
  %85 = load double, ptr %72, align 8, !tbaa !16
  store double %85, ptr %6, align 8, !tbaa !16
  %86 = fcmp uge double %85, %81
  %87 = zext i1 %86 to i64
  %88 = getelementptr inbounds nuw double, ptr %6, i64 %87
  store double %81, ptr %88, align 8, !tbaa !16
  br label %89

89:                                               ; preds = %84, %79
  %90 = getelementptr inbounds nuw i8, ptr %80, i64 8
  %91 = icmp ult ptr %90, %2
  br i1 %91, label %79, label %102, !llvm.loop !129

92:                                               ; preds = %77, %98
  %93 = phi double [ %99, %98 ], [ %78, %77 ]
  %94 = phi ptr [ %100, %98 ], [ %7, %77 ]
  %95 = load double, ptr %94, align 8, !tbaa !16
  %96 = fcmp olt double %95, %93
  br i1 %96, label %97, label %98

97:                                               ; preds = %92
  store double %93, ptr %94, align 8, !tbaa !16
  store double %95, ptr %6, align 8, !tbaa !16
  br label %98

98:                                               ; preds = %97, %92
  %99 = phi double [ %95, %97 ], [ %93, %92 ]
  %100 = getelementptr inbounds nuw i8, ptr %94, i64 8
  %101 = icmp ult ptr %100, %2
  br i1 %101, label %92, label %102, !llvm.loop !129

102:                                              ; preds = %109, %98, %89, %65, %4
  ret void

103:                                              ; preds = %73, %109
  %104 = phi double [ %110, %109 ], [ %74, %73 ]
  %105 = phi ptr [ %111, %109 ], [ %7, %73 ]
  %106 = load double, ptr %105, align 8, !tbaa !16
  %107 = fcmp olt double %106, %104
  br i1 %107, label %108, label %109

108:                                              ; preds = %103
  store double %104, ptr %105, align 8, !tbaa !16
  store double %106, ptr %6, align 8, !tbaa !16
  br label %109

109:                                              ; preds = %103, %108
  %110 = phi double [ %104, %103 ], [ %106, %108 ]
  %111 = getelementptr inbounds nuw i8, ptr %105, i64 8
  %112 = icmp ult ptr %111, %2
  br i1 %112, label %103, label %102, !llvm.loop !129
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_(ptr %0, ptr %1, ptr noundef nonnull align 1 dereferenceable(1) %2) local_unnamed_addr #2 comdat {
  %4 = freeze ptr %0
  %5 = freeze ptr %1
  %6 = ptrtoint ptr %5 to i64
  %7 = ptrtoint ptr %4 to i64
  %8 = sub i64 %6, %7
  %9 = ashr exact i64 %8, 3
  %10 = icmp slt i64 %9, 2
  br i1 %10, label %103, label %11

11:                                               ; preds = %3
  %12 = add nsw i64 %9, -2
  %13 = lshr i64 %12, 1
  %14 = add nsw i64 %9, -1
  %15 = lshr i64 %14, 1
  %16 = and i64 %8, 8
  %17 = icmp eq i64 %16, 0
  %18 = lshr exact i64 %12, 1
  br i1 %17, label %19, label %23

19:                                               ; preds = %11
  %20 = or disjoint i64 %12, 1
  %21 = getelementptr inbounds nuw double, ptr %4, i64 %20
  %22 = getelementptr inbounds nuw double, ptr %4, i64 %18
  br label %60

23:                                               ; preds = %11, %55
  %24 = phi i64 [ %59, %55 ], [ %13, %11 ]
  %25 = getelementptr inbounds double, ptr %4, i64 %24
  %26 = load double, ptr %25, align 8, !tbaa !16
  %27 = icmp slt i64 %24, %15
  br i1 %27, label %28, label %55

28:                                               ; preds = %23, %28
  %29 = phi i64 [ %38, %28 ], [ %24, %23 ]
  %30 = shl i64 %29, 1
  %31 = add i64 %30, 2
  %32 = getelementptr inbounds double, ptr %4, i64 %31
  %33 = or disjoint i64 %30, 1
  %34 = getelementptr inbounds double, ptr %4, i64 %33
  %35 = load double, ptr %32, align 8, !tbaa !16
  %36 = load double, ptr %34, align 8, !tbaa !16
  %37 = fcmp olt double %35, %36
  %38 = select i1 %37, i64 %33, i64 %31
  %39 = getelementptr inbounds double, ptr %4, i64 %38
  %40 = load double, ptr %39, align 8, !tbaa !16
  %41 = getelementptr inbounds double, ptr %4, i64 %29
  store double %40, ptr %41, align 8, !tbaa !16
  %42 = icmp slt i64 %38, %15
  br i1 %42, label %28, label %43, !llvm.loop !119

43:                                               ; preds = %28
  %44 = icmp sgt i64 %38, %24
  br i1 %44, label %45, label %55

45:                                               ; preds = %43, %52
  %46 = phi i64 [ %48, %52 ], [ %38, %43 ]
  %47 = add nsw i64 %46, -1
  %48 = sdiv i64 %47, 2
  %49 = getelementptr inbounds nuw double, ptr %4, i64 %48
  %50 = load double, ptr %49, align 8, !tbaa !16
  %51 = fcmp olt double %50, %26
  br i1 %51, label %52, label %55

52:                                               ; preds = %45
  %53 = getelementptr inbounds nuw double, ptr %4, i64 %46
  store double %50, ptr %53, align 8, !tbaa !16
  %54 = icmp sgt i64 %48, %24
  br i1 %54, label %45, label %55, !llvm.loop !120

55:                                               ; preds = %45, %52, %23, %43
  %56 = phi i64 [ %38, %43 ], [ %24, %23 ], [ %48, %52 ], [ %46, %45 ]
  %57 = getelementptr inbounds nuw double, ptr %4, i64 %56
  store double %26, ptr %57, align 8, !tbaa !16
  %58 = icmp eq i64 %24, 0
  %59 = add nsw i64 %24, -1
  br i1 %58, label %103, label %23, !llvm.loop !130

60:                                               ; preds = %19, %98
  %61 = phi i64 [ %102, %98 ], [ %13, %19 ]
  %62 = getelementptr inbounds double, ptr %4, i64 %61
  %63 = load double, ptr %62, align 8, !tbaa !16
  %64 = icmp slt i64 %61, %15
  br i1 %64, label %65, label %80

65:                                               ; preds = %60, %65
  %66 = phi i64 [ %75, %65 ], [ %61, %60 ]
  %67 = shl i64 %66, 1
  %68 = add i64 %67, 2
  %69 = getelementptr inbounds double, ptr %4, i64 %68
  %70 = or disjoint i64 %67, 1
  %71 = getelementptr inbounds double, ptr %4, i64 %70
  %72 = load double, ptr %69, align 8, !tbaa !16
  %73 = load double, ptr %71, align 8, !tbaa !16
  %74 = fcmp olt double %72, %73
  %75 = select i1 %74, i64 %70, i64 %68
  %76 = getelementptr inbounds double, ptr %4, i64 %75
  %77 = load double, ptr %76, align 8, !tbaa !16
  %78 = getelementptr inbounds double, ptr %4, i64 %66
  store double %77, ptr %78, align 8, !tbaa !16
  %79 = icmp slt i64 %75, %15
  br i1 %79, label %65, label %80, !llvm.loop !119

80:                                               ; preds = %65, %60
  %81 = phi i64 [ %61, %60 ], [ %75, %65 ]
  %82 = icmp eq i64 %81, %18
  br i1 %82, label %83, label %85

83:                                               ; preds = %80
  %84 = load double, ptr %21, align 8, !tbaa !16
  store double %84, ptr %22, align 8, !tbaa !16
  br label %85

85:                                               ; preds = %83, %80
  %86 = phi i64 [ %20, %83 ], [ %81, %80 ]
  %87 = icmp sgt i64 %86, %61
  br i1 %87, label %88, label %98

88:                                               ; preds = %85, %95
  %89 = phi i64 [ %91, %95 ], [ %86, %85 ]
  %90 = add nsw i64 %89, -1
  %91 = sdiv i64 %90, 2
  %92 = getelementptr inbounds nuw double, ptr %4, i64 %91
  %93 = load double, ptr %92, align 8, !tbaa !16
  %94 = fcmp olt double %93, %63
  br i1 %94, label %95, label %98

95:                                               ; preds = %88
  %96 = getelementptr inbounds nuw double, ptr %4, i64 %89
  store double %93, ptr %96, align 8, !tbaa !16
  %97 = icmp sgt i64 %91, %61
  br i1 %97, label %88, label %98, !llvm.loop !120

98:                                               ; preds = %88, %95, %85
  %99 = phi i64 [ %86, %85 ], [ %91, %95 ], [ %89, %88 ]
  %100 = getelementptr inbounds nuw double, ptr %4, i64 %99
  store double %63, ptr %100, align 8, !tbaa !16
  %101 = icmp eq i64 %61, 0
  %102 = add nsw i64 %61, -1
  br i1 %101, label %103, label %60, !llvm.loop !130

103:                                              ; preds = %55, %98, %3
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNSt5dequeIdSaIdEE18_M_fill_initializeERKd(ptr noundef nonnull align 8 dereferenceable(80) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) local_unnamed_addr #2 comdat personality ptr @__gxx_personality_v0 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %4 = load ptr, ptr %3, align 8, !tbaa !69
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %6 = load ptr, ptr %5, align 8, !tbaa !70
  %7 = icmp ult ptr %4, %6
  br i1 %7, label %8, label %30

8:                                                ; preds = %2, %8
  %9 = phi ptr [ %28, %8 ], [ %4, %2 ]
  %10 = load ptr, ptr %9, align 8, !tbaa !40
  %11 = load <1 x double>, ptr %1, align 8
  %12 = shufflevector <1 x double> %11, <1 x double> poison, <4 x i32> zeroinitializer
  store <4 x double> %12, ptr %10, align 8, !tbaa !16
  %13 = getelementptr inbounds nuw i8, ptr %10, i64 32
  store <4 x double> %12, ptr %13, align 8, !tbaa !16
  %14 = getelementptr inbounds nuw i8, ptr %10, i64 64
  store <4 x double> %12, ptr %14, align 8, !tbaa !16
  %15 = getelementptr inbounds nuw i8, ptr %10, i64 96
  store <4 x double> %12, ptr %15, align 8, !tbaa !16
  %16 = getelementptr inbounds nuw i8, ptr %10, i64 128
  store <4 x double> %12, ptr %16, align 8, !tbaa !16
  %17 = getelementptr inbounds nuw i8, ptr %10, i64 160
  store <4 x double> %12, ptr %17, align 8, !tbaa !16
  %18 = getelementptr inbounds nuw i8, ptr %10, i64 192
  store <4 x double> %12, ptr %18, align 8, !tbaa !16
  %19 = getelementptr inbounds nuw i8, ptr %10, i64 224
  store <4 x double> %12, ptr %19, align 8, !tbaa !16
  %20 = getelementptr inbounds nuw i8, ptr %10, i64 256
  store <4 x double> %12, ptr %20, align 8, !tbaa !16
  %21 = getelementptr inbounds nuw i8, ptr %10, i64 288
  store <4 x double> %12, ptr %21, align 8, !tbaa !16
  %22 = getelementptr inbounds nuw i8, ptr %10, i64 320
  store <4 x double> %12, ptr %22, align 8, !tbaa !16
  %23 = getelementptr inbounds nuw i8, ptr %10, i64 352
  store <4 x double> %12, ptr %23, align 8, !tbaa !16
  %24 = getelementptr inbounds nuw i8, ptr %10, i64 384
  store <4 x double> %12, ptr %24, align 8, !tbaa !16
  %25 = getelementptr inbounds nuw i8, ptr %10, i64 416
  store <4 x double> %12, ptr %25, align 8, !tbaa !16
  %26 = getelementptr inbounds nuw i8, ptr %10, i64 448
  store <4 x double> %12, ptr %26, align 8, !tbaa !16
  %27 = getelementptr inbounds nuw i8, ptr %10, i64 480
  store <4 x double> %12, ptr %27, align 8, !tbaa !16
  %28 = getelementptr inbounds nuw i8, ptr %9, i64 8
  %29 = icmp ult ptr %28, %6
  br i1 %29, label %8, label %30, !llvm.loop !131

30:                                               ; preds = %8, %2
  %31 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %32 = getelementptr inbounds nuw i8, ptr %0, i64 56
  %33 = load ptr, ptr %32, align 8, !tbaa !132
  %34 = load ptr, ptr %31, align 8, !tbaa !133
  %35 = load double, ptr %1, align 8, !tbaa !16
  %36 = icmp eq ptr %33, %34
  br i1 %36, label %66, label %37

37:                                               ; preds = %30
  %38 = ptrtoint ptr %34 to i64
  %39 = ptrtoint ptr %33 to i64
  %40 = add i64 %38, -8
  %41 = sub i64 %40, %39
  %42 = lshr i64 %41, 3
  %43 = add nuw nsw i64 %42, 1
  %44 = icmp ult i64 %41, 24
  br i1 %44, label %60, label %45

45:                                               ; preds = %37
  %46 = and i64 %43, 4611686018427387900
  %47 = shl i64 %46, 3
  %48 = getelementptr i8, ptr %33, i64 %47
  %49 = insertelement <2 x double> poison, double %35, i64 0
  %50 = shufflevector <2 x double> %49, <2 x double> poison, <2 x i32> zeroinitializer
  br label %51

51:                                               ; preds = %51, %45
  %52 = phi i64 [ 0, %45 ], [ %56, %51 ]
  %53 = shl i64 %52, 3
  %54 = getelementptr i8, ptr %33, i64 %53
  %55 = getelementptr i8, ptr %54, i64 16
  store <2 x double> %50, ptr %54, align 8, !tbaa !16
  store <2 x double> %50, ptr %55, align 8, !tbaa !16
  %56 = add nuw i64 %52, 4
  %57 = icmp eq i64 %56, %46
  br i1 %57, label %58, label %51, !llvm.loop !134

58:                                               ; preds = %51
  %59 = icmp eq i64 %43, %46
  br i1 %59, label %66, label %60

60:                                               ; preds = %37, %58
  %61 = phi ptr [ %33, %37 ], [ %48, %58 ]
  br label %62

62:                                               ; preds = %60, %62
  %63 = phi ptr [ %64, %62 ], [ %61, %60 ]
  store double %35, ptr %63, align 8, !tbaa !16
  %64 = getelementptr inbounds nuw i8, ptr %63, i64 8
  %65 = icmp eq ptr %64, %34
  br i1 %65, label %66, label %62, !llvm.loop !137

66:                                               ; preds = %62, %58, %30
  ret void
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZNSt11_Deque_baseIdSaIdEED2Ev(ptr noundef nonnull align 8 dereferenceable(80) %0) unnamed_addr #0 comdat personality ptr @__gxx_personality_v0 {
  %2 = load ptr, ptr %0, align 8, !tbaa !64
  %3 = icmp eq ptr %2, null
  br i1 %3, label %23, label %4

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %6 = load ptr, ptr %5, align 8, !tbaa !69
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %8 = load ptr, ptr %7, align 8, !tbaa !70
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %10 = icmp ult ptr %6, %9
  br i1 %10, label %11, label %18

11:                                               ; preds = %4, %11
  %12 = phi ptr [ %14, %11 ], [ %6, %4 ]
  %13 = load ptr, ptr %12, align 8, !tbaa !40
  tail call void @_ZdlPvm(ptr noundef %13, i64 noundef 512) #22
  %14 = getelementptr inbounds nuw i8, ptr %12, i64 8
  %15 = icmp ult ptr %12, %8
  br i1 %15, label %11, label %16, !llvm.loop !67

16:                                               ; preds = %11
  %17 = load ptr, ptr %0, align 8, !tbaa !64
  br label %18

18:                                               ; preds = %16, %4
  %19 = phi ptr [ %17, %16 ], [ %2, %4 ]
  %20 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %21 = load i64, ptr %20, align 8, !tbaa !68
  %22 = shl i64 %21, 3
  tail call void @_ZdlPvm(ptr noundef %19, i64 noundef %22) #22
  br label %23

23:                                               ; preds = %18, %1
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNSt11_Deque_baseIdSaIdEE17_M_initialize_mapEm(ptr noundef nonnull align 8 dereferenceable(80) %0, i64 noundef %1) local_unnamed_addr #2 comdat personality ptr @__gxx_personality_v0 {
  %3 = lshr i64 %1, 6
  %4 = add nuw nsw i64 %3, 1
  %5 = tail call i64 @llvm.umax.i64(i64 %3, i64 5)
  %6 = add nuw nsw i64 %5, 3
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %6, ptr %7, align 8, !tbaa !68
  %8 = shl nuw nsw i64 %6, 3
  %9 = tail call noalias noundef nonnull ptr @_Znwm(i64 noundef %8) #23
  store ptr %9, ptr %0, align 8, !tbaa !64
  %10 = sub nsw i64 %6, %4
  %11 = lshr i64 %10, 1
  %12 = getelementptr inbounds nuw ptr, ptr %9, i64 %11
  %13 = shl nuw nsw i64 %4, 3
  %14 = getelementptr inbounds nuw i8, ptr %12, i64 %13
  br label %15

15:                                               ; preds = %2, %18
  %16 = phi ptr [ %19, %18 ], [ %12, %2 ]
  %17 = invoke noalias noundef nonnull dereferenceable(512) ptr @_Znwm(i64 noundef 512) #23
          to label %18 unwind label %21

18:                                               ; preds = %15
  store ptr %17, ptr %16, align 8, !tbaa !40
  %19 = getelementptr inbounds nuw i8, ptr %16, i64 8
  %20 = icmp ult ptr %19, %14
  br i1 %20, label %15, label %47, !llvm.loop !138

21:                                               ; preds = %15
  %22 = landingpad { ptr, i32 }
          catch ptr null
  %23 = extractvalue { ptr, i32 } %22, 0
  %24 = tail call ptr @__cxa_begin_catch(ptr %23) #25
  %25 = icmp ult ptr %12, %16
  br i1 %25, label %26, label %31

26:                                               ; preds = %21, %26
  %27 = phi ptr [ %29, %26 ], [ %12, %21 ]
  %28 = load ptr, ptr %27, align 8, !tbaa !40
  tail call void @_ZdlPvm(ptr noundef %28, i64 noundef 512) #22
  %29 = getelementptr inbounds nuw i8, ptr %27, i64 8
  %30 = icmp ult ptr %29, %16
  br i1 %30, label %26, label %31, !llvm.loop !67

31:                                               ; preds = %26, %21
  invoke void @__cxa_rethrow() #28
          to label %37 unwind label %32

32:                                               ; preds = %31
  %33 = landingpad { ptr, i32 }
          catch ptr null
  invoke void @__cxa_end_catch()
          to label %38 unwind label %34

34:                                               ; preds = %32
  %35 = landingpad { ptr, i32 }
          catch ptr null
  %36 = extractvalue { ptr, i32 } %35, 0
  tail call void @__clang_call_terminate(ptr %36) #26
  unreachable

37:                                               ; preds = %31
  unreachable

38:                                               ; preds = %32
  %39 = extractvalue { ptr, i32 } %33, 0
  %40 = tail call ptr @__cxa_begin_catch(ptr %39) #25
  %41 = load ptr, ptr %0, align 8, !tbaa !64
  %42 = load i64, ptr %7, align 8, !tbaa !68
  %43 = shl i64 %42, 3
  tail call void @_ZdlPvm(ptr noundef %41, i64 noundef %43) #22
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %0, i8 0, i64 16, i1 false)
  invoke void @__cxa_rethrow() #28
          to label %66 unwind label %44

44:                                               ; preds = %38
  %45 = landingpad { ptr, i32 }
          cleanup
  invoke void @__cxa_end_catch()
          to label %46 unwind label %63

46:                                               ; preds = %44
  resume { ptr, i32 } %45

47:                                               ; preds = %18
  %48 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %49 = getelementptr inbounds nuw i8, ptr %0, i64 40
  store ptr %12, ptr %49, align 8, !tbaa !32
  %50 = load ptr, ptr %12, align 8, !tbaa !40
  %51 = getelementptr inbounds nuw i8, ptr %0, i64 24
  store ptr %50, ptr %51, align 8, !tbaa !30
  %52 = getelementptr inbounds nuw i8, ptr %50, i64 512
  %53 = getelementptr inbounds nuw i8, ptr %0, i64 32
  store ptr %52, ptr %53, align 8, !tbaa !31
  %54 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %55 = getelementptr inbounds i8, ptr %14, i64 -8
  %56 = getelementptr inbounds nuw i8, ptr %0, i64 72
  store ptr %55, ptr %56, align 8, !tbaa !32
  %57 = load ptr, ptr %55, align 8, !tbaa !40
  %58 = getelementptr inbounds nuw i8, ptr %0, i64 56
  store ptr %57, ptr %58, align 8, !tbaa !30
  %59 = getelementptr inbounds nuw i8, ptr %57, i64 512
  %60 = getelementptr inbounds nuw i8, ptr %0, i64 64
  store ptr %59, ptr %60, align 8, !tbaa !31
  store ptr %50, ptr %48, align 8, !tbaa !139
  %61 = and i64 %1, 63
  %62 = getelementptr inbounds nuw double, ptr %57, i64 %61
  store ptr %62, ptr %54, align 8, !tbaa !133
  ret void

63:                                               ; preds = %44
  %64 = landingpad { ptr, i32 }
          catch ptr null
  %65 = extractvalue { ptr, i32 } %64, 0
  tail call void @__clang_call_terminate(ptr %65) #26
  unreachable

66:                                               ; preds = %38
  unreachable
}

declare void @__cxa_rethrow() local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt16__introsort_loopISt15_Deque_iteratorIdRdPdElN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S7_T0_T1_(ptr dead_on_return noundef %0, ptr dead_on_return noundef %1, i64 noundef %2, i8 %3) local_unnamed_addr #2 comdat {
  %5 = alloca %"struct.std::_Deque_iterator", align 8
  %6 = alloca %"struct.std::_Deque_iterator", align 8
  %7 = alloca %"struct.std::_Deque_iterator", align 8
  %8 = alloca %"struct.std::_Deque_iterator", align 16
  %9 = alloca %"struct.std::_Deque_iterator", align 8
  %10 = alloca %"struct.std::_Deque_iterator", align 8
  %11 = alloca %"struct.std::_Deque_iterator", align 16
  %12 = alloca %"struct.std::_Deque_iterator", align 16
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %15 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %16 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %17 = getelementptr inbounds nuw i8, ptr %9, i64 8
  %18 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %19 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %20 = getelementptr inbounds nuw i8, ptr %9, i64 24
  %21 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %22 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %23 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %24 = getelementptr inbounds nuw i8, ptr %10, i64 24
  %25 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %26 = getelementptr inbounds nuw i8, ptr %8, i64 16
  %27 = getelementptr inbounds nuw i8, ptr %12, i64 16
  br label %28

28:                                               ; preds = %69, %4
  %29 = phi i64 [ %2, %4 ], [ %70, %69 ]
  %30 = load ptr, ptr %13, align 8, !tbaa !32
  %31 = load ptr, ptr %14, align 8, !tbaa !32
  %32 = ptrtoint ptr %30 to i64
  %33 = ptrtoint ptr %31 to i64
  %34 = sub i64 %32, %33
  %35 = ashr exact i64 %34, 3
  %36 = icmp ne ptr %30, null
  %37 = sext i1 %36 to i64
  %38 = add nsw i64 %35, %37
  %39 = shl nsw i64 %38, 6
  %40 = load ptr, ptr %1, align 8, !tbaa !23
  %41 = load ptr, ptr %15, align 8, !tbaa !30
  %42 = ptrtoint ptr %40 to i64
  %43 = ptrtoint ptr %41 to i64
  %44 = sub i64 %42, %43
  %45 = ashr exact i64 %44, 3
  %46 = add nsw i64 %39, %45
  %47 = load ptr, ptr %16, align 8, !tbaa !31
  %48 = load ptr, ptr %0, align 8, !tbaa !23
  %49 = ptrtoint ptr %47 to i64
  %50 = ptrtoint ptr %48 to i64
  %51 = sub i64 %49, %50
  %52 = ashr exact i64 %51, 3
  %53 = add nsw i64 %46, %52
  %54 = icmp sgt i64 %53, 16
  br i1 %54, label %55, label %77

55:                                               ; preds = %28
  %56 = icmp eq i64 %29, 0
  br i1 %56, label %57, label %69

57:                                               ; preds = %55
  store ptr %48, ptr %5, align 8, !tbaa !23
  %58 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %59 = load ptr, ptr %18, align 8, !tbaa !30
  store ptr %59, ptr %58, align 8, !tbaa !30
  %60 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store ptr %47, ptr %60, align 8, !tbaa !31
  %61 = getelementptr inbounds nuw i8, ptr %5, i64 24
  store ptr %31, ptr %61, align 8, !tbaa !32
  store ptr %40, ptr %6, align 8, !tbaa !23
  %62 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store ptr %41, ptr %62, align 8, !tbaa !30
  %63 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %64 = load ptr, ptr %23, align 8, !tbaa !31
  store ptr %64, ptr %63, align 8, !tbaa !31
  %65 = getelementptr inbounds nuw i8, ptr %6, i64 24
  store ptr %30, ptr %65, align 8, !tbaa !32
  store ptr %40, ptr %7, align 8, !tbaa !23
  %66 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store ptr %41, ptr %66, align 8, !tbaa !30
  %67 = getelementptr inbounds nuw i8, ptr %7, i64 16
  store ptr %64, ptr %67, align 8, !tbaa !31
  %68 = getelementptr inbounds nuw i8, ptr %7, i64 24
  store ptr %30, ptr %68, align 8, !tbaa !32
  call void @_ZSt14__partial_sortISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S7_S7_T0_(ptr dead_on_return noundef nonnull %5, ptr dead_on_return noundef nonnull %6, ptr dead_on_return noundef nonnull %7, i8 undef)
  br label %77

69:                                               ; preds = %55
  %70 = add nsw i64 %29, -1
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #25
  store ptr %48, ptr %9, align 8, !tbaa !23
  %71 = load ptr, ptr %18, align 8, !tbaa !30
  store ptr %71, ptr %17, align 8, !tbaa !30
  store ptr %47, ptr %19, align 8, !tbaa !31
  store ptr %31, ptr %20, align 8, !tbaa !32
  store ptr %40, ptr %10, align 8, !tbaa !23
  store ptr %41, ptr %21, align 8, !tbaa !30
  %72 = load ptr, ptr %23, align 8, !tbaa !31
  store ptr %72, ptr %22, align 8, !tbaa !31
  store ptr %30, ptr %24, align 8, !tbaa !32
  call void @_ZSt27__unguarded_partition_pivotISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops15_Iter_less_iterEET_S7_S7_T0_(ptr dead_on_unwind nonnull writable sret(%"struct.std::_Deque_iterator") align 8 %8, ptr dead_on_return noundef nonnull %9, ptr dead_on_return noundef nonnull %10, i8 undef)
  %73 = load <2 x ptr>, ptr %8, align 16, !tbaa !40
  store <2 x ptr> %73, ptr %11, align 16, !tbaa !40
  %74 = load <2 x ptr>, ptr %26, align 16, !tbaa !140
  store <2 x ptr> %74, ptr %25, align 16, !tbaa !140
  %75 = load <2 x ptr>, ptr %1, align 8, !tbaa !40
  store <2 x ptr> %75, ptr %12, align 16, !tbaa !40
  %76 = load <2 x ptr>, ptr %23, align 8, !tbaa !140
  store <2 x ptr> %76, ptr %27, align 16, !tbaa !140
  call void @_ZSt16__introsort_loopISt15_Deque_iteratorIdRdPdElN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S7_T0_T1_(ptr dead_on_return noundef nonnull %11, ptr dead_on_return noundef nonnull %12, i64 noundef %70, i8 undef)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 16 dereferenceable(32) %8, i64 32, i1 false), !tbaa.struct !141
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #25
  br label %28, !llvm.loop !143

77:                                               ; preds = %28, %57
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt22__final_insertion_sortISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S7_T0_(ptr dead_on_return noundef %0, ptr dead_on_return noundef %1, i8 %2) local_unnamed_addr #2 comdat {
  %4 = alloca %"struct.std::_Deque_iterator", align 8
  %5 = alloca %"struct.std::_Deque_iterator", align 8
  %6 = alloca %"struct.std::_Deque_iterator", align 8
  %7 = alloca %"struct.std::_Deque_iterator", align 8
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %9 = load ptr, ptr %8, align 8, !tbaa !32
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %11 = load ptr, ptr %10, align 8, !tbaa !32
  %12 = ptrtoint ptr %9 to i64
  %13 = ptrtoint ptr %11 to i64
  %14 = sub i64 %12, %13
  %15 = ashr exact i64 %14, 3
  %16 = icmp ne ptr %9, null
  %17 = sext i1 %16 to i64
  %18 = add nsw i64 %15, %17
  %19 = shl nsw i64 %18, 6
  %20 = load ptr, ptr %1, align 8, !tbaa !23
  %21 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %22 = load ptr, ptr %21, align 8, !tbaa !30
  %23 = ptrtoint ptr %20 to i64
  %24 = ptrtoint ptr %22 to i64
  %25 = sub i64 %23, %24
  %26 = ashr exact i64 %25, 3
  %27 = add nsw i64 %19, %26
  %28 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %29 = load ptr, ptr %28, align 8, !tbaa !31
  %30 = load ptr, ptr %0, align 8, !tbaa !23
  %31 = ptrtoint ptr %29 to i64
  %32 = ptrtoint ptr %30 to i64
  %33 = sub i64 %31, %32
  %34 = ashr exact i64 %33, 3
  %35 = add nsw i64 %27, %34
  %36 = icmp sgt i64 %35, 16
  %37 = getelementptr inbounds nuw i8, ptr %0, i64 8
  br i1 %36, label %38, label %152

38:                                               ; preds = %3
  store ptr %30, ptr %4, align 8, !tbaa !23
  %39 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %40 = load ptr, ptr %37, align 8, !tbaa !30
  store ptr %40, ptr %39, align 8, !tbaa !30
  %41 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store ptr %29, ptr %41, align 8, !tbaa !31
  %42 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store ptr %11, ptr %42, align 8, !tbaa !32
  tail call void @llvm.experimental.noalias.scope.decl(metadata !144)
  %43 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store ptr %40, ptr %43, align 8, !tbaa !30, !alias.scope !144
  %44 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store ptr %29, ptr %44, align 8, !tbaa !31, !alias.scope !144
  %45 = getelementptr inbounds nuw i8, ptr %5, i64 24
  store ptr %11, ptr %45, align 8, !tbaa !32, !alias.scope !144
  %46 = ptrtoint ptr %40 to i64
  %47 = sub i64 %32, %46
  %48 = ashr exact i64 %47, 3
  %49 = add nsw i64 %48, 16
  %50 = icmp sgt i64 %48, -17
  br i1 %50, label %51, label %57

51:                                               ; preds = %38
  %52 = icmp samesign ult i64 %49, 64
  br i1 %52, label %53, label %55

53:                                               ; preds = %51
  %54 = getelementptr inbounds nuw i8, ptr %30, i64 128
  br label %67

55:                                               ; preds = %51
  %56 = lshr i64 %49, 6
  br label %59

57:                                               ; preds = %38
  %58 = ashr i64 %49, 6
  br label %59

59:                                               ; preds = %57, %55
  %60 = phi i64 [ %56, %55 ], [ %58, %57 ]
  %61 = getelementptr inbounds ptr, ptr %11, i64 %60
  store ptr %61, ptr %45, align 8, !tbaa !32, !alias.scope !144
  %62 = load ptr, ptr %61, align 8, !tbaa !40, !noalias !144
  store ptr %62, ptr %43, align 8, !tbaa !30, !alias.scope !144
  %63 = getelementptr inbounds nuw i8, ptr %62, i64 512
  store ptr %63, ptr %44, align 8, !tbaa !31, !alias.scope !144
  %64 = shl nsw i64 %60, 6
  %65 = sub nsw i64 %49, %64
  %66 = getelementptr inbounds double, ptr %62, i64 %65
  br label %67

67:                                               ; preds = %53, %59
  %68 = phi ptr [ %66, %59 ], [ %54, %53 ]
  store ptr %68, ptr %5, align 8, !tbaa !23, !alias.scope !144
  call void @_ZSt16__insertion_sortISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S7_T0_(ptr dead_on_return noundef nonnull %4, ptr dead_on_return noundef nonnull %5, i8 undef)
  %69 = load ptr, ptr %0, align 8, !tbaa !23, !noalias !147
  %70 = load ptr, ptr %37, align 8, !tbaa !30, !noalias !147
  %71 = load ptr, ptr %28, align 8, !tbaa !31, !noalias !147
  %72 = load ptr, ptr %10, align 8, !tbaa !32, !noalias !147
  %73 = ptrtoint ptr %69 to i64
  %74 = ptrtoint ptr %70 to i64
  %75 = sub i64 %73, %74
  %76 = ashr exact i64 %75, 3
  %77 = add nsw i64 %76, 16
  %78 = icmp sgt i64 %76, -17
  br i1 %78, label %79, label %85

79:                                               ; preds = %67
  %80 = icmp samesign ult i64 %77, 64
  br i1 %80, label %81, label %83

81:                                               ; preds = %79
  %82 = getelementptr inbounds nuw i8, ptr %69, i64 128
  br label %95

83:                                               ; preds = %79
  %84 = lshr i64 %77, 6
  br label %87

85:                                               ; preds = %67
  %86 = ashr i64 %77, 6
  br label %87

87:                                               ; preds = %85, %83
  %88 = phi i64 [ %84, %83 ], [ %86, %85 ]
  %89 = getelementptr inbounds ptr, ptr %72, i64 %88
  %90 = load ptr, ptr %89, align 8, !tbaa !40, !noalias !147
  %91 = getelementptr inbounds nuw i8, ptr %90, i64 512
  %92 = shl nsw i64 %88, 6
  %93 = sub nsw i64 %77, %92
  %94 = getelementptr inbounds double, ptr %90, i64 %93
  br label %95

95:                                               ; preds = %81, %87
  %96 = phi ptr [ %70, %81 ], [ %90, %87 ]
  %97 = phi ptr [ %71, %81 ], [ %91, %87 ]
  %98 = phi ptr [ %72, %81 ], [ %89, %87 ]
  %99 = phi ptr [ %82, %81 ], [ %94, %87 ]
  %100 = load ptr, ptr %1, align 8, !tbaa !23
  %101 = icmp eq ptr %99, %100
  br i1 %101, label %162, label %102

102:                                              ; preds = %95, %146
  %103 = phi ptr [ %150, %146 ], [ %98, %95 ]
  %104 = phi ptr [ %149, %146 ], [ %99, %95 ]
  %105 = phi ptr [ %148, %146 ], [ %96, %95 ]
  %106 = phi ptr [ %147, %146 ], [ %97, %95 ]
  %107 = load double, ptr %104, align 8, !tbaa !16
  %108 = icmp eq ptr %104, %105
  br i1 %108, label %109, label %113

109:                                              ; preds = %102
  %110 = getelementptr inbounds i8, ptr %103, i64 -8
  %111 = load ptr, ptr %110, align 8, !tbaa !40
  %112 = getelementptr inbounds nuw i8, ptr %111, i64 512
  br label %113

113:                                              ; preds = %109, %102
  %114 = phi ptr [ %111, %109 ], [ %105, %102 ]
  %115 = phi ptr [ %110, %109 ], [ %103, %102 ]
  %116 = phi ptr [ %112, %109 ], [ %104, %102 ]
  %117 = getelementptr inbounds i8, ptr %116, i64 -8
  %118 = load double, ptr %117, align 8, !tbaa !16
  %119 = fcmp olt double %107, %118
  br i1 %119, label %120, label %138

120:                                              ; preds = %113, %131
  %121 = phi ptr [ %123, %131 ], [ %104, %113 ]
  %122 = phi double [ %136, %131 ], [ %118, %113 ]
  %123 = phi ptr [ %135, %131 ], [ %117, %113 ]
  %124 = phi ptr [ %133, %131 ], [ %115, %113 ]
  %125 = phi ptr [ %132, %131 ], [ %114, %113 ]
  store double %122, ptr %121, align 8, !tbaa !16
  %126 = icmp eq ptr %123, %125
  br i1 %126, label %127, label %131

127:                                              ; preds = %120
  %128 = getelementptr inbounds i8, ptr %124, i64 -8
  %129 = load ptr, ptr %128, align 8, !tbaa !40
  %130 = getelementptr inbounds nuw i8, ptr %129, i64 512
  br label %131

131:                                              ; preds = %127, %120
  %132 = phi ptr [ %129, %127 ], [ %125, %120 ]
  %133 = phi ptr [ %128, %127 ], [ %124, %120 ]
  %134 = phi ptr [ %130, %127 ], [ %123, %120 ]
  %135 = getelementptr inbounds i8, ptr %134, i64 -8
  %136 = load double, ptr %135, align 8, !tbaa !16
  %137 = fcmp olt double %107, %136
  br i1 %137, label %120, label %138, !llvm.loop !150

138:                                              ; preds = %131, %113
  %139 = phi ptr [ %104, %113 ], [ %123, %131 ]
  store double %107, ptr %139, align 8, !tbaa !16
  %140 = getelementptr inbounds nuw i8, ptr %104, i64 8
  %141 = icmp eq ptr %140, %106
  br i1 %141, label %142, label %146

142:                                              ; preds = %138
  %143 = getelementptr inbounds nuw i8, ptr %103, i64 8
  %144 = load ptr, ptr %143, align 8, !tbaa !40
  %145 = getelementptr inbounds nuw i8, ptr %144, i64 512
  br label %146

146:                                              ; preds = %142, %138
  %147 = phi ptr [ %145, %142 ], [ %106, %138 ]
  %148 = phi ptr [ %144, %142 ], [ %105, %138 ]
  %149 = phi ptr [ %144, %142 ], [ %140, %138 ]
  %150 = phi ptr [ %143, %142 ], [ %103, %138 ]
  %151 = icmp eq ptr %149, %100
  br i1 %151, label %162, label %102, !llvm.loop !151

152:                                              ; preds = %3
  store ptr %30, ptr %6, align 8, !tbaa !23
  %153 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %154 = load ptr, ptr %37, align 8, !tbaa !30
  store ptr %154, ptr %153, align 8, !tbaa !30
  %155 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store ptr %29, ptr %155, align 8, !tbaa !31
  %156 = getelementptr inbounds nuw i8, ptr %6, i64 24
  store ptr %11, ptr %156, align 8, !tbaa !32
  store ptr %20, ptr %7, align 8, !tbaa !23
  %157 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store ptr %22, ptr %157, align 8, !tbaa !30
  %158 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %159 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %160 = load ptr, ptr %159, align 8, !tbaa !31
  store ptr %160, ptr %158, align 8, !tbaa !31
  %161 = getelementptr inbounds nuw i8, ptr %7, i64 24
  store ptr %9, ptr %161, align 8, !tbaa !32
  call void @_ZSt16__insertion_sortISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S7_T0_(ptr dead_on_return noundef nonnull %6, ptr dead_on_return noundef nonnull %7, i8 undef)
  br label %162

162:                                              ; preds = %146, %95, %152
  ret void
}

; Function Attrs: inlinehint mustprogress uwtable
define linkonce_odr dso_local void @_ZSt14__partial_sortISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S7_S7_T0_(ptr dead_on_return noundef %0, ptr dead_on_return noundef %1, ptr dead_on_return noundef %2, i8 %3) local_unnamed_addr #15 comdat {
  %5 = alloca %"struct.std::_Deque_iterator", align 16
  %6 = alloca %"struct.std::_Deque_iterator", align 16
  %7 = alloca %"struct.std::_Deque_iterator", align 16
  %8 = alloca %"struct.std::_Deque_iterator", align 16
  %9 = load <2 x ptr>, ptr %0, align 8, !tbaa !40
  store <2 x ptr> %9, ptr %6, align 16, !tbaa !40
  %10 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %13 = load <2 x ptr>, ptr %11, align 8, !tbaa !140
  store <2 x ptr> %13, ptr %10, align 16, !tbaa !140
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %15 = load <2 x ptr>, ptr %1, align 8, !tbaa !40
  store <2 x ptr> %15, ptr %7, align 16, !tbaa !40
  %16 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %19 = load <2 x ptr>, ptr %17, align 8, !tbaa !140
  store <2 x ptr> %19, ptr %16, align 16, !tbaa !140
  %20 = load <2 x ptr>, ptr %2, align 8, !tbaa !40
  store <2 x ptr> %20, ptr %8, align 16, !tbaa !40
  %21 = getelementptr inbounds nuw i8, ptr %8, i64 16
  %22 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %23 = load <2 x ptr>, ptr %22, align 8, !tbaa !140
  store <2 x ptr> %23, ptr %21, align 16, !tbaa !140
  call void @_ZSt13__heap_selectISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S7_S7_T0_(ptr dead_on_return noundef nonnull %6, ptr dead_on_return noundef nonnull %7, ptr dead_on_return noundef nonnull %8, i8 undef)
  %24 = load <2 x ptr>, ptr %0, align 8, !tbaa !40
  %25 = load ptr, ptr %11, align 8, !tbaa !31
  %26 = load ptr, ptr %12, align 8, !tbaa !32
  %27 = load ptr, ptr %1, align 8, !tbaa !23
  %28 = load ptr, ptr %14, align 8, !tbaa !30
  %29 = load ptr, ptr %18, align 8, !tbaa !32
  %30 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %31 = getelementptr inbounds nuw i8, ptr %5, i64 24
  %32 = ptrtoint ptr %26 to i64
  %33 = ptrtoint ptr %25 to i64
  %34 = extractelement <2 x ptr> %24, i64 0
  %35 = ptrtoint ptr %34 to i64
  %36 = sub i64 %33, %35
  %37 = ashr exact i64 %36, 3
  %38 = ptrtoint ptr %29 to i64
  %39 = sub i64 %38, %32
  %40 = ashr exact i64 %39, 3
  %41 = icmp ne ptr %29, null
  %42 = sext i1 %41 to i64
  %43 = add nsw i64 %40, %42
  %44 = shl nsw i64 %43, 6
  %45 = ptrtoint ptr %27 to i64
  %46 = ptrtoint ptr %28 to i64
  %47 = sub i64 %45, %46
  %48 = ashr exact i64 %47, 3
  %49 = add nsw i64 %48, %37
  %50 = add i64 %49, %44
  %51 = icmp sgt i64 %50, 1
  br i1 %51, label %52, label %99

52:                                               ; preds = %4, %72
  %53 = phi i64 [ %73, %72 ], [ %46, %4 ]
  %54 = phi i64 [ %74, %72 ], [ %40, %4 ]
  %55 = phi ptr [ %80, %72 ], [ %27, %4 ]
  %56 = phi ptr [ %76, %72 ], [ %28, %4 ]
  %57 = phi ptr [ %75, %72 ], [ %29, %4 ]
  %58 = icmp eq ptr %55, %56
  br i1 %58, label %64, label %59

59:                                               ; preds = %52
  %60 = ptrtoint ptr %57 to i64
  %61 = sub i64 %60, %32
  %62 = ashr exact i64 %61, 3
  %63 = ptrtoint ptr %56 to i64
  br label %72

64:                                               ; preds = %52
  %65 = getelementptr inbounds i8, ptr %57, i64 -8
  %66 = load ptr, ptr %65, align 8, !tbaa !40
  %67 = getelementptr inbounds nuw i8, ptr %66, i64 512
  %68 = ptrtoint ptr %65 to i64
  %69 = sub i64 %68, %32
  %70 = ashr exact i64 %69, 3
  %71 = ptrtoint ptr %66 to i64
  br label %72

72:                                               ; preds = %59, %64
  %73 = phi i64 [ %63, %59 ], [ %71, %64 ]
  %74 = phi i64 [ %62, %59 ], [ %70, %64 ]
  %75 = phi ptr [ %57, %59 ], [ %65, %64 ]
  %76 = phi ptr [ %56, %59 ], [ %66, %64 ]
  %77 = phi i64 [ %53, %59 ], [ %71, %64 ]
  %78 = phi i64 [ %54, %59 ], [ %70, %64 ]
  %79 = phi ptr [ %55, %59 ], [ %67, %64 ]
  %80 = getelementptr inbounds i8, ptr %79, i64 -8
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  %81 = load double, ptr %80, align 8, !tbaa !16
  %82 = load double, ptr %34, align 8, !tbaa !16
  store double %82, ptr %80, align 8, !tbaa !16
  store <2 x ptr> %24, ptr %5, align 16, !tbaa !40
  store ptr %25, ptr %30, align 16, !tbaa !31
  store ptr %26, ptr %31, align 8, !tbaa !32
  %83 = icmp ne ptr %75, null
  %84 = sext i1 %83 to i64
  %85 = add nsw i64 %78, %84
  %86 = shl nsw i64 %85, 6
  %87 = ptrtoint ptr %80 to i64
  %88 = sub i64 %87, %77
  %89 = ashr exact i64 %88, 3
  %90 = add i64 %86, %37
  %91 = add i64 %90, %89
  call void @_ZSt13__adjust_heapISt15_Deque_iteratorIdRdPdEldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S8_T1_T2_(ptr dead_on_return noundef nonnull %5, i64 noundef 0, i64 noundef %91, double noundef %81, i8 undef)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  %92 = add nsw i64 %74, %84
  %93 = shl nsw i64 %92, 6
  %94 = sub i64 %87, %73
  %95 = ashr exact i64 %94, 3
  %96 = add nsw i64 %95, %37
  %97 = add i64 %96, %93
  %98 = icmp sgt i64 %97, 1
  br i1 %98, label %52, label %99, !llvm.loop !152

99:                                               ; preds = %72, %4
  ret void
}

; Function Attrs: inlinehint mustprogress uwtable
define linkonce_odr dso_local void @_ZSt27__unguarded_partition_pivotISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops15_Iter_less_iterEET_S7_S7_T0_(ptr dead_on_unwind noalias writable sret(%"struct.std::_Deque_iterator") align 8 %0, ptr dead_on_return noundef %1, ptr dead_on_return noundef %2, i8 %3) local_unnamed_addr #15 comdat {
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %6 = load ptr, ptr %5, align 8, !tbaa !32
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %8 = load ptr, ptr %7, align 8, !tbaa !32
  %9 = ptrtoint ptr %6 to i64
  %10 = ptrtoint ptr %8 to i64
  %11 = sub i64 %9, %10
  %12 = ashr exact i64 %11, 3
  %13 = icmp ne ptr %6, null
  %14 = sext i1 %13 to i64
  %15 = add nsw i64 %12, %14
  %16 = shl nsw i64 %15, 6
  %17 = load ptr, ptr %2, align 8, !tbaa !23
  %18 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %19 = load ptr, ptr %18, align 8, !tbaa !30
  %20 = ptrtoint ptr %17 to i64
  %21 = ptrtoint ptr %19 to i64
  %22 = sub i64 %20, %21
  %23 = ashr exact i64 %22, 3
  %24 = add nsw i64 %16, %23
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %26 = load ptr, ptr %25, align 8, !tbaa !31
  %27 = load ptr, ptr %1, align 8, !tbaa !23
  %28 = ptrtoint ptr %26 to i64
  %29 = ptrtoint ptr %27 to i64
  %30 = sub i64 %28, %29
  %31 = ashr exact i64 %30, 3
  %32 = add nsw i64 %24, %31
  %33 = sdiv i64 %32, 2
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %35 = load ptr, ptr %34, align 8, !tbaa !30, !noalias !153
  %36 = ptrtoint ptr %35 to i64
  %37 = sub i64 %29, %36
  %38 = ashr exact i64 %37, 3
  %39 = add nsw i64 %33, %38
  %40 = icmp sgt i64 %39, -1
  br i1 %40, label %41, label %47

41:                                               ; preds = %4
  %42 = icmp samesign ult i64 %39, 64
  br i1 %42, label %43, label %45

43:                                               ; preds = %41
  %44 = getelementptr inbounds double, ptr %27, i64 %33
  br label %56

45:                                               ; preds = %41
  %46 = lshr i64 %39, 6
  br label %49

47:                                               ; preds = %4
  %48 = ashr i64 %39, 6
  br label %49

49:                                               ; preds = %47, %45
  %50 = phi i64 [ %46, %45 ], [ %48, %47 ]
  %51 = getelementptr inbounds ptr, ptr %8, i64 %50
  %52 = load ptr, ptr %51, align 8, !tbaa !40, !noalias !153
  %53 = shl nsw i64 %50, 6
  %54 = sub nsw i64 %39, %53
  %55 = getelementptr inbounds double, ptr %52, i64 %54
  br label %56

56:                                               ; preds = %43, %49
  %57 = phi ptr [ %55, %49 ], [ %44, %43 ]
  %58 = add nsw i64 %38, 1
  %59 = icmp sgt i64 %38, -2
  br i1 %59, label %60, label %66

60:                                               ; preds = %56
  %61 = icmp samesign ult i64 %58, 64
  br i1 %61, label %62, label %64

62:                                               ; preds = %60
  %63 = getelementptr inbounds nuw i8, ptr %27, i64 8
  br label %75

64:                                               ; preds = %60
  %65 = lshr i64 %58, 6
  br label %68

66:                                               ; preds = %56
  %67 = ashr i64 %58, 6
  br label %68

68:                                               ; preds = %66, %64
  %69 = phi i64 [ %65, %64 ], [ %67, %66 ]
  %70 = getelementptr inbounds ptr, ptr %8, i64 %69
  %71 = load ptr, ptr %70, align 8, !tbaa !40, !noalias !156
  %72 = shl nsw i64 %69, 6
  %73 = sub nsw i64 %58, %72
  %74 = getelementptr inbounds double, ptr %71, i64 %73
  br label %75

75:                                               ; preds = %62, %68
  %76 = phi ptr [ %74, %68 ], [ %63, %62 ]
  %77 = add nsw i64 %23, -1
  %78 = icmp sgt i64 %23, 0
  br i1 %78, label %79, label %85

79:                                               ; preds = %75
  %80 = icmp samesign ult i64 %23, 65
  br i1 %80, label %81, label %83

81:                                               ; preds = %79
  %82 = getelementptr inbounds i8, ptr %17, i64 -8
  br label %94

83:                                               ; preds = %79
  %84 = lshr i64 %77, 6
  br label %87

85:                                               ; preds = %75
  %86 = ashr i64 %77, 6
  br label %87

87:                                               ; preds = %85, %83
  %88 = phi i64 [ %84, %83 ], [ %86, %85 ]
  %89 = getelementptr inbounds ptr, ptr %6, i64 %88
  %90 = load ptr, ptr %89, align 8, !tbaa !40, !noalias !159
  %91 = shl nsw i64 %88, 6
  %92 = sub nsw i64 %77, %91
  %93 = getelementptr inbounds double, ptr %90, i64 %92
  br label %94

94:                                               ; preds = %81, %87
  %95 = phi ptr [ %93, %87 ], [ %82, %81 ]
  %96 = load double, ptr %76, align 8, !tbaa !16
  %97 = load double, ptr %57, align 8, !tbaa !16
  %98 = fcmp olt double %96, %97
  %99 = load double, ptr %95, align 8, !tbaa !16
  br i1 %98, label %100, label %109

100:                                              ; preds = %94
  %101 = fcmp olt double %97, %99
  br i1 %101, label %102, label %104

102:                                              ; preds = %100
  %103 = load double, ptr %27, align 8, !tbaa !16
  store double %97, ptr %27, align 8, !tbaa !16
  store double %103, ptr %57, align 8, !tbaa !16
  br label %118

104:                                              ; preds = %100
  %105 = fcmp olt double %96, %99
  %106 = load double, ptr %27, align 8, !tbaa !16
  br i1 %105, label %107, label %108

107:                                              ; preds = %104
  store double %99, ptr %27, align 8, !tbaa !16
  store double %106, ptr %95, align 8, !tbaa !16
  br label %118

108:                                              ; preds = %104
  store double %96, ptr %27, align 8, !tbaa !16
  store double %106, ptr %76, align 8, !tbaa !16
  br label %118

109:                                              ; preds = %94
  %110 = fcmp olt double %96, %99
  br i1 %110, label %111, label %113

111:                                              ; preds = %109
  %112 = load double, ptr %27, align 8, !tbaa !16
  store double %96, ptr %27, align 8, !tbaa !16
  store double %112, ptr %76, align 8, !tbaa !16
  br label %118

113:                                              ; preds = %109
  %114 = fcmp olt double %97, %99
  %115 = load double, ptr %27, align 8, !tbaa !16
  br i1 %114, label %116, label %117

116:                                              ; preds = %113
  store double %99, ptr %27, align 8, !tbaa !16
  store double %115, ptr %95, align 8, !tbaa !16
  br label %118

117:                                              ; preds = %113
  store double %97, ptr %27, align 8, !tbaa !16
  store double %115, ptr %57, align 8, !tbaa !16
  br label %118

118:                                              ; preds = %102, %107, %108, %111, %116, %117
  br i1 %59, label %119, label %125

119:                                              ; preds = %118
  %120 = icmp samesign ult i64 %58, 64
  br i1 %120, label %121, label %123

121:                                              ; preds = %119
  %122 = getelementptr inbounds nuw i8, ptr %27, i64 8
  br label %135

123:                                              ; preds = %119
  %124 = lshr i64 %58, 6
  br label %127

125:                                              ; preds = %118
  %126 = ashr i64 %58, 6
  br label %127

127:                                              ; preds = %125, %123
  %128 = phi i64 [ %124, %123 ], [ %126, %125 ]
  %129 = getelementptr inbounds ptr, ptr %8, i64 %128
  %130 = load ptr, ptr %129, align 8, !tbaa !40, !noalias !162
  %131 = getelementptr inbounds nuw i8, ptr %130, i64 512
  %132 = shl nsw i64 %128, 6
  %133 = sub nsw i64 %58, %132
  %134 = getelementptr inbounds double, ptr %130, i64 %133
  br label %135

135:                                              ; preds = %121, %127
  %136 = phi ptr [ %8, %121 ], [ %129, %127 ]
  %137 = phi ptr [ %26, %121 ], [ %131, %127 ]
  %138 = phi ptr [ %35, %121 ], [ %130, %127 ]
  %139 = phi ptr [ %122, %121 ], [ %134, %127 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !165)
  br label %140

140:                                              ; preds = %223, %135
  %141 = phi ptr [ %136, %135 ], [ %224, %223 ]
  %142 = phi ptr [ %137, %135 ], [ %225, %223 ]
  %143 = phi ptr [ %138, %135 ], [ %226, %223 ]
  %144 = phi ptr [ %139, %135 ], [ %227, %223 ]
  %145 = phi ptr [ %6, %135 ], [ %208, %223 ]
  %146 = phi ptr [ %19, %135 ], [ %209, %223 ]
  %147 = phi ptr [ %17, %135 ], [ %210, %223 ]
  %148 = load double, ptr %144, align 8, !tbaa !16, !noalias !165
  %149 = load double, ptr %27, align 8, !tbaa !16, !noalias !165
  %150 = fcmp olt double %148, %149
  br i1 %150, label %151, label %171

151:                                              ; preds = %140, %163
  %152 = phi ptr [ %164, %163 ], [ %141, %140 ]
  %153 = phi ptr [ %165, %163 ], [ %142, %140 ]
  %154 = phi ptr [ %166, %163 ], [ %143, %140 ]
  %155 = phi ptr [ %168, %163 ], [ %142, %140 ]
  %156 = phi ptr [ %167, %163 ], [ %144, %140 ]
  %157 = getelementptr inbounds nuw i8, ptr %156, i64 8
  %158 = icmp eq ptr %157, %155
  br i1 %158, label %159, label %163

159:                                              ; preds = %151
  %160 = getelementptr inbounds nuw i8, ptr %152, i64 8
  %161 = load ptr, ptr %160, align 8, !tbaa !40, !noalias !165
  %162 = getelementptr inbounds nuw i8, ptr %161, i64 512
  br label %163

163:                                              ; preds = %159, %151
  %164 = phi ptr [ %160, %159 ], [ %152, %151 ]
  %165 = phi ptr [ %162, %159 ], [ %153, %151 ]
  %166 = phi ptr [ %161, %159 ], [ %154, %151 ]
  %167 = phi ptr [ %161, %159 ], [ %157, %151 ]
  %168 = phi ptr [ %162, %159 ], [ %155, %151 ]
  %169 = load double, ptr %167, align 8, !tbaa !16, !noalias !165
  %170 = fcmp olt double %169, %149
  br i1 %170, label %151, label %171, !llvm.loop !168

171:                                              ; preds = %163, %140
  %172 = phi double [ %148, %140 ], [ %169, %163 ]
  %173 = phi ptr [ %141, %140 ], [ %164, %163 ]
  %174 = phi ptr [ %142, %140 ], [ %165, %163 ]
  %175 = phi ptr [ %143, %140 ], [ %166, %163 ]
  %176 = phi ptr [ %144, %140 ], [ %167, %163 ]
  %177 = icmp eq ptr %147, %146
  br i1 %177, label %178, label %182

178:                                              ; preds = %171
  %179 = getelementptr inbounds i8, ptr %145, i64 -8
  %180 = load ptr, ptr %179, align 8, !tbaa !40, !noalias !165
  %181 = getelementptr inbounds nuw i8, ptr %180, i64 512
  br label %182

182:                                              ; preds = %178, %171
  %183 = phi ptr [ %179, %178 ], [ %145, %171 ]
  %184 = phi ptr [ %180, %178 ], [ %146, %171 ]
  %185 = phi ptr [ %181, %178 ], [ %147, %171 ]
  %186 = getelementptr inbounds i8, ptr %185, i64 -8
  %187 = load double, ptr %186, align 8, !tbaa !16, !noalias !165
  %188 = fcmp olt double %149, %187
  br i1 %188, label %189, label %207

189:                                              ; preds = %182, %199
  %190 = phi ptr [ %200, %199 ], [ %183, %182 ]
  %191 = phi ptr [ %201, %199 ], [ %184, %182 ]
  %192 = phi ptr [ %202, %199 ], [ %184, %182 ]
  %193 = phi ptr [ %204, %199 ], [ %186, %182 ]
  %194 = icmp eq ptr %193, %192
  br i1 %194, label %195, label %199

195:                                              ; preds = %189
  %196 = getelementptr inbounds i8, ptr %190, i64 -8
  %197 = load ptr, ptr %196, align 8, !tbaa !40, !noalias !165
  %198 = getelementptr inbounds nuw i8, ptr %197, i64 512
  br label %199

199:                                              ; preds = %195, %189
  %200 = phi ptr [ %196, %195 ], [ %190, %189 ]
  %201 = phi ptr [ %197, %195 ], [ %191, %189 ]
  %202 = phi ptr [ %197, %195 ], [ %192, %189 ]
  %203 = phi ptr [ %198, %195 ], [ %193, %189 ]
  %204 = getelementptr inbounds i8, ptr %203, i64 -8
  %205 = load double, ptr %204, align 8, !tbaa !16, !noalias !165
  %206 = fcmp olt double %149, %205
  br i1 %206, label %189, label %207, !llvm.loop !169

207:                                              ; preds = %199, %182
  %208 = phi ptr [ %183, %182 ], [ %200, %199 ]
  %209 = phi ptr [ %184, %182 ], [ %201, %199 ]
  %210 = phi ptr [ %186, %182 ], [ %204, %199 ]
  %211 = phi double [ %187, %182 ], [ %205, %199 ]
  %212 = icmp eq ptr %173, %208
  %213 = icmp ult ptr %176, %210
  %214 = icmp ult ptr %173, %208
  %215 = select i1 %212, i1 %213, i1 %214
  br i1 %215, label %216, label %228

216:                                              ; preds = %207
  store double %211, ptr %176, align 8, !tbaa !16, !noalias !165
  store double %172, ptr %210, align 8, !tbaa !16, !noalias !165
  %217 = getelementptr inbounds nuw i8, ptr %176, i64 8
  %218 = icmp eq ptr %217, %174
  br i1 %218, label %219, label %223

219:                                              ; preds = %216
  %220 = getelementptr inbounds nuw i8, ptr %173, i64 8
  %221 = load ptr, ptr %220, align 8, !tbaa !40, !noalias !165
  %222 = getelementptr inbounds nuw i8, ptr %221, i64 512
  br label %223

223:                                              ; preds = %219, %216
  %224 = phi ptr [ %220, %219 ], [ %173, %216 ]
  %225 = phi ptr [ %222, %219 ], [ %174, %216 ]
  %226 = phi ptr [ %221, %219 ], [ %175, %216 ]
  %227 = phi ptr [ %221, %219 ], [ %217, %216 ]
  br label %140, !llvm.loop !170

228:                                              ; preds = %207
  store ptr %176, ptr %0, align 8, !tbaa !23, !alias.scope !165
  %229 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %175, ptr %229, align 8, !tbaa !30, !alias.scope !165
  %230 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %174, ptr %230, align 8, !tbaa !31, !alias.scope !165
  %231 = getelementptr inbounds nuw i8, ptr %0, i64 24
  store ptr %173, ptr %231, align 8, !tbaa !32, !alias.scope !165
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt13__heap_selectISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S7_S7_T0_(ptr dead_on_return noundef %0, ptr dead_on_return noundef %1, ptr dead_on_return noundef %2, i8 %3) local_unnamed_addr #2 comdat {
  %5 = alloca %"struct.std::_Deque_iterator", align 8
  %6 = alloca %"struct.std::_Deque_iterator", align 8
  %7 = load ptr, ptr %0, align 8, !tbaa !23
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %9 = load ptr, ptr %8, align 8, !tbaa !30
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %11 = load ptr, ptr %10, align 8, !tbaa !31
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %13 = load ptr, ptr %12, align 8, !tbaa !32
  %14 = load ptr, ptr %1, align 8, !tbaa !23
  %15 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %16 = load ptr, ptr %15, align 8, !tbaa !30
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %19 = load ptr, ptr %18, align 8, !tbaa !32
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  %20 = ptrtoint ptr %19 to i64
  %21 = ptrtoint ptr %13 to i64
  %22 = sub i64 %20, %21
  %23 = ashr exact i64 %22, 3
  %24 = icmp ne ptr %19, null
  %25 = sext i1 %24 to i64
  %26 = add nsw i64 %23, %25
  %27 = shl nsw i64 %26, 6
  %28 = ptrtoint ptr %14 to i64
  %29 = ptrtoint ptr %16 to i64
  %30 = sub i64 %28, %29
  %31 = ashr exact i64 %30, 3
  %32 = ptrtoint ptr %11 to i64
  %33 = ptrtoint ptr %7 to i64
  %34 = sub i64 %32, %33
  %35 = ashr exact i64 %34, 3
  %36 = add nsw i64 %31, %35
  %37 = add i64 %36, %27
  %38 = icmp slt i64 %37, 2
  br i1 %38, label %75, label %39

39:                                               ; preds = %4
  %40 = add nsw i64 %37, -2
  %41 = lshr i64 %40, 1
  %42 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %43 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %44 = getelementptr inbounds nuw i8, ptr %6, i64 24
  %45 = ptrtoint ptr %9 to i64
  %46 = sub i64 %33, %45
  %47 = ashr exact i64 %46, 3
  br label %48

48:                                               ; preds = %67, %39
  %49 = phi i64 [ %41, %39 ], [ %71, %67 ]
  %50 = add nsw i64 %49, %47
  %51 = icmp sgt i64 %50, -1
  br i1 %51, label %52, label %58

52:                                               ; preds = %48
  %53 = icmp samesign ult i64 %50, 64
  br i1 %53, label %54, label %56

54:                                               ; preds = %52
  %55 = getelementptr inbounds double, ptr %7, i64 %49
  br label %67

56:                                               ; preds = %52
  %57 = lshr i64 %50, 6
  br label %60

58:                                               ; preds = %48
  %59 = ashr i64 %50, 6
  br label %60

60:                                               ; preds = %58, %56
  %61 = phi i64 [ %57, %56 ], [ %59, %58 ]
  %62 = getelementptr inbounds ptr, ptr %13, i64 %61
  %63 = load ptr, ptr %62, align 8, !tbaa !40, !noalias !171
  %64 = shl nsw i64 %61, 6
  %65 = sub nsw i64 %50, %64
  %66 = getelementptr inbounds double, ptr %63, i64 %65
  br label %67

67:                                               ; preds = %60, %54
  %68 = phi ptr [ %66, %60 ], [ %55, %54 ]
  %69 = load double, ptr %68, align 8, !tbaa !16
  store ptr %7, ptr %6, align 8, !tbaa !23
  store ptr %9, ptr %42, align 8, !tbaa !30
  store ptr %11, ptr %43, align 8, !tbaa !31
  store ptr %13, ptr %44, align 8, !tbaa !32
  call void @_ZSt13__adjust_heapISt15_Deque_iteratorIdRdPdEldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S8_T1_T2_(ptr dead_on_return noundef nonnull %6, i64 noundef %49, i64 noundef %37, double noundef %69, i8 undef)
  %70 = icmp eq i64 %49, 0
  %71 = add nsw i64 %49, -1
  br i1 %70, label %72, label %48, !llvm.loop !174

72:                                               ; preds = %67
  %73 = load ptr, ptr %1, align 8, !tbaa !23
  %74 = load ptr, ptr %18, align 8, !tbaa !32
  br label %75

75:                                               ; preds = %72, %4
  %76 = phi ptr [ %74, %72 ], [ %19, %4 ]
  %77 = phi ptr [ %73, %72 ], [ %14, %4 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  %78 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %79 = load ptr, ptr %78, align 8, !tbaa !32
  %80 = icmp eq ptr %76, %79
  %81 = load ptr, ptr %2, align 8
  %82 = icmp ult ptr %77, %81
  %83 = icmp ult ptr %76, %79
  %84 = select i1 %80, i1 %82, i1 %83
  br i1 %84, label %85, label %89

85:                                               ; preds = %75
  %86 = load ptr, ptr %17, align 8, !tbaa !31
  %87 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %88 = getelementptr inbounds nuw i8, ptr %5, i64 24
  br label %90

89:                                               ; preds = %130, %75
  ret void

90:                                               ; preds = %85, %130
  %91 = phi ptr [ %76, %85 ], [ %133, %130 ]
  %92 = phi ptr [ %86, %85 ], [ %132, %130 ]
  %93 = phi ptr [ %77, %85 ], [ %131, %130 ]
  %94 = load ptr, ptr %0, align 8, !tbaa !23
  %95 = load double, ptr %93, align 8, !tbaa !16
  %96 = load double, ptr %94, align 8, !tbaa !16
  %97 = fcmp olt double %95, %96
  br i1 %97, label %98, label %123

98:                                               ; preds = %90
  %99 = load ptr, ptr %12, align 8, !tbaa !32
  %100 = load ptr, ptr %1, align 8, !tbaa !23
  %101 = load ptr, ptr %15, align 8, !tbaa !30
  %102 = load ptr, ptr %18, align 8, !tbaa !32
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  store double %96, ptr %93, align 8, !tbaa !16
  store ptr %94, ptr %5, align 8, !tbaa !23
  %103 = load ptr, ptr %10, align 8, !tbaa !31
  %104 = load <2 x ptr>, ptr %8, align 8, !tbaa !40
  store <2 x ptr> %104, ptr %87, align 8, !tbaa !40
  store ptr %99, ptr %88, align 8, !tbaa !32
  %105 = ptrtoint ptr %102 to i64
  %106 = ptrtoint ptr %99 to i64
  %107 = sub i64 %105, %106
  %108 = ashr exact i64 %107, 3
  %109 = icmp ne ptr %102, null
  %110 = sext i1 %109 to i64
  %111 = add nsw i64 %108, %110
  %112 = shl nsw i64 %111, 6
  %113 = ptrtoint ptr %100 to i64
  %114 = ptrtoint ptr %101 to i64
  %115 = sub i64 %113, %114
  %116 = ashr exact i64 %115, 3
  %117 = ptrtoint ptr %103 to i64
  %118 = ptrtoint ptr %94 to i64
  %119 = sub i64 %117, %118
  %120 = ashr exact i64 %119, 3
  %121 = add nsw i64 %116, %120
  %122 = add i64 %121, %112
  call void @_ZSt13__adjust_heapISt15_Deque_iteratorIdRdPdEldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S8_T1_T2_(ptr dead_on_return noundef nonnull %5, i64 noundef 0, i64 noundef %122, double noundef %95, i8 undef)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  br label %123

123:                                              ; preds = %90, %98
  %124 = getelementptr inbounds nuw i8, ptr %93, i64 8
  %125 = icmp eq ptr %124, %92
  br i1 %125, label %126, label %130

126:                                              ; preds = %123
  %127 = getelementptr inbounds nuw i8, ptr %91, i64 8
  %128 = load ptr, ptr %127, align 8, !tbaa !40
  %129 = getelementptr inbounds nuw i8, ptr %128, i64 512
  br label %130

130:                                              ; preds = %123, %126
  %131 = phi ptr [ %128, %126 ], [ %124, %123 ]
  %132 = phi ptr [ %129, %126 ], [ %92, %123 ]
  %133 = phi ptr [ %127, %126 ], [ %91, %123 ]
  %134 = load ptr, ptr %78, align 8, !tbaa !32
  %135 = icmp eq ptr %133, %134
  %136 = load ptr, ptr %2, align 8
  %137 = icmp ult ptr %131, %136
  %138 = icmp ult ptr %133, %134
  %139 = select i1 %135, i1 %137, i1 %138
  br i1 %139, label %90, label %89, !llvm.loop !175
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt13__adjust_heapISt15_Deque_iteratorIdRdPdEldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S8_T1_T2_(ptr dead_on_return noundef %0, i64 noundef %1, i64 noundef %2, double noundef %3, i8 %4) local_unnamed_addr #2 comdat {
  %6 = add nsw i64 %2, -1
  %7 = sdiv i64 %6, 2
  %8 = icmp slt i64 %1, %7
  br i1 %8, label %9, label %106

9:                                                ; preds = %5
  %10 = load ptr, ptr %0, align 8, !tbaa !23, !noalias !176
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %12 = load ptr, ptr %11, align 8, !tbaa !30, !noalias !176
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %14 = load ptr, ptr %13, align 8, !tbaa !32, !noalias !176
  %15 = ptrtoint ptr %10 to i64
  %16 = ptrtoint ptr %12 to i64
  %17 = sub i64 %15, %16
  %18 = ashr exact i64 %17, 3
  br label %19

19:                                               ; preds = %9, %103
  %20 = phi i64 [ %1, %9 ], [ %65, %103 ]
  %21 = shl i64 %20, 1
  %22 = add i64 %21, 2
  %23 = add nsw i64 %18, %22
  %24 = icmp sgt i64 %23, -1
  br i1 %24, label %25, label %31

25:                                               ; preds = %19
  %26 = icmp samesign ult i64 %23, 64
  br i1 %26, label %27, label %29

27:                                               ; preds = %25
  %28 = getelementptr inbounds double, ptr %10, i64 %22
  br label %40

29:                                               ; preds = %25
  %30 = lshr i64 %23, 6
  br label %33

31:                                               ; preds = %19
  %32 = ashr i64 %23, 6
  br label %33

33:                                               ; preds = %31, %29
  %34 = phi i64 [ %30, %29 ], [ %32, %31 ]
  %35 = getelementptr inbounds ptr, ptr %14, i64 %34
  %36 = load ptr, ptr %35, align 8, !tbaa !40, !noalias !176
  %37 = shl nsw i64 %34, 6
  %38 = sub nsw i64 %23, %37
  %39 = getelementptr inbounds double, ptr %36, i64 %38
  br label %40

40:                                               ; preds = %27, %33
  %41 = phi ptr [ %39, %33 ], [ %28, %27 ]
  %42 = or disjoint i64 %21, 1
  %43 = add nsw i64 %18, %42
  %44 = icmp sgt i64 %43, -1
  br i1 %44, label %45, label %51

45:                                               ; preds = %40
  %46 = icmp samesign ult i64 %43, 64
  br i1 %46, label %47, label %49

47:                                               ; preds = %45
  %48 = getelementptr inbounds double, ptr %10, i64 %42
  br label %60

49:                                               ; preds = %45
  %50 = lshr i64 %43, 6
  br label %53

51:                                               ; preds = %40
  %52 = ashr i64 %43, 6
  br label %53

53:                                               ; preds = %51, %49
  %54 = phi i64 [ %50, %49 ], [ %52, %51 ]
  %55 = getelementptr inbounds ptr, ptr %14, i64 %54
  %56 = load ptr, ptr %55, align 8, !tbaa !40, !noalias !179
  %57 = shl nsw i64 %54, 6
  %58 = sub nsw i64 %43, %57
  %59 = getelementptr inbounds double, ptr %56, i64 %58
  br label %60

60:                                               ; preds = %47, %53
  %61 = phi ptr [ %59, %53 ], [ %48, %47 ]
  %62 = load double, ptr %41, align 8, !tbaa !16
  %63 = load double, ptr %61, align 8, !tbaa !16
  %64 = fcmp olt double %62, %63
  %65 = select i1 %64, i64 %42, i64 %22
  %66 = add nsw i64 %65, %18
  %67 = icmp sgt i64 %66, -1
  br i1 %67, label %68, label %74

68:                                               ; preds = %60
  %69 = icmp samesign ult i64 %66, 64
  br i1 %69, label %70, label %72

70:                                               ; preds = %68
  %71 = getelementptr inbounds double, ptr %10, i64 %65
  br label %83

72:                                               ; preds = %68
  %73 = lshr i64 %66, 6
  br label %76

74:                                               ; preds = %60
  %75 = ashr i64 %66, 6
  br label %76

76:                                               ; preds = %74, %72
  %77 = phi i64 [ %73, %72 ], [ %75, %74 ]
  %78 = getelementptr inbounds ptr, ptr %14, i64 %77
  %79 = load ptr, ptr %78, align 8, !tbaa !40, !noalias !182
  %80 = shl nsw i64 %77, 6
  %81 = sub nsw i64 %66, %80
  %82 = getelementptr inbounds double, ptr %79, i64 %81
  br label %83

83:                                               ; preds = %70, %76
  %84 = phi ptr [ %82, %76 ], [ %71, %70 ]
  %85 = load double, ptr %84, align 8, !tbaa !16
  %86 = add nsw i64 %18, %20
  %87 = icmp sgt i64 %86, -1
  br i1 %87, label %88, label %94

88:                                               ; preds = %83
  %89 = icmp samesign ult i64 %86, 64
  br i1 %89, label %90, label %92

90:                                               ; preds = %88
  %91 = getelementptr inbounds double, ptr %10, i64 %20
  br label %103

92:                                               ; preds = %88
  %93 = lshr i64 %86, 6
  br label %96

94:                                               ; preds = %83
  %95 = ashr i64 %86, 6
  br label %96

96:                                               ; preds = %94, %92
  %97 = phi i64 [ %93, %92 ], [ %95, %94 ]
  %98 = getelementptr inbounds ptr, ptr %14, i64 %97
  %99 = load ptr, ptr %98, align 8, !tbaa !40, !noalias !185
  %100 = shl nsw i64 %97, 6
  %101 = sub nsw i64 %86, %100
  %102 = getelementptr inbounds double, ptr %99, i64 %101
  br label %103

103:                                              ; preds = %90, %96
  %104 = phi ptr [ %102, %96 ], [ %91, %90 ]
  store double %85, ptr %104, align 8, !tbaa !16
  %105 = icmp slt i64 %65, %7
  br i1 %105, label %19, label %106, !llvm.loop !188

106:                                              ; preds = %103, %5
  %107 = phi i64 [ %1, %5 ], [ %65, %103 ]
  %108 = and i64 %2, 1
  %109 = icmp eq i64 %108, 0
  br i1 %109, label %110, label %165

110:                                              ; preds = %106
  %111 = add nsw i64 %2, -2
  %112 = ashr exact i64 %111, 1
  %113 = icmp eq i64 %107, %112
  br i1 %113, label %114, label %165

114:                                              ; preds = %110
  %115 = shl nsw i64 %107, 1
  %116 = or disjoint i64 %115, 1
  %117 = load ptr, ptr %0, align 8, !tbaa !23, !noalias !189
  %118 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %119 = load ptr, ptr %118, align 8, !tbaa !30, !noalias !189
  %120 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %121 = load ptr, ptr %120, align 8, !tbaa !32, !noalias !189
  %122 = ptrtoint ptr %117 to i64
  %123 = ptrtoint ptr %119 to i64
  %124 = sub i64 %122, %123
  %125 = ashr exact i64 %124, 3
  %126 = add nsw i64 %125, %116
  %127 = icmp sgt i64 %126, -1
  br i1 %127, label %128, label %134

128:                                              ; preds = %114
  %129 = icmp samesign ult i64 %126, 64
  br i1 %129, label %130, label %132

130:                                              ; preds = %128
  %131 = getelementptr inbounds double, ptr %117, i64 %116
  br label %143

132:                                              ; preds = %128
  %133 = lshr i64 %126, 6
  br label %136

134:                                              ; preds = %114
  %135 = ashr i64 %126, 6
  br label %136

136:                                              ; preds = %134, %132
  %137 = phi i64 [ %133, %132 ], [ %135, %134 ]
  %138 = getelementptr inbounds ptr, ptr %121, i64 %137
  %139 = load ptr, ptr %138, align 8, !tbaa !40, !noalias !189
  %140 = shl nsw i64 %137, 6
  %141 = sub nsw i64 %126, %140
  %142 = getelementptr inbounds double, ptr %139, i64 %141
  br label %143

143:                                              ; preds = %130, %136
  %144 = phi ptr [ %142, %136 ], [ %131, %130 ]
  %145 = load double, ptr %144, align 8, !tbaa !16
  %146 = add nsw i64 %125, %107
  %147 = icmp sgt i64 %146, -1
  br i1 %147, label %148, label %154

148:                                              ; preds = %143
  %149 = icmp samesign ult i64 %146, 64
  br i1 %149, label %150, label %152

150:                                              ; preds = %148
  %151 = getelementptr inbounds double, ptr %117, i64 %107
  br label %163

152:                                              ; preds = %148
  %153 = lshr i64 %146, 6
  br label %156

154:                                              ; preds = %143
  %155 = ashr i64 %146, 6
  br label %156

156:                                              ; preds = %154, %152
  %157 = phi i64 [ %153, %152 ], [ %155, %154 ]
  %158 = getelementptr inbounds ptr, ptr %121, i64 %157
  %159 = load ptr, ptr %158, align 8, !tbaa !40, !noalias !192
  %160 = shl nsw i64 %157, 6
  %161 = sub nsw i64 %146, %160
  %162 = getelementptr inbounds double, ptr %159, i64 %161
  br label %163

163:                                              ; preds = %150, %156
  %164 = phi ptr [ %162, %156 ], [ %151, %150 ]
  store double %145, ptr %164, align 8, !tbaa !16
  br label %165

165:                                              ; preds = %163, %110, %106
  %166 = phi i64 [ %116, %163 ], [ %107, %110 ], [ %107, %106 ]
  %167 = load ptr, ptr %0, align 8, !tbaa !23
  %168 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %169 = load ptr, ptr %168, align 8, !tbaa !30
  %170 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %171 = load ptr, ptr %170, align 8, !tbaa !32
  %172 = icmp sgt i64 %166, %1
  %173 = ptrtoint ptr %167 to i64
  %174 = ptrtoint ptr %169 to i64
  %175 = sub i64 %173, %174
  %176 = ashr exact i64 %175, 3
  br i1 %172, label %177, label %234

177:                                              ; preds = %165, %231
  %178 = phi i64 [ %180, %231 ], [ %166, %165 ]
  %179 = add nsw i64 %178, -1
  %180 = sdiv i64 %179, 2
  %181 = add nsw i64 %180, %176
  %182 = icmp sgt i64 %181, -1
  br i1 %182, label %183, label %193

183:                                              ; preds = %177
  %184 = icmp samesign ult i64 %181, 64
  br i1 %184, label %201, label %185

185:                                              ; preds = %183
  %186 = lshr i64 %181, 6
  %187 = getelementptr inbounds nuw ptr, ptr %171, i64 %186
  %188 = load ptr, ptr %187, align 8, !tbaa !40, !noalias !195
  %189 = and i64 %181, 63
  %190 = getelementptr inbounds nuw double, ptr %188, i64 %189
  %191 = load double, ptr %190, align 8, !tbaa !16
  %192 = fcmp olt double %191, %3
  br i1 %192, label %205, label %234

193:                                              ; preds = %177
  %194 = ashr i64 %181, 6
  %195 = getelementptr inbounds ptr, ptr %171, i64 %194
  %196 = load ptr, ptr %195, align 8, !tbaa !40, !noalias !195
  %197 = and i64 %181, 63
  %198 = getelementptr inbounds nuw double, ptr %196, i64 %197
  %199 = load double, ptr %198, align 8, !tbaa !16
  %200 = fcmp olt double %199, %3
  br i1 %200, label %205, label %234

201:                                              ; preds = %183
  %202 = getelementptr inbounds double, ptr %167, i64 %180
  %203 = load double, ptr %202, align 8, !tbaa !16
  %204 = fcmp olt double %203, %3
  br i1 %204, label %212, label %234

205:                                              ; preds = %193, %185
  %206 = phi ptr [ %188, %185 ], [ %196, %193 ]
  %207 = phi i64 [ %186, %185 ], [ %194, %193 ]
  %208 = shl nsw i64 %207, 6
  %209 = sub nsw i64 %181, %208
  %210 = getelementptr inbounds double, ptr %206, i64 %209
  %211 = load double, ptr %210, align 8, !tbaa !16
  br label %212

212:                                              ; preds = %205, %201
  %213 = phi double [ %211, %205 ], [ %203, %201 ]
  %214 = add nsw i64 %178, %176
  %215 = icmp sgt i64 %214, -1
  br i1 %215, label %216, label %222

216:                                              ; preds = %212
  %217 = icmp samesign ult i64 %214, 64
  br i1 %217, label %218, label %220

218:                                              ; preds = %216
  %219 = getelementptr inbounds double, ptr %167, i64 %178
  br label %231

220:                                              ; preds = %216
  %221 = lshr i64 %214, 6
  br label %224

222:                                              ; preds = %212
  %223 = ashr i64 %214, 6
  br label %224

224:                                              ; preds = %222, %220
  %225 = phi i64 [ %221, %220 ], [ %223, %222 ]
  %226 = getelementptr inbounds ptr, ptr %171, i64 %225
  %227 = load ptr, ptr %226, align 8, !tbaa !40, !noalias !198
  %228 = shl nsw i64 %225, 6
  %229 = sub nsw i64 %214, %228
  %230 = getelementptr inbounds double, ptr %227, i64 %229
  br label %231

231:                                              ; preds = %224, %218
  %232 = phi ptr [ %230, %224 ], [ %219, %218 ]
  store double %213, ptr %232, align 8, !tbaa !16
  %233 = icmp sgt i64 %180, %1
  br i1 %233, label %177, label %234, !llvm.loop !201

234:                                              ; preds = %231, %201, %193, %185, %165
  %235 = phi i64 [ %166, %165 ], [ %178, %185 ], [ %180, %231 ], [ %178, %201 ], [ %178, %193 ]
  %236 = add nsw i64 %235, %176
  %237 = icmp sgt i64 %236, -1
  br i1 %237, label %238, label %244

238:                                              ; preds = %234
  %239 = icmp samesign ult i64 %236, 64
  br i1 %239, label %240, label %242

240:                                              ; preds = %238
  %241 = getelementptr inbounds double, ptr %167, i64 %235
  br label %253

242:                                              ; preds = %238
  %243 = lshr i64 %236, 6
  br label %246

244:                                              ; preds = %234
  %245 = ashr i64 %236, 6
  br label %246

246:                                              ; preds = %244, %242
  %247 = phi i64 [ %243, %242 ], [ %245, %244 ]
  %248 = getelementptr inbounds ptr, ptr %171, i64 %247
  %249 = load ptr, ptr %248, align 8, !tbaa !40, !noalias !202
  %250 = shl nsw i64 %247, 6
  %251 = sub nsw i64 %236, %250
  %252 = getelementptr inbounds double, ptr %249, i64 %251
  br label %253

253:                                              ; preds = %240, %246
  %254 = phi ptr [ %252, %246 ], [ %241, %240 ]
  store double %3, ptr %254, align 8, !tbaa !16
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt16__insertion_sortISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S7_T0_(ptr dead_on_return noundef %0, ptr dead_on_return noundef %1, i8 %2) local_unnamed_addr #2 comdat {
  %4 = alloca %"struct.std::_Deque_iterator", align 8
  %5 = alloca %"struct.std::_Deque_iterator", align 8
  %6 = alloca %"struct.std::_Deque_iterator", align 8
  %7 = alloca %"struct.std::_Deque_iterator", align 8
  %8 = load ptr, ptr %0, align 8, !tbaa !23
  %9 = load ptr, ptr %1, align 8, !tbaa !23
  %10 = icmp eq ptr %8, %9
  br i1 %10, label %142, label %11

11:                                               ; preds = %3
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %13 = load ptr, ptr %12, align 8, !tbaa !30, !noalias !205
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %15 = load ptr, ptr %14, align 8, !tbaa !31, !noalias !205
  %16 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %17 = load ptr, ptr %16, align 8, !tbaa !32, !noalias !205
  %18 = ptrtoint ptr %8 to i64
  %19 = ptrtoint ptr %13 to i64
  %20 = sub i64 %18, %19
  %21 = ashr exact i64 %20, 3
  %22 = add nsw i64 %21, 1
  %23 = icmp sgt i64 %21, -2
  br i1 %23, label %24, label %30

24:                                               ; preds = %11
  %25 = icmp samesign ult i64 %22, 64
  br i1 %25, label %26, label %28

26:                                               ; preds = %24
  %27 = getelementptr inbounds nuw i8, ptr %8, i64 8
  br label %40

28:                                               ; preds = %24
  %29 = lshr i64 %22, 6
  br label %32

30:                                               ; preds = %11
  %31 = ashr i64 %22, 6
  br label %32

32:                                               ; preds = %30, %28
  %33 = phi i64 [ %29, %28 ], [ %31, %30 ]
  %34 = getelementptr inbounds ptr, ptr %17, i64 %33
  %35 = load ptr, ptr %34, align 8, !tbaa !40, !noalias !205
  %36 = getelementptr inbounds nuw i8, ptr %35, i64 512
  %37 = shl nsw i64 %33, 6
  %38 = sub nsw i64 %22, %37
  %39 = getelementptr inbounds double, ptr %35, i64 %38
  br label %40

40:                                               ; preds = %26, %32
  %41 = phi ptr [ %13, %26 ], [ %35, %32 ]
  %42 = phi ptr [ %15, %26 ], [ %36, %32 ]
  %43 = phi ptr [ %17, %26 ], [ %34, %32 ]
  %44 = phi ptr [ %27, %26 ], [ %39, %32 ]
  %45 = icmp eq ptr %44, %9
  br i1 %45, label %142, label %46

46:                                               ; preds = %40
  %47 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %48 = getelementptr inbounds nuw i8, ptr %4, i64 24
  %49 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %50 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %51 = getelementptr inbounds nuw i8, ptr %5, i64 24
  %52 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %53 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %54 = getelementptr inbounds nuw i8, ptr %6, i64 24
  br label %55

55:                                               ; preds = %46, %135
  %56 = phi ptr [ %8, %46 ], [ %128, %135 ]
  %57 = phi ptr [ %43, %46 ], [ %139, %135 ]
  %58 = phi ptr [ %42, %46 ], [ %138, %135 ]
  %59 = phi ptr [ %41, %46 ], [ %137, %135 ]
  %60 = phi ptr [ %44, %46 ], [ %136, %135 ]
  %61 = load double, ptr %60, align 8, !tbaa !16
  %62 = load double, ptr %56, align 8, !tbaa !16
  %63 = fcmp olt double %61, %62
  br i1 %63, label %64, label %95

64:                                               ; preds = %55
  %65 = load <2 x ptr>, ptr %12, align 8, !tbaa !40
  %66 = load ptr, ptr %16, align 8, !tbaa !32
  %67 = ptrtoint ptr %60 to i64
  %68 = ptrtoint ptr %59 to i64
  %69 = sub i64 %67, %68
  %70 = ashr exact i64 %69, 3
  %71 = add nsw i64 %70, 1
  %72 = icmp sgt i64 %70, -2
  br i1 %72, label %73, label %79

73:                                               ; preds = %64
  %74 = icmp samesign ult i64 %71, 64
  br i1 %74, label %75, label %77

75:                                               ; preds = %73
  %76 = getelementptr inbounds nuw i8, ptr %60, i64 8
  br label %89

77:                                               ; preds = %73
  %78 = lshr i64 %71, 6
  br label %81

79:                                               ; preds = %64
  %80 = ashr i64 %71, 6
  br label %81

81:                                               ; preds = %79, %77
  %82 = phi i64 [ %78, %77 ], [ %80, %79 ]
  %83 = getelementptr inbounds ptr, ptr %57, i64 %82
  %84 = load ptr, ptr %83, align 8, !tbaa !40, !noalias !208
  %85 = getelementptr inbounds nuw i8, ptr %84, i64 512
  %86 = shl nsw i64 %82, 6
  %87 = sub nsw i64 %71, %86
  %88 = getelementptr inbounds double, ptr %84, i64 %87
  br label %89

89:                                               ; preds = %75, %81
  %90 = phi ptr [ %59, %75 ], [ %84, %81 ]
  %91 = phi ptr [ %58, %75 ], [ %85, %81 ]
  %92 = phi ptr [ %57, %75 ], [ %83, %81 ]
  %93 = phi ptr [ %76, %75 ], [ %88, %81 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %7), !noalias !211
  call void @llvm.lifetime.start.p0(ptr nonnull %4), !noalias !214
  call void @llvm.lifetime.start.p0(ptr nonnull %5), !noalias !214
  call void @llvm.lifetime.start.p0(ptr nonnull %6), !noalias !214
  store ptr %56, ptr %4, align 8, !tbaa !23, !noalias !217
  store <2 x ptr> %65, ptr %47, align 8, !tbaa !40, !noalias !217
  store ptr %66, ptr %48, align 8, !tbaa !32, !noalias !217
  store ptr %60, ptr %5, align 8, !tbaa !23, !noalias !217
  store ptr %59, ptr %49, align 8, !tbaa !30, !noalias !217
  store ptr %58, ptr %50, align 8, !tbaa !31, !noalias !217
  store ptr %57, ptr %51, align 8, !tbaa !32, !noalias !217
  store ptr %93, ptr %6, align 8, !tbaa !23, !noalias !217
  store ptr %90, ptr %52, align 8, !tbaa !30, !noalias !217
  store ptr %91, ptr %53, align 8, !tbaa !31, !noalias !217
  store ptr %92, ptr %54, align 8, !tbaa !32, !noalias !217
  call void @_ZSt24__copy_move_backward_ditILb1EdRdPdSt15_Deque_iteratorIdS0_S1_EET3_S2_IT0_T1_T2_ES8_S4_(ptr dead_on_unwind nonnull writable sret(%"struct.std::_Deque_iterator") align 8 %7, ptr dead_on_return noundef nonnull %4, ptr dead_on_return noundef nonnull %5, ptr dead_on_return noundef nonnull %6), !noalias !214
  call void @llvm.lifetime.end.p0(ptr nonnull %4), !noalias !214
  call void @llvm.lifetime.end.p0(ptr nonnull %5), !noalias !214
  call void @llvm.lifetime.end.p0(ptr nonnull %6), !noalias !214
  call void @llvm.lifetime.end.p0(ptr nonnull %7), !noalias !211
  %94 = load ptr, ptr %0, align 8, !tbaa !23
  br label %126

95:                                               ; preds = %55
  %96 = icmp eq ptr %60, %59
  br i1 %96, label %97, label %101

97:                                               ; preds = %95
  %98 = getelementptr inbounds i8, ptr %57, i64 -8
  %99 = load ptr, ptr %98, align 8, !tbaa !40
  %100 = getelementptr inbounds nuw i8, ptr %99, i64 512
  br label %101

101:                                              ; preds = %97, %95
  %102 = phi ptr [ %99, %97 ], [ %59, %95 ]
  %103 = phi ptr [ %98, %97 ], [ %57, %95 ]
  %104 = phi ptr [ %100, %97 ], [ %60, %95 ]
  %105 = getelementptr inbounds i8, ptr %104, i64 -8
  %106 = load double, ptr %105, align 8, !tbaa !16
  %107 = fcmp olt double %61, %106
  br i1 %107, label %108, label %126

108:                                              ; preds = %101, %119
  %109 = phi ptr [ %111, %119 ], [ %60, %101 ]
  %110 = phi double [ %124, %119 ], [ %106, %101 ]
  %111 = phi ptr [ %123, %119 ], [ %105, %101 ]
  %112 = phi ptr [ %121, %119 ], [ %103, %101 ]
  %113 = phi ptr [ %120, %119 ], [ %102, %101 ]
  store double %110, ptr %109, align 8, !tbaa !16
  %114 = icmp eq ptr %111, %113
  br i1 %114, label %115, label %119

115:                                              ; preds = %108
  %116 = getelementptr inbounds i8, ptr %112, i64 -8
  %117 = load ptr, ptr %116, align 8, !tbaa !40
  %118 = getelementptr inbounds nuw i8, ptr %117, i64 512
  br label %119

119:                                              ; preds = %115, %108
  %120 = phi ptr [ %117, %115 ], [ %113, %108 ]
  %121 = phi ptr [ %116, %115 ], [ %112, %108 ]
  %122 = phi ptr [ %118, %115 ], [ %111, %108 ]
  %123 = getelementptr inbounds i8, ptr %122, i64 -8
  %124 = load double, ptr %123, align 8, !tbaa !16
  %125 = fcmp olt double %61, %124
  br i1 %125, label %108, label %126, !llvm.loop !150

126:                                              ; preds = %119, %101, %89
  %127 = phi ptr [ %94, %89 ], [ %60, %101 ], [ %111, %119 ]
  %128 = phi ptr [ %94, %89 ], [ %56, %101 ], [ %56, %119 ]
  store double %61, ptr %127, align 8, !tbaa !16
  %129 = getelementptr inbounds nuw i8, ptr %60, i64 8
  %130 = icmp eq ptr %129, %58
  br i1 %130, label %131, label %135

131:                                              ; preds = %126
  %132 = getelementptr inbounds nuw i8, ptr %57, i64 8
  %133 = load ptr, ptr %132, align 8, !tbaa !40
  %134 = getelementptr inbounds nuw i8, ptr %133, i64 512
  br label %135

135:                                              ; preds = %126, %131
  %136 = phi ptr [ %133, %131 ], [ %129, %126 ]
  %137 = phi ptr [ %133, %131 ], [ %59, %126 ]
  %138 = phi ptr [ %134, %131 ], [ %58, %126 ]
  %139 = phi ptr [ %132, %131 ], [ %57, %126 ]
  %140 = load ptr, ptr %1, align 8, !tbaa !23
  %141 = icmp eq ptr %136, %140
  br i1 %141, label %142, label %55, !llvm.loop !220

142:                                              ; preds = %135, %40, %3
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt24__copy_move_backward_ditILb1EdRdPdSt15_Deque_iteratorIdS0_S1_EET3_S2_IT0_T1_T2_ES8_S4_(ptr dead_on_unwind noalias writable sret(%"struct.std::_Deque_iterator") align 8 %0, ptr dead_on_return noundef %1, ptr dead_on_return noundef %2, ptr dead_on_return noundef %3) local_unnamed_addr #2 comdat {
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %6 = load ptr, ptr %5, align 8, !tbaa !32
  %7 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %8 = load ptr, ptr %7, align 8, !tbaa !32
  %9 = icmp eq ptr %6, %8
  br i1 %9, label %252, label %10

10:                                               ; preds = %4
  %11 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %12 = load ptr, ptr %11, align 8, !tbaa !30
  %13 = load ptr, ptr %2, align 8, !tbaa !23
  %14 = load ptr, ptr %3, align 8, !tbaa !23
  %15 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %16 = load ptr, ptr %15, align 8, !tbaa !30
  %17 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %18 = load ptr, ptr %17, align 8, !tbaa !31
  %19 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %20 = load ptr, ptr %19, align 8, !tbaa !32
  %21 = ptrtoint ptr %13 to i64
  %22 = ptrtoint ptr %12 to i64
  %23 = sub i64 %21, %22
  %24 = ashr exact i64 %23, 3
  %25 = icmp sgt i64 %24, 0
  br i1 %25, label %26, label %90

26:                                               ; preds = %10, %82
  %27 = phi ptr [ %87, %82 ], [ %14, %10 ]
  %28 = phi ptr [ %83, %82 ], [ %16, %10 ]
  %29 = phi ptr [ %84, %82 ], [ %18, %10 ]
  %30 = phi ptr [ %85, %82 ], [ %20, %10 ]
  %31 = phi ptr [ %86, %82 ], [ %16, %10 ]
  %32 = phi ptr [ %48, %82 ], [ %13, %10 ]
  %33 = phi i64 [ %88, %82 ], [ %24, %10 ]
  %34 = ptrtoint ptr %27 to i64
  %35 = ptrtoint ptr %31 to i64
  %36 = sub i64 %34, %35
  %37 = ashr exact i64 %36, 3
  %38 = icmp eq ptr %27, %31
  br i1 %38, label %39, label %43

39:                                               ; preds = %26
  %40 = getelementptr inbounds i8, ptr %30, i64 -8
  %41 = load ptr, ptr %40, align 8, !tbaa !40, !noalias !221
  %42 = getelementptr inbounds nuw i8, ptr %41, i64 512
  br label %43

43:                                               ; preds = %39, %26
  %44 = phi i64 [ 64, %39 ], [ %37, %26 ]
  %45 = phi ptr [ %42, %39 ], [ %27, %26 ]
  %46 = tail call i64 @llvm.smin.i64(i64 %44, i64 %33)
  %47 = sub nsw i64 0, %46
  %48 = getelementptr inbounds double, ptr %32, i64 %47
  %49 = icmp sgt i64 %46, 1
  br i1 %49, label %50, label %56, !prof !15

50:                                               ; preds = %43
  %51 = shl nuw nsw i64 %46, 3
  %52 = getelementptr inbounds double, ptr %45, i64 %47
  tail call void @llvm.memmove.p0.p0.i64(ptr nonnull align 8 %52, ptr nonnull align 8 %48, i64 %51, i1 false), !noalias !221
  %53 = ptrtoint ptr %28 to i64
  %54 = sub i64 %34, %53
  %55 = ashr exact i64 %54, 3
  br label %61

56:                                               ; preds = %43
  %57 = icmp eq i64 %46, 1
  br i1 %57, label %58, label %61

58:                                               ; preds = %56
  %59 = getelementptr inbounds i8, ptr %45, i64 -8
  %60 = load double, ptr %48, align 8, !tbaa !16, !noalias !221
  store double %60, ptr %59, align 8, !tbaa !16, !noalias !221
  br label %61

61:                                               ; preds = %58, %56, %50
  %62 = phi i64 [ %55, %50 ], [ %37, %56 ], [ %37, %58 ]
  %63 = phi ptr [ %28, %50 ], [ %31, %56 ], [ %31, %58 ]
  %64 = sub nsw i64 %62, %46
  %65 = icmp sgt i64 %64, -1
  br i1 %65, label %66, label %72

66:                                               ; preds = %61
  %67 = icmp samesign ult i64 %64, 64
  br i1 %67, label %68, label %70

68:                                               ; preds = %66
  %69 = getelementptr inbounds double, ptr %27, i64 %47
  br label %82

70:                                               ; preds = %66
  %71 = lshr i64 %64, 6
  br label %74

72:                                               ; preds = %61
  %73 = ashr i64 %64, 6
  br label %74

74:                                               ; preds = %72, %70
  %75 = phi i64 [ %71, %70 ], [ %73, %72 ]
  %76 = getelementptr inbounds ptr, ptr %30, i64 %75
  %77 = load ptr, ptr %76, align 8, !tbaa !40, !noalias !221
  %78 = getelementptr inbounds nuw i8, ptr %77, i64 512
  %79 = shl nsw i64 %75, 6
  %80 = sub nsw i64 %64, %79
  %81 = getelementptr inbounds double, ptr %77, i64 %80
  br label %82

82:                                               ; preds = %74, %68
  %83 = phi ptr [ %28, %68 ], [ %77, %74 ]
  %84 = phi ptr [ %29, %68 ], [ %78, %74 ]
  %85 = phi ptr [ %30, %68 ], [ %76, %74 ]
  %86 = phi ptr [ %63, %68 ], [ %77, %74 ]
  %87 = phi ptr [ %69, %68 ], [ %81, %74 ]
  %88 = sub nsw i64 %33, %46
  %89 = icmp sgt i64 %88, 0
  br i1 %89, label %26, label %90, !llvm.loop !224

90:                                               ; preds = %82, %10
  %91 = phi ptr [ %18, %10 ], [ %84, %82 ]
  %92 = phi ptr [ %20, %10 ], [ %85, %82 ]
  %93 = phi ptr [ %16, %10 ], [ %86, %82 ]
  %94 = phi ptr [ %14, %10 ], [ %87, %82 ]
  store ptr %94, ptr %3, align 8, !tbaa !40
  store ptr %93, ptr %15, align 8, !tbaa !40
  store ptr %91, ptr %17, align 8, !tbaa !40
  store ptr %92, ptr %19, align 8, !tbaa !142
  %95 = load ptr, ptr %7, align 8, !tbaa !32
  %96 = getelementptr inbounds i8, ptr %95, i64 -8
  %97 = load ptr, ptr %5, align 8, !tbaa !32
  %98 = icmp eq ptr %96, %97
  br i1 %98, label %99, label %176

99:                                               ; preds = %248, %90
  %100 = phi ptr [ %92, %90 ], [ %241, %248 ]
  %101 = phi ptr [ %91, %90 ], [ %242, %248 ]
  %102 = phi ptr [ %93, %90 ], [ %244, %248 ]
  %103 = phi ptr [ %94, %90 ], [ %245, %248 ]
  %104 = load ptr, ptr %1, align 8, !tbaa !23
  %105 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %106 = load ptr, ptr %105, align 8, !tbaa !31
  %107 = ptrtoint ptr %106 to i64
  %108 = ptrtoint ptr %104 to i64
  %109 = sub i64 %107, %108
  %110 = ashr exact i64 %109, 3
  %111 = icmp sgt i64 %110, 0
  br i1 %111, label %112, label %331

112:                                              ; preds = %99, %168
  %113 = phi ptr [ %169, %168 ], [ %100, %99 ]
  %114 = phi ptr [ %170, %168 ], [ %101, %99 ]
  %115 = phi ptr [ %171, %168 ], [ %102, %99 ]
  %116 = phi ptr [ %173, %168 ], [ %103, %99 ]
  %117 = phi ptr [ %172, %168 ], [ %102, %99 ]
  %118 = phi ptr [ %134, %168 ], [ %106, %99 ]
  %119 = phi i64 [ %174, %168 ], [ %110, %99 ]
  %120 = ptrtoint ptr %116 to i64
  %121 = ptrtoint ptr %117 to i64
  %122 = sub i64 %120, %121
  %123 = ashr exact i64 %122, 3
  %124 = icmp eq ptr %116, %117
  br i1 %124, label %125, label %129

125:                                              ; preds = %112
  %126 = getelementptr inbounds i8, ptr %113, i64 -8
  %127 = load ptr, ptr %126, align 8, !tbaa !40, !noalias !225
  %128 = getelementptr inbounds nuw i8, ptr %127, i64 512
  br label %129

129:                                              ; preds = %125, %112
  %130 = phi i64 [ 64, %125 ], [ %123, %112 ]
  %131 = phi ptr [ %128, %125 ], [ %116, %112 ]
  %132 = tail call i64 @llvm.smin.i64(i64 %130, i64 %119)
  %133 = sub nsw i64 0, %132
  %134 = getelementptr inbounds double, ptr %118, i64 %133
  %135 = icmp sgt i64 %132, 1
  br i1 %135, label %136, label %142, !prof !15

136:                                              ; preds = %129
  %137 = shl nuw nsw i64 %132, 3
  %138 = getelementptr inbounds double, ptr %131, i64 %133
  tail call void @llvm.memmove.p0.p0.i64(ptr nonnull align 8 %138, ptr nonnull align 8 %134, i64 %137, i1 false), !noalias !225
  %139 = ptrtoint ptr %115 to i64
  %140 = sub i64 %120, %139
  %141 = ashr exact i64 %140, 3
  br label %147

142:                                              ; preds = %129
  %143 = icmp eq i64 %132, 1
  br i1 %143, label %144, label %147

144:                                              ; preds = %142
  %145 = getelementptr inbounds i8, ptr %131, i64 -8
  %146 = load double, ptr %134, align 8, !tbaa !16, !noalias !225
  store double %146, ptr %145, align 8, !tbaa !16, !noalias !225
  br label %147

147:                                              ; preds = %144, %142, %136
  %148 = phi i64 [ %141, %136 ], [ %123, %142 ], [ %123, %144 ]
  %149 = phi ptr [ %115, %136 ], [ %117, %142 ], [ %117, %144 ]
  %150 = sub nsw i64 %148, %132
  %151 = icmp sgt i64 %150, -1
  br i1 %151, label %152, label %158

152:                                              ; preds = %147
  %153 = icmp samesign ult i64 %150, 64
  br i1 %153, label %154, label %156

154:                                              ; preds = %152
  %155 = getelementptr inbounds double, ptr %116, i64 %133
  br label %168

156:                                              ; preds = %152
  %157 = lshr i64 %150, 6
  br label %160

158:                                              ; preds = %147
  %159 = ashr i64 %150, 6
  br label %160

160:                                              ; preds = %158, %156
  %161 = phi i64 [ %157, %156 ], [ %159, %158 ]
  %162 = getelementptr inbounds ptr, ptr %113, i64 %161
  %163 = load ptr, ptr %162, align 8, !tbaa !40, !noalias !225
  %164 = getelementptr inbounds nuw i8, ptr %163, i64 512
  %165 = shl nsw i64 %161, 6
  %166 = sub nsw i64 %150, %165
  %167 = getelementptr inbounds double, ptr %163, i64 %166
  br label %168

168:                                              ; preds = %160, %154
  %169 = phi ptr [ %113, %154 ], [ %162, %160 ]
  %170 = phi ptr [ %114, %154 ], [ %164, %160 ]
  %171 = phi ptr [ %115, %154 ], [ %163, %160 ]
  %172 = phi ptr [ %149, %154 ], [ %163, %160 ]
  %173 = phi ptr [ %155, %154 ], [ %167, %160 ]
  %174 = sub nsw i64 %119, %132
  %175 = icmp sgt i64 %174, 0
  br i1 %175, label %112, label %331, !llvm.loop !224

176:                                              ; preds = %90, %248
  %177 = phi ptr [ %241, %248 ], [ %92, %90 ]
  %178 = phi ptr [ %242, %248 ], [ %91, %90 ]
  %179 = phi ptr [ %244, %248 ], [ %93, %90 ]
  %180 = phi ptr [ %245, %248 ], [ %94, %90 ]
  %181 = phi ptr [ %249, %248 ], [ %96, %90 ]
  %182 = load ptr, ptr %181, align 8, !tbaa !40
  %183 = getelementptr inbounds nuw i8, ptr %182, i64 512
  br label %184

184:                                              ; preds = %240, %176
  %185 = phi ptr [ %177, %176 ], [ %241, %240 ]
  %186 = phi ptr [ %178, %176 ], [ %242, %240 ]
  %187 = phi ptr [ %179, %176 ], [ %243, %240 ]
  %188 = phi ptr [ %180, %176 ], [ %245, %240 ]
  %189 = phi ptr [ %179, %176 ], [ %244, %240 ]
  %190 = phi ptr [ %183, %176 ], [ %206, %240 ]
  %191 = phi i64 [ 64, %176 ], [ %246, %240 ]
  %192 = ptrtoint ptr %188 to i64
  %193 = ptrtoint ptr %189 to i64
  %194 = sub i64 %192, %193
  %195 = ashr exact i64 %194, 3
  %196 = icmp eq ptr %188, %189
  br i1 %196, label %197, label %201

197:                                              ; preds = %184
  %198 = getelementptr inbounds i8, ptr %185, i64 -8
  %199 = load ptr, ptr %198, align 8, !tbaa !40, !noalias !228
  %200 = getelementptr inbounds nuw i8, ptr %199, i64 512
  br label %201

201:                                              ; preds = %197, %184
  %202 = phi i64 [ 64, %197 ], [ %195, %184 ]
  %203 = phi ptr [ %200, %197 ], [ %188, %184 ]
  %204 = tail call i64 @llvm.smin.i64(i64 %202, i64 %191)
  %205 = sub nsw i64 0, %204
  %206 = getelementptr inbounds double, ptr %190, i64 %205
  %207 = icmp sgt i64 %204, 1
  br i1 %207, label %208, label %214, !prof !15

208:                                              ; preds = %201
  %209 = shl nuw nsw i64 %204, 3
  %210 = getelementptr inbounds double, ptr %203, i64 %205
  tail call void @llvm.memmove.p0.p0.i64(ptr nonnull align 8 %210, ptr nonnull align 8 %206, i64 %209, i1 false), !noalias !228
  %211 = ptrtoint ptr %187 to i64
  %212 = sub i64 %192, %211
  %213 = ashr exact i64 %212, 3
  br label %219

214:                                              ; preds = %201
  %215 = icmp eq i64 %204, 1
  br i1 %215, label %216, label %219

216:                                              ; preds = %214
  %217 = getelementptr inbounds i8, ptr %203, i64 -8
  %218 = load double, ptr %206, align 8, !tbaa !16, !noalias !228
  store double %218, ptr %217, align 8, !tbaa !16, !noalias !228
  br label %219

219:                                              ; preds = %216, %214, %208
  %220 = phi i64 [ %213, %208 ], [ %195, %214 ], [ %195, %216 ]
  %221 = phi ptr [ %187, %208 ], [ %189, %214 ], [ %189, %216 ]
  %222 = sub nsw i64 %220, %204
  %223 = icmp sgt i64 %222, -1
  br i1 %223, label %224, label %230

224:                                              ; preds = %219
  %225 = icmp samesign ult i64 %222, 64
  br i1 %225, label %226, label %228

226:                                              ; preds = %224
  %227 = getelementptr inbounds double, ptr %188, i64 %205
  br label %240

228:                                              ; preds = %224
  %229 = lshr i64 %222, 6
  br label %232

230:                                              ; preds = %219
  %231 = ashr i64 %222, 6
  br label %232

232:                                              ; preds = %230, %228
  %233 = phi i64 [ %229, %228 ], [ %231, %230 ]
  %234 = getelementptr inbounds ptr, ptr %185, i64 %233
  %235 = load ptr, ptr %234, align 8, !tbaa !40, !noalias !228
  %236 = getelementptr inbounds nuw i8, ptr %235, i64 512
  %237 = shl nsw i64 %233, 6
  %238 = sub nsw i64 %222, %237
  %239 = getelementptr inbounds double, ptr %235, i64 %238
  br label %240

240:                                              ; preds = %232, %226
  %241 = phi ptr [ %185, %226 ], [ %234, %232 ]
  %242 = phi ptr [ %186, %226 ], [ %236, %232 ]
  %243 = phi ptr [ %187, %226 ], [ %235, %232 ]
  %244 = phi ptr [ %221, %226 ], [ %235, %232 ]
  %245 = phi ptr [ %227, %226 ], [ %239, %232 ]
  %246 = sub nsw i64 %191, %204
  %247 = icmp sgt i64 %246, 0
  br i1 %247, label %184, label %248, !llvm.loop !224

248:                                              ; preds = %240
  store ptr %245, ptr %3, align 8, !tbaa !40
  store ptr %244, ptr %15, align 8, !tbaa !40
  store ptr %242, ptr %17, align 8, !tbaa !40
  store ptr %241, ptr %19, align 8, !tbaa !142
  %249 = getelementptr inbounds i8, ptr %181, i64 -8
  %250 = load ptr, ptr %5, align 8, !tbaa !32
  %251 = icmp eq ptr %249, %250
  br i1 %251, label %99, label %176, !llvm.loop !231

252:                                              ; preds = %4
  %253 = load ptr, ptr %1, align 8, !tbaa !23
  %254 = load ptr, ptr %2, align 8, !tbaa !23
  %255 = load ptr, ptr %3, align 8, !tbaa !23
  %256 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %257 = load ptr, ptr %256, align 8, !tbaa !30
  %258 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %259 = load ptr, ptr %258, align 8, !tbaa !31
  %260 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %261 = load ptr, ptr %260, align 8, !tbaa !32
  %262 = ptrtoint ptr %254 to i64
  %263 = ptrtoint ptr %253 to i64
  %264 = sub i64 %262, %263
  %265 = ashr exact i64 %264, 3
  %266 = icmp sgt i64 %265, 0
  br i1 %266, label %267, label %331

267:                                              ; preds = %252, %323
  %268 = phi ptr [ %324, %323 ], [ %261, %252 ]
  %269 = phi ptr [ %325, %323 ], [ %259, %252 ]
  %270 = phi ptr [ %326, %323 ], [ %257, %252 ]
  %271 = phi ptr [ %328, %323 ], [ %255, %252 ]
  %272 = phi ptr [ %327, %323 ], [ %257, %252 ]
  %273 = phi ptr [ %289, %323 ], [ %254, %252 ]
  %274 = phi i64 [ %329, %323 ], [ %265, %252 ]
  %275 = ptrtoint ptr %271 to i64
  %276 = ptrtoint ptr %272 to i64
  %277 = sub i64 %275, %276
  %278 = ashr exact i64 %277, 3
  %279 = icmp eq ptr %271, %272
  br i1 %279, label %280, label %284

280:                                              ; preds = %267
  %281 = getelementptr inbounds i8, ptr %268, i64 -8
  %282 = load ptr, ptr %281, align 8, !tbaa !40, !noalias !232
  %283 = getelementptr inbounds nuw i8, ptr %282, i64 512
  br label %284

284:                                              ; preds = %280, %267
  %285 = phi i64 [ 64, %280 ], [ %278, %267 ]
  %286 = phi ptr [ %283, %280 ], [ %271, %267 ]
  %287 = tail call i64 @llvm.smin.i64(i64 %285, i64 %274)
  %288 = sub nsw i64 0, %287
  %289 = getelementptr inbounds double, ptr %273, i64 %288
  %290 = icmp sgt i64 %287, 1
  br i1 %290, label %291, label %297, !prof !15

291:                                              ; preds = %284
  %292 = shl nuw nsw i64 %287, 3
  %293 = getelementptr inbounds double, ptr %286, i64 %288
  tail call void @llvm.memmove.p0.p0.i64(ptr nonnull align 8 %293, ptr nonnull align 8 %289, i64 %292, i1 false), !noalias !232
  %294 = ptrtoint ptr %270 to i64
  %295 = sub i64 %275, %294
  %296 = ashr exact i64 %295, 3
  br label %302

297:                                              ; preds = %284
  %298 = icmp eq i64 %287, 1
  br i1 %298, label %299, label %302

299:                                              ; preds = %297
  %300 = getelementptr inbounds i8, ptr %286, i64 -8
  %301 = load double, ptr %289, align 8, !tbaa !16, !noalias !232
  store double %301, ptr %300, align 8, !tbaa !16, !noalias !232
  br label %302

302:                                              ; preds = %299, %297, %291
  %303 = phi i64 [ %296, %291 ], [ %278, %297 ], [ %278, %299 ]
  %304 = phi ptr [ %270, %291 ], [ %272, %297 ], [ %272, %299 ]
  %305 = sub nsw i64 %303, %287
  %306 = icmp sgt i64 %305, -1
  br i1 %306, label %307, label %313

307:                                              ; preds = %302
  %308 = icmp samesign ult i64 %305, 64
  br i1 %308, label %309, label %311

309:                                              ; preds = %307
  %310 = getelementptr inbounds double, ptr %271, i64 %288
  br label %323

311:                                              ; preds = %307
  %312 = lshr i64 %305, 6
  br label %315

313:                                              ; preds = %302
  %314 = ashr i64 %305, 6
  br label %315

315:                                              ; preds = %313, %311
  %316 = phi i64 [ %312, %311 ], [ %314, %313 ]
  %317 = getelementptr inbounds ptr, ptr %268, i64 %316
  %318 = load ptr, ptr %317, align 8, !tbaa !40, !noalias !232
  %319 = getelementptr inbounds nuw i8, ptr %318, i64 512
  %320 = shl nsw i64 %316, 6
  %321 = sub nsw i64 %305, %320
  %322 = getelementptr inbounds double, ptr %318, i64 %321
  br label %323

323:                                              ; preds = %315, %309
  %324 = phi ptr [ %268, %309 ], [ %317, %315 ]
  %325 = phi ptr [ %269, %309 ], [ %319, %315 ]
  %326 = phi ptr [ %270, %309 ], [ %318, %315 ]
  %327 = phi ptr [ %304, %309 ], [ %318, %315 ]
  %328 = phi ptr [ %310, %309 ], [ %322, %315 ]
  %329 = sub nsw i64 %274, %287
  %330 = icmp sgt i64 %329, 0
  br i1 %330, label %267, label %331, !llvm.loop !224

331:                                              ; preds = %168, %323, %252, %99
  %332 = phi ptr [ %103, %99 ], [ %255, %252 ], [ %328, %323 ], [ %173, %168 ]
  %333 = phi ptr [ %102, %99 ], [ %257, %252 ], [ %327, %323 ], [ %172, %168 ]
  %334 = phi ptr [ %101, %99 ], [ %259, %252 ], [ %325, %323 ], [ %170, %168 ]
  %335 = phi ptr [ %100, %99 ], [ %261, %252 ], [ %324, %323 ], [ %169, %168 ]
  store ptr %332, ptr %0, align 8, !tbaa !23
  %336 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %333, ptr %336, align 8, !tbaa !30
  %337 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %334, ptr %337, align 8, !tbaa !31
  %338 = getelementptr inbounds nuw i8, ptr %0, i64 24
  store ptr %335, ptr %338, align 8, !tbaa !32
  ret void
}

; Function Attrs: nounwind
declare void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16), ptr noundef) local_unnamed_addr #16

; Function Attrs: nounwind
declare void @_ZNSt8__detail15_List_node_base11_M_transferEPS0_S1_(ptr noundef nonnull align 8 dereferenceable(16), ptr noundef, ptr noundef) local_unnamed_addr #16

; Function Attrs: nounwind
declare void @_ZNSt8__detail15_List_node_base4swapERS0_S1_(ptr noundef nonnull align 8 dereferenceable(16), ptr noundef nonnull align 8 dereferenceable(16)) local_unnamed_addr #16

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #17

; Function Attrs: mustprogress nofree nounwind willreturn memory(read)
declare noundef ptr @_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base(ptr noundef) local_unnamed_addr #18

; Function Attrs: mustprogress nofree nounwind willreturn memory(read)
declare noundef ptr @_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base(ptr noundef) local_unnamed_addr #18

; Function Attrs: nounwind
declare void @_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_node_baseS0_RS_(i1 noundef, ptr noundef, ptr noundef, ptr noundef nonnull align 8 dereferenceable(32)) local_unnamed_addr #16

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local i64 @_ZNSt8_Rb_treeIddSt9_IdentityIdESt4lessIdESaIdEE16_M_insert_equal_IRdNS5_11_Alloc_nodeEEESt17_Rb_tree_iteratorIdESt23_Rb_tree_const_iteratorIdEOT_RT0_(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr %1, ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull align 8 dereferenceable(8) %3) local_unnamed_addr #2 comdat {
  %5 = tail call [2 x i64] @_ZNSt8_Rb_treeIddSt9_IdentityIdESt4lessIdESaIdEE28_M_get_insert_hint_equal_posESt23_Rb_tree_const_iteratorIdERKd(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr %1, ptr noundef nonnull align 8 dereferenceable(8) %2)
  %6 = extractvalue [2 x i64] %5, 1
  %7 = icmp eq i64 %6, 0
  br i1 %7, label %24, label %8

8:                                                ; preds = %4
  %9 = inttoptr i64 %6 to ptr
  %10 = extractvalue [2 x i64] %5, 0
  %11 = icmp ne i64 %10, 0
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %13 = icmp eq ptr %12, %9
  %14 = select i1 %11, i1 true, i1 %13
  %15 = load double, ptr %2, align 8, !tbaa !16
  br i1 %14, label %20, label %16

16:                                               ; preds = %8
  %17 = getelementptr inbounds nuw i8, ptr %9, i64 32
  %18 = load double, ptr %17, align 8, !tbaa !16
  %19 = fcmp olt double %15, %18
  br label %20

20:                                               ; preds = %8, %16
  %21 = phi i1 [ true, %8 ], [ %19, %16 ]
  %22 = tail call noalias noundef nonnull dereferenceable(40) ptr @_Znwm(i64 noundef 40) #23
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 32
  store double %15, ptr %23, align 8, !tbaa !16
  tail call void @_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_node_baseS0_RS_(i1 noundef %21, ptr noundef nonnull %22, ptr noundef nonnull %9, ptr noundef nonnull align 8 dereferenceable(32) %12) #25
  br label %48

24:                                               ; preds = %4
  %25 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %26 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %27 = load ptr, ptr %25, align 8, !tbaa !94
  %28 = icmp eq ptr %27, null
  %29 = load double, ptr %2, align 8, !tbaa !16
  br i1 %28, label %43, label %30

30:                                               ; preds = %24, %30
  %31 = phi ptr [ %37, %30 ], [ %27, %24 ]
  %32 = getelementptr inbounds nuw i8, ptr %31, i64 32
  %33 = load double, ptr %32, align 8, !tbaa !16
  %34 = fcmp olt double %33, %29
  %35 = select i1 %34, i64 24, i64 16
  %36 = getelementptr inbounds nuw i8, ptr %31, i64 %35
  %37 = load ptr, ptr %36, align 8, !tbaa !94
  %38 = icmp eq ptr %37, null
  br i1 %38, label %39, label %30, !llvm.loop !235

39:                                               ; preds = %30
  %40 = icmp eq ptr %31, %26
  %41 = fcmp uge double %33, %29
  %42 = select i1 %40, i1 true, i1 %41
  br label %43

43:                                               ; preds = %39, %24
  %44 = phi ptr [ %26, %24 ], [ %31, %39 ]
  %45 = phi i1 [ true, %24 ], [ %42, %39 ]
  %46 = tail call noalias noundef nonnull dereferenceable(40) ptr @_Znwm(i64 noundef 40) #23
  %47 = getelementptr inbounds nuw i8, ptr %46, i64 32
  store double %29, ptr %47, align 8, !tbaa !16
  tail call void @_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_node_baseS0_RS_(i1 noundef %45, ptr noundef nonnull %46, ptr noundef nonnull %44, ptr noundef nonnull align 8 dereferenceable(32) %26) #25
  br label %48

48:                                               ; preds = %43, %20
  %49 = phi ptr [ %22, %20 ], [ %46, %43 ]
  %50 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %51 = load i64, ptr %50, align 8, !tbaa !93
  %52 = add i64 %51, 1
  store i64 %52, ptr %50, align 8, !tbaa !93
  %53 = ptrtoint ptr %49 to i64
  ret i64 %53
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local [2 x i64] @_ZNSt8_Rb_treeIddSt9_IdentityIdESt4lessIdESaIdEE28_M_get_insert_hint_equal_posESt23_Rb_tree_const_iteratorIdERKd(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr %1, ptr noundef nonnull align 8 dereferenceable(8) %2) local_unnamed_addr #2 comdat {
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %5 = icmp eq ptr %4, %1
  br i1 %5, label %6, label %32

6:                                                ; preds = %3
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %8 = load i64, ptr %7, align 8, !tbaa !93
  %9 = icmp eq i64 %8, 0
  br i1 %9, label %17, label %10

10:                                               ; preds = %6
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %12 = load ptr, ptr %11, align 8, !tbaa !94
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 32
  %14 = load double, ptr %2, align 8, !tbaa !16
  %15 = load double, ptr %13, align 8, !tbaa !16
  %16 = fcmp olt double %14, %15
  br i1 %16, label %17, label %80

17:                                               ; preds = %10, %6
  %18 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %19 = load ptr, ptr %18, align 8, !tbaa !94
  %20 = icmp eq ptr %19, null
  br i1 %20, label %80, label %21

21:                                               ; preds = %17
  %22 = load double, ptr %2, align 8, !tbaa !16
  br label %23

23:                                               ; preds = %23, %21
  %24 = phi ptr [ %19, %21 ], [ %30, %23 ]
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 32
  %26 = load double, ptr %25, align 8, !tbaa !16
  %27 = fcmp olt double %22, %26
  %28 = select i1 %27, i64 16, i64 24
  %29 = getelementptr inbounds nuw i8, ptr %24, i64 %28
  %30 = load ptr, ptr %29, align 8, !tbaa !94
  %31 = icmp eq ptr %30, null
  br i1 %31, label %80, label %23, !llvm.loop !236

32:                                               ; preds = %3
  %33 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %34 = load double, ptr %33, align 8, !tbaa !16
  %35 = load double, ptr %2, align 8, !tbaa !16
  %36 = fcmp olt double %34, %35
  br i1 %36, label %65, label %37

37:                                               ; preds = %32
  %38 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %39 = load ptr, ptr %38, align 8, !tbaa !94
  %40 = icmp eq ptr %39, %1
  br i1 %40, label %80, label %41

41:                                               ; preds = %37
  %42 = tail call noundef ptr @_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base(ptr noundef nonnull %1) #27
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 32
  %44 = load double, ptr %43, align 8, !tbaa !16
  %45 = fcmp olt double %35, %44
  br i1 %45, label %52, label %46

46:                                               ; preds = %41
  %47 = getelementptr inbounds nuw i8, ptr %42, i64 24
  %48 = load ptr, ptr %47, align 8, !tbaa !105
  %49 = icmp eq ptr %48, null
  %50 = select i1 %49, ptr null, ptr %1
  %51 = select i1 %49, ptr %42, ptr %1
  br label %80

52:                                               ; preds = %41
  %53 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %54 = load ptr, ptr %53, align 8, !tbaa !94
  %55 = icmp eq ptr %54, null
  br i1 %55, label %80, label %56

56:                                               ; preds = %52, %56
  %57 = phi ptr [ %63, %56 ], [ %54, %52 ]
  %58 = getelementptr inbounds nuw i8, ptr %57, i64 32
  %59 = load double, ptr %58, align 8, !tbaa !16
  %60 = fcmp olt double %35, %59
  %61 = select i1 %60, i64 16, i64 24
  %62 = getelementptr inbounds nuw i8, ptr %57, i64 %61
  %63 = load ptr, ptr %62, align 8, !tbaa !94
  %64 = icmp eq ptr %63, null
  br i1 %64, label %80, label %56, !llvm.loop !236

65:                                               ; preds = %32
  %66 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %67 = load ptr, ptr %66, align 8, !tbaa !94
  %68 = icmp eq ptr %67, %1
  br i1 %68, label %80, label %69

69:                                               ; preds = %65
  %70 = tail call noundef ptr @_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base(ptr noundef nonnull %1) #27
  %71 = getelementptr inbounds nuw i8, ptr %70, i64 32
  %72 = load double, ptr %71, align 8, !tbaa !16
  %73 = fcmp olt double %72, %35
  br i1 %73, label %80, label %74

74:                                               ; preds = %69
  %75 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %76 = load ptr, ptr %75, align 8, !tbaa !105
  %77 = icmp eq ptr %76, null
  %78 = select i1 %77, ptr null, ptr %70
  %79 = select i1 %77, ptr %1, ptr %70
  br label %80

80:                                               ; preds = %56, %23, %52, %17, %74, %46, %65, %69, %37, %10
  %81 = phi ptr [ null, %10 ], [ %1, %37 ], [ null, %65 ], [ null, %69 ], [ %50, %46 ], [ %78, %74 ], [ null, %17 ], [ null, %52 ], [ null, %23 ], [ null, %56 ]
  %82 = phi ptr [ %12, %10 ], [ %1, %37 ], [ %1, %65 ], [ null, %69 ], [ %51, %46 ], [ %79, %74 ], [ %1, %17 ], [ %4, %52 ], [ %24, %23 ], [ %57, %56 ]
  %83 = ptrtoint ptr %81 to i64
  %84 = insertvalue [2 x i64] poison, i64 %83, 0
  %85 = ptrtoint ptr %82 to i64
  %86 = insertvalue [2 x i64] %84, i64 %85, 1
  ret [2 x i64] %86
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(read)
declare noundef ptr @_ZSt18_Rb_tree_incrementPKSt18_Rb_tree_node_base(ptr noundef) local_unnamed_addr #18

; Function Attrs: nounwind
declare noundef nonnull ptr @_ZSt28_Rb_tree_rebalance_for_erasePSt18_Rb_tree_node_baseRS_(ptr noundef, ptr noundef nonnull align 8 dereferenceable(32)) local_unnamed_addr #16

; Function Attrs: nounwind
declare i32 @rand() local_unnamed_addr #16

; Function Attrs: nofree nounwind uwtable
define internal void @_GLOBAL__sub_I_stepanov_container.cpp() #19 section ".text.startup" {
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) @result_times, i8 0, i64 24, i1 false)
  %1 = tail call i32 @__cxa_atexit(ptr nonnull @_ZNSt6vectorIdSaIdEED2Ev, ptr nonnull @result_times, ptr nonnull @__dso_handle) #25
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #20

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #20

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #20

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #21

attributes #0 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind }
attributes #2 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { mustprogress nofree norecurse nosync nounwind memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { mustprogress nofree norecurse nounwind willreturn memory(errnomem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { mustprogress nocallback nofree nounwind willreturn memory(errnomem: write) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #11 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #12 = { noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #13 = { cold nofree noreturn }
attributes #14 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #15 = { inlinehint mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #16 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #17 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #18 = { mustprogress nofree nounwind willreturn memory(read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #19 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #20 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #21 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #22 = { builtin nounwind }
attributes #23 = { builtin allocsize(0) }
attributes #24 = { cold noreturn }
attributes #25 = { nounwind }
attributes #26 = { noreturn nounwind }
attributes #27 = { nounwind willreturn memory(read) }
attributes #28 = { noreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"_ZTSNSt12_Vector_baseIdSaIdEE17_Vector_impl_dataE", !8, i64 0, !8, i64 8, !8, i64 16}
!8 = !{!"p1 double", !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
!12 = !{!7, !8, i64 16}
!13 = distinct !{!13, !14}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!16 = !{!17, !17, i64 0}
!17 = !{!"double", !10, i64 0}
!18 = distinct !{!18, !14}
!19 = distinct !{!19, !14}
!20 = !{!"branch_weights", !"expected", i32 -2147483648, i32 0}
!21 = distinct !{!21, !14}
!22 = distinct !{!22, !14}
!23 = !{!24, !8, i64 0}
!24 = !{!"_ZTSSt15_Deque_iteratorIdRdPdE", !8, i64 0, !8, i64 8, !8, i64 16, !25, i64 24}
!25 = !{!"p2 double", !26, i64 0}
!26 = !{!"any p2 pointer", !9, i64 0}
!27 = !{!28}
!28 = distinct !{!28, !29, !"_ZNSt5dequeIdSaIdEE5beginEv: argument 0"}
!29 = distinct !{!29, !"_ZNSt5dequeIdSaIdEE5beginEv"}
!30 = !{!24, !8, i64 8}
!31 = !{!24, !8, i64 16}
!32 = !{!24, !25, i64 24}
!33 = !{!34, !36, !38}
!34 = distinct !{!34, !35, !"_ZSt14__copy_move_a1ILb0EPddEN9__gnu_cxx11__enable_ifIXsr23__is_random_access_iterIT0_EE7__valueESt15_Deque_iteratorIT1_RS5_PS5_EE6__typeES3_S3_S8_: argument 0"}
!35 = distinct !{!35, !"_ZSt14__copy_move_a1ILb0EPddEN9__gnu_cxx11__enable_ifIXsr23__is_random_access_iterIT0_EE7__valueESt15_Deque_iteratorIT1_RS5_PS5_EE6__typeES3_S3_S8_"}
!36 = distinct !{!36, !37, !"_ZSt13__copy_move_aILb0EPdSt15_Deque_iteratorIdRdS0_EET1_T0_S5_S4_: argument 0"}
!37 = distinct !{!37, !"_ZSt13__copy_move_aILb0EPdSt15_Deque_iteratorIdRdS0_EET1_T0_S5_S4_"}
!38 = distinct !{!38, !39, !"_ZSt4copyIPdSt15_Deque_iteratorIdRdS0_EET0_T_S5_S4_: argument 0"}
!39 = distinct !{!39, !"_ZSt4copyIPdSt15_Deque_iteratorIdRdS0_EET0_T_S5_S4_"}
!40 = !{!8, !8, i64 0}
!41 = distinct !{!41, !14}
!42 = !{!43}
!43 = distinct !{!43, !44, !"_ZNSt5dequeIdSaIdEE5beginEv: argument 0"}
!44 = distinct !{!44, !"_ZNSt5dequeIdSaIdEE5beginEv"}
!45 = !{!46}
!46 = distinct !{!46, !47, !"_ZNSt5dequeIdSaIdEE3endEv: argument 0"}
!47 = distinct !{!47, !"_ZNSt5dequeIdSaIdEE3endEv"}
!48 = !{!49}
!49 = distinct !{!49, !50, !"_ZNSt5dequeIdSaIdEE5beginEv: argument 0"}
!50 = distinct !{!50, !"_ZNSt5dequeIdSaIdEE5beginEv"}
!51 = !{!52}
!52 = distinct !{!52, !53, !"_ZNSt5dequeIdSaIdEE3endEv: argument 0"}
!53 = distinct !{!53, !"_ZNSt5dequeIdSaIdEE3endEv"}
!54 = !{!55, !57, !59}
!55 = distinct !{!55, !56, !"_ZSt15__adjacent_findISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops19_Iter_equal_to_iterEET_S7_S7_T0_: argument 0"}
!56 = distinct !{!56, !"_ZSt15__adjacent_findISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops19_Iter_equal_to_iterEET_S7_S7_T0_"}
!57 = distinct !{!57, !58, !"_ZSt8__uniqueISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops19_Iter_equal_to_iterEET_S7_S7_T0_: argument 0"}
!58 = distinct !{!58, !"_ZSt8__uniqueISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops19_Iter_equal_to_iterEET_S7_S7_T0_"}
!59 = distinct !{!59, !60, !"_ZSt6uniqueISt15_Deque_iteratorIdRdPdEET_S4_S4_: argument 0"}
!60 = distinct !{!60, !"_ZSt6uniqueISt15_Deque_iteratorIdRdPdEET_S4_S4_"}
!61 = distinct !{!61, !14}
!62 = !{!57, !59}
!63 = distinct !{!63, !14}
!64 = !{!65, !25, i64 0}
!65 = !{!"_ZTSNSt11_Deque_baseIdSaIdEE16_Deque_impl_dataE", !25, i64 0, !66, i64 8, !24, i64 16, !24, i64 48}
!66 = !{!"long", !10, i64 0}
!67 = distinct !{!67, !14}
!68 = !{!65, !66, i64 8}
!69 = !{!65, !25, i64 40}
!70 = !{!65, !25, i64 72}
!71 = !{!72, !73, i64 8}
!72 = !{!"_ZTSNSt8__detail15_List_node_baseE", !73, i64 0, !73, i64 8}
!73 = !{!"p1 _ZTSNSt8__detail15_List_node_baseE", !9, i64 0}
!74 = !{!72, !73, i64 0}
!75 = !{!76, !66, i64 16}
!76 = !{!"_ZTSNSt8__detail17_List_node_headerE", !72, i64 0, !66, i64 16}
!77 = !{!78, !66, i64 16}
!78 = !{!"_ZTSNSt7__cxx1110_List_baseIdSaIdEEE", !79, i64 0}
!79 = !{!"_ZTSNSt7__cxx1110_List_baseIdSaIdEE10_List_implE", !76, i64 0}
!80 = distinct !{!80, !14}
!81 = distinct !{!81, !14}
!82 = distinct !{!82, !14}
!83 = distinct !{!83, !14}
!84 = distinct !{!84, !14}
!85 = distinct !{!85, !14}
!86 = !{!87, !90, i64 8}
!87 = !{!"_ZTSSt15_Rb_tree_header", !88, i64 0, !66, i64 32}
!88 = !{!"_ZTSSt18_Rb_tree_node_base", !89, i64 0, !90, i64 8, !90, i64 16, !90, i64 24}
!89 = !{!"_ZTSSt14_Rb_tree_color", !10, i64 0}
!90 = !{!"p1 _ZTSSt18_Rb_tree_node_base", !9, i64 0}
!91 = !{!87, !90, i64 16}
!92 = !{!87, !90, i64 24}
!93 = !{!87, !66, i64 32}
!94 = !{!90, !90, i64 0}
!95 = distinct !{!95, !14}
!96 = distinct !{!96, !14}
!97 = !{!98, !98, i64 0}
!98 = !{!"p1 _ZTSSt8_Rb_treeIddSt9_IdentityIdESt4lessIdESaIdEE", !9, i64 0}
!99 = distinct !{!99, !14}
!100 = distinct !{!100, !14}
!101 = !{!102, !102, i64 0}
!102 = !{!"int", !10, i64 0}
!103 = !{!7, !8, i64 8}
!104 = distinct !{!104, !14}
!105 = !{!88, !90, i64 24}
!106 = !{!88, !90, i64 16}
!107 = distinct !{!107, !14}
!108 = distinct !{!108, !14}
!109 = distinct !{!109, !14}
!110 = distinct !{!110, !14}
!111 = distinct !{!111, !14}
!112 = distinct !{!112, !14}
!113 = distinct !{!113, !14}
!114 = distinct !{!114, !14}
!115 = distinct !{!115, !14}
!116 = distinct !{!116, !14}
!117 = distinct !{!117, !14}
!118 = distinct !{!118, !14}
!119 = distinct !{!119, !14}
!120 = distinct !{!120, !14}
!121 = distinct !{!121, !14}
!122 = distinct !{!122, !14}
!123 = distinct !{!123, !14}
!124 = distinct !{!124, !14}
!125 = distinct !{!125, !14}
!126 = distinct !{!126, !14}
!127 = distinct !{!127, !14}
!128 = distinct !{!128, !14}
!129 = distinct !{!129, !14}
!130 = distinct !{!130, !14}
!131 = distinct !{!131, !14}
!132 = !{!65, !8, i64 56}
!133 = !{!65, !8, i64 48}
!134 = distinct !{!134, !14, !135, !136}
!135 = !{!"llvm.loop.isvectorized", i32 1}
!136 = !{!"llvm.loop.unroll.runtime.disable"}
!137 = distinct !{!137, !14, !136, !135}
!138 = distinct !{!138, !14}
!139 = !{!65, !8, i64 16}
!140 = !{!9, !9, i64 0}
!141 = !{i64 0, i64 8, !40, i64 8, i64 8, !40, i64 16, i64 8, !40, i64 24, i64 8, !142}
!142 = !{!25, !25, i64 0}
!143 = distinct !{!143, !14}
!144 = !{!145}
!145 = distinct !{!145, !146, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl: argument 0"}
!146 = distinct !{!146, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl"}
!147 = !{!148}
!148 = distinct !{!148, !149, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl: argument 0"}
!149 = distinct !{!149, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl"}
!150 = distinct !{!150, !14}
!151 = distinct !{!151, !14}
!152 = distinct !{!152, !14}
!153 = !{!154}
!154 = distinct !{!154, !155, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl: argument 0"}
!155 = distinct !{!155, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl"}
!156 = !{!157}
!157 = distinct !{!157, !158, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl: argument 0"}
!158 = distinct !{!158, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl"}
!159 = !{!160}
!160 = distinct !{!160, !161, !"_ZStmiRKSt15_Deque_iteratorIdRdPdEl: argument 0"}
!161 = distinct !{!161, !"_ZStmiRKSt15_Deque_iteratorIdRdPdEl"}
!162 = !{!163}
!163 = distinct !{!163, !164, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl: argument 0"}
!164 = distinct !{!164, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl"}
!165 = !{!166}
!166 = distinct !{!166, !167, !"_ZSt21__unguarded_partitionISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops15_Iter_less_iterEET_S7_S7_S7_T0_: argument 0"}
!167 = distinct !{!167, !"_ZSt21__unguarded_partitionISt15_Deque_iteratorIdRdPdEN9__gnu_cxx5__ops15_Iter_less_iterEET_S7_S7_S7_T0_"}
!168 = distinct !{!168, !14}
!169 = distinct !{!169, !14}
!170 = distinct !{!170, !14}
!171 = !{!172}
!172 = distinct !{!172, !173, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl: argument 0"}
!173 = distinct !{!173, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl"}
!174 = distinct !{!174, !14}
!175 = distinct !{!175, !14}
!176 = !{!177}
!177 = distinct !{!177, !178, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl: argument 0"}
!178 = distinct !{!178, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl"}
!179 = !{!180}
!180 = distinct !{!180, !181, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl: argument 0"}
!181 = distinct !{!181, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl"}
!182 = !{!183}
!183 = distinct !{!183, !184, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl: argument 0"}
!184 = distinct !{!184, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl"}
!185 = !{!186}
!186 = distinct !{!186, !187, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl: argument 0"}
!187 = distinct !{!187, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl"}
!188 = distinct !{!188, !14}
!189 = !{!190}
!190 = distinct !{!190, !191, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl: argument 0"}
!191 = distinct !{!191, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl"}
!192 = !{!193}
!193 = distinct !{!193, !194, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl: argument 0"}
!194 = distinct !{!194, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl"}
!195 = !{!196}
!196 = distinct !{!196, !197, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl: argument 0"}
!197 = distinct !{!197, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl"}
!198 = !{!199}
!199 = distinct !{!199, !200, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl: argument 0"}
!200 = distinct !{!200, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl"}
!201 = distinct !{!201, !14}
!202 = !{!203}
!203 = distinct !{!203, !204, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl: argument 0"}
!204 = distinct !{!204, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl"}
!205 = !{!206}
!206 = distinct !{!206, !207, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl: argument 0"}
!207 = distinct !{!207, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl"}
!208 = !{!209}
!209 = distinct !{!209, !210, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl: argument 0"}
!210 = distinct !{!210, !"_ZStplRKSt15_Deque_iteratorIdRdPdEl"}
!211 = !{!212}
!212 = distinct !{!212, !213, !"_ZSt13move_backwardISt15_Deque_iteratorIdRdPdES3_ET0_T_S5_S4_: argument 0"}
!213 = distinct !{!213, !"_ZSt13move_backwardISt15_Deque_iteratorIdRdPdES3_ET0_T_S5_S4_"}
!214 = !{!215, !212}
!215 = distinct !{!215, !216, !"_ZSt22__copy_move_backward_aILb1ESt15_Deque_iteratorIdRdPdES3_ET1_T0_S5_S4_: argument 0"}
!216 = distinct !{!216, !"_ZSt22__copy_move_backward_aILb1ESt15_Deque_iteratorIdRdPdES3_ET1_T0_S5_S4_"}
!217 = !{!218, !215, !212}
!218 = distinct !{!218, !219, !"_ZSt23__copy_move_backward_a1ILb1EdRdPddESt15_Deque_iteratorIT3_RS3_PS3_ES2_IT0_T1_T2_ESA_S6_: argument 0"}
!219 = distinct !{!219, !"_ZSt23__copy_move_backward_a1ILb1EdRdPddESt15_Deque_iteratorIT3_RS3_PS3_ES2_IT0_T1_T2_ESA_S6_"}
!220 = distinct !{!220, !14}
!221 = !{!222}
!222 = distinct !{!222, !223, !"_ZSt23__copy_move_backward_a1ILb1EPddEN9__gnu_cxx11__enable_ifIXsr23__is_random_access_iterIT0_EE7__valueESt15_Deque_iteratorIT1_RS5_PS5_EE6__typeES3_S3_S8_: argument 0"}
!223 = distinct !{!223, !"_ZSt23__copy_move_backward_a1ILb1EPddEN9__gnu_cxx11__enable_ifIXsr23__is_random_access_iterIT0_EE7__valueESt15_Deque_iteratorIT1_RS5_PS5_EE6__typeES3_S3_S8_"}
!224 = distinct !{!224, !14}
!225 = !{!226}
!226 = distinct !{!226, !227, !"_ZSt23__copy_move_backward_a1ILb1EPddEN9__gnu_cxx11__enable_ifIXsr23__is_random_access_iterIT0_EE7__valueESt15_Deque_iteratorIT1_RS5_PS5_EE6__typeES3_S3_S8_: argument 0"}
!227 = distinct !{!227, !"_ZSt23__copy_move_backward_a1ILb1EPddEN9__gnu_cxx11__enable_ifIXsr23__is_random_access_iterIT0_EE7__valueESt15_Deque_iteratorIT1_RS5_PS5_EE6__typeES3_S3_S8_"}
!228 = !{!229}
!229 = distinct !{!229, !230, !"_ZSt23__copy_move_backward_a1ILb1EPddEN9__gnu_cxx11__enable_ifIXsr23__is_random_access_iterIT0_EE7__valueESt15_Deque_iteratorIT1_RS5_PS5_EE6__typeES3_S3_S8_: argument 0"}
!230 = distinct !{!230, !"_ZSt23__copy_move_backward_a1ILb1EPddEN9__gnu_cxx11__enable_ifIXsr23__is_random_access_iterIT0_EE7__valueESt15_Deque_iteratorIT1_RS5_PS5_EE6__typeES3_S3_S8_"}
!231 = distinct !{!231, !14}
!232 = !{!233}
!233 = distinct !{!233, !234, !"_ZSt23__copy_move_backward_a1ILb1EPddEN9__gnu_cxx11__enable_ifIXsr23__is_random_access_iterIT0_EE7__valueESt15_Deque_iteratorIT1_RS5_PS5_EE6__typeES3_S3_S8_: argument 0"}
!234 = distinct !{!234, !"_ZSt23__copy_move_backward_a1ILb1EPddEN9__gnu_cxx11__enable_ifIXsr23__is_random_access_iterIT0_EE7__valueESt15_Deque_iteratorIT1_RS5_PS5_EE6__typeES3_S3_S8_"}
!235 = distinct !{!235, !14}
!236 = distinct !{!236, !14}
