; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc-C++/Large/ray.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc-C++/Large/ray.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"struct.std::numeric_limits" = type { i8 }
%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }
%struct.Vec = type { double, double, double }
%"struct.std::pair" = type { double, %struct.Vec }
%struct.Ray = type { %struct.Vec, %struct.Vec }
%"class.std::__cxx11::list" = type { %"class.std::__cxx11::_List_base" }
%"class.std::__cxx11::_List_base" = type { %"struct.std::__cxx11::_List_base<Scene *, std::allocator<Scene *>>::_List_impl" }
%"struct.std::__cxx11::_List_base<Scene *, std::allocator<Scene *>>::_List_impl" = type { %"struct.std::__detail::_List_node_header" }
%"struct.std::__detail::_List_node_header" = type { %"struct.std::__detail::_List_node_base", i64 }
%"struct.std::__detail::_List_node_base" = type { ptr, ptr }

$_ZN5SceneD2Ev = comdat any

$_ZN6SphereD0Ev = comdat any

$_ZNK6Sphere9intersectERKSt4pairId3VecERK3Ray = comdat any

$_ZN5GroupD2Ev = comdat any

$_ZN5GroupD0Ev = comdat any

$_ZNK5Group9intersectERKSt4pairId3VecERK3Ray = comdat any

$_ZTV6Sphere = comdat any

$_ZTI6Sphere = comdat any

$_ZTS6Sphere = comdat any

$_ZTI5Scene = comdat any

$_ZTS5Scene = comdat any

$_ZTV5Group = comdat any

$_ZTI5Group = comdat any

$_ZTS5Group = comdat any

@real = dso_local local_unnamed_addr global %"struct.std::numeric_limits" zeroinitializer, align 1
@delta = dso_local local_unnamed_addr global double 0x3E50000000000000, align 8
@infinity = dso_local local_unnamed_addr global double 0x7FF0000000000000, align 8
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [4 x i8] c"P5\0A\00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c" \00", align 1
@.str.2 = private unnamed_addr constant [6 x i8] c"\0A255\0A\00", align 1
@_ZTV6Sphere = linkonce_odr dso_local unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI6Sphere, ptr @_ZN5SceneD2Ev, ptr @_ZN6SphereD0Ev, ptr @_ZNK6Sphere9intersectERKSt4pairId3VecERK3Ray] }, comdat, align 8
@_ZTI6Sphere = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS6Sphere, ptr @_ZTI5Scene }, comdat, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
@_ZTS6Sphere = linkonce_odr dso_local constant [8 x i8] c"6Sphere\00", comdat, align 1
@_ZTI5Scene = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS5Scene }, comdat, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@_ZTS5Scene = linkonce_odr dso_local constant [7 x i8] c"5Scene\00", comdat, align 1
@_ZTV5Group = linkonce_odr dso_local unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI5Group, ptr @_ZN5GroupD2Ev, ptr @_ZN5GroupD0Ev, ptr @_ZNK5Group9intersectERKSt4pairId3VecERK3Ray] }, comdat, align 8
@_ZTI5Group = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS5Group, ptr @_ZTI5Scene }, comdat, align 8
@_ZTS5Group = linkonce_odr dso_local constant [7 x i8] c"5Group\00", comdat, align 1
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @sqrt(double noundef) local_unnamed_addr #0

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local %struct.Vec @_ZplRK3VecS1_(ptr noundef nonnull readonly align 8 captures(none) dereferenceable(24) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(24) %1) local_unnamed_addr #1 {
  %3 = load double, ptr %0, align 8, !tbaa !6
  %4 = load double, ptr %1, align 8, !tbaa !6
  %5 = fadd double %3, %4
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %7 = load double, ptr %6, align 8, !tbaa !11
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %9 = load double, ptr %8, align 8, !tbaa !11
  %10 = fadd double %7, %9
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %12 = load double, ptr %11, align 8, !tbaa !12
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %14 = load double, ptr %13, align 8, !tbaa !12
  %15 = fadd double %12, %14
  %16 = insertvalue %struct.Vec poison, double %5, 0
  %17 = insertvalue %struct.Vec %16, double %10, 1
  %18 = insertvalue %struct.Vec %17, double %15, 2
  ret %struct.Vec %18
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local %struct.Vec @_ZmiRK3VecS1_(ptr noundef nonnull readonly align 8 captures(none) dereferenceable(24) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(24) %1) local_unnamed_addr #1 {
  %3 = load double, ptr %0, align 8, !tbaa !6
  %4 = load double, ptr %1, align 8, !tbaa !6
  %5 = fsub double %3, %4
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %7 = load double, ptr %6, align 8, !tbaa !11
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %9 = load double, ptr %8, align 8, !tbaa !11
  %10 = fsub double %7, %9
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %12 = load double, ptr %11, align 8, !tbaa !12
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %14 = load double, ptr %13, align 8, !tbaa !12
  %15 = fsub double %12, %14
  %16 = insertvalue %struct.Vec poison, double %5, 0
  %17 = insertvalue %struct.Vec %16, double %10, 1
  %18 = insertvalue %struct.Vec %17, double %15, 2
  ret %struct.Vec %18
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local %struct.Vec @_ZmldRK3Vec(double noundef %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(24) %1) local_unnamed_addr #1 {
  %3 = load double, ptr %1, align 8, !tbaa !6
  %4 = fmul double %0, %3
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load double, ptr %5, align 8, !tbaa !11
  %7 = fmul double %0, %6
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %9 = load double, ptr %8, align 8, !tbaa !12
  %10 = fmul double %0, %9
  %11 = insertvalue %struct.Vec poison, double %4, 0
  %12 = insertvalue %struct.Vec %11, double %7, 1
  %13 = insertvalue %struct.Vec %12, double %10, 2
  ret %struct.Vec %13
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local noundef double @_Z3dotRK3VecS1_(ptr noundef nonnull readonly align 8 captures(none) dereferenceable(24) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(24) %1) local_unnamed_addr #1 {
  %3 = load double, ptr %0, align 8, !tbaa !6
  %4 = load double, ptr %1, align 8, !tbaa !6
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %6 = load double, ptr %5, align 8, !tbaa !11
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %8 = load double, ptr %7, align 8, !tbaa !11
  %9 = fmul double %6, %8
  %10 = tail call double @llvm.fmuladd.f64(double %3, double %4, double %9)
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %12 = load double, ptr %11, align 8, !tbaa !12
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %14 = load double, ptr %13, align 8, !tbaa !12
  %15 = tail call double @llvm.fmuladd.f64(double %12, double %14, double %10)
  ret double %15
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local %struct.Vec @_Z7unitiseRK3Vec(ptr noundef nonnull readonly align 8 captures(none) dereferenceable(24) %0) local_unnamed_addr #1 {
  %2 = load double, ptr %0, align 8, !tbaa !6
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %4 = load double, ptr %3, align 8, !tbaa !11
  %5 = fmul double %4, %4
  %6 = tail call double @llvm.fmuladd.f64(double %2, double %2, double %5)
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %8 = load double, ptr %7, align 8, !tbaa !12
  %9 = tail call noundef double @llvm.fmuladd.f64(double %8, double %8, double %6)
  %10 = tail call double @llvm.sqrt.f64(double %9)
  %11 = fdiv double 1.000000e+00, %10
  %12 = fmul double %2, %11
  %13 = fmul double %4, %11
  %14 = fmul double %8, %11
  %15 = insertvalue %struct.Vec poison, double %12, 0
  %16 = insertvalue %struct.Vec %15, double %13, 1
  %17 = insertvalue %struct.Vec %16, double %14, 2
  ret %struct.Vec %17
}

; Function Attrs: mustprogress uwtable
define dso_local %"struct.std::pair" @_Z9intersectRK3RayRK5Scene(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) local_unnamed_addr #3 {
  %3 = alloca %"struct.std::pair", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #14
  %4 = load double, ptr @infinity, align 8, !tbaa !13
  store double %4, ptr %3, align 8, !tbaa !14
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 8
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %5, i8 0, i64 24, i1 false)
  %6 = load ptr, ptr %1, align 8, !tbaa !16
  %7 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %8 = load ptr, ptr %7, align 8
  %9 = call %"struct.std::pair" %8(ptr noundef nonnull align 8 dereferenceable(8) %1, ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(48) %0)
  %10 = extractvalue %"struct.std::pair" %9, 1
  %11 = extractvalue %struct.Vec %10, 0
  %12 = extractvalue %struct.Vec %10, 1
  %13 = extractvalue %struct.Vec %10, 2
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #14
  %14 = insertvalue %"struct.std::pair" %9, double %11, 1, 0
  %15 = insertvalue %"struct.std::pair" %14, double %12, 1, 1
  %16 = insertvalue %"struct.std::pair" %15, double %13, 1, 2
  ret %"struct.std::pair" %16
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #4

; Function Attrs: mustprogress uwtable
define dso_local noundef double @_Z9ray_traceRK3VecRK3RayRK5Scene(ptr noundef nonnull readonly align 8 captures(none) dereferenceable(24) %0, ptr noundef nonnull align 8 dereferenceable(48) %1, ptr noundef nonnull align 8 dereferenceable(8) %2) local_unnamed_addr #3 {
  %4 = alloca %"struct.std::pair", align 8
  %5 = alloca %"struct.std::pair", align 8
  %6 = alloca %struct.Ray, align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #14
  %7 = load double, ptr @infinity, align 8, !tbaa !13
  store double %7, ptr %5, align 8, !tbaa !14
  %8 = getelementptr inbounds nuw i8, ptr %5, i64 8
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %8, i8 0, i64 24, i1 false)
  %9 = load ptr, ptr %2, align 8, !tbaa !16
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %11 = load ptr, ptr %10, align 8
  %12 = call %"struct.std::pair" %11(ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(48) %1)
  %13 = extractvalue %"struct.std::pair" %12, 1
  %14 = extractvalue %struct.Vec %13, 0
  %15 = extractvalue %struct.Vec %13, 1
  %16 = extractvalue %struct.Vec %13, 2
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #14
  %17 = extractvalue %"struct.std::pair" %12, 0
  %18 = load double, ptr @infinity, align 8, !tbaa !13
  %19 = fcmp oeq double %17, %18
  br i1 %19, label %68, label %20

20:                                               ; preds = %3
  %21 = load <2 x double>, ptr %0, align 8, !tbaa !13
  %22 = extractelement <2 x double> %21, i64 1
  %23 = fmul double %15, %22
  %24 = extractelement <2 x double> %21, i64 0
  %25 = call double @llvm.fmuladd.f64(double %14, double %24, double %23)
  %26 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %27 = load double, ptr %26, align 8, !tbaa !12
  %28 = call noundef double @llvm.fmuladd.f64(double %16, double %27, double %25)
  %29 = fcmp ult double %28, 0.000000e+00
  br i1 %29, label %30, label %68

30:                                               ; preds = %20
  %31 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %33 = load double, ptr %32, align 8, !tbaa !12
  %34 = fmul double %17, %33
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %36 = load double, ptr %35, align 8, !tbaa !12
  %37 = fadd double %34, %36
  %38 = load double, ptr @delta, align 8, !tbaa !13
  %39 = fmul double %16, %38
  %40 = fadd double %37, %39
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #14
  %41 = fneg <2 x double> %21
  %42 = fneg double %27
  %43 = load <2 x double>, ptr %31, align 8, !tbaa !13
  %44 = insertelement <2 x double> poison, double %17, i64 0
  %45 = shufflevector <2 x double> %44, <2 x double> poison, <2 x i32> zeroinitializer
  %46 = fmul <2 x double> %45, %43
  %47 = load <2 x double>, ptr %1, align 8, !tbaa !13
  %48 = fadd <2 x double> %46, %47
  %49 = insertelement <2 x double> poison, double %14, i64 0
  %50 = insertelement <2 x double> %49, double %15, i64 1
  %51 = insertelement <2 x double> poison, double %38, i64 0
  %52 = shufflevector <2 x double> %51, <2 x double> poison, <2 x i32> zeroinitializer
  %53 = fmul <2 x double> %50, %52
  %54 = fadd <2 x double> %48, %53
  store <2 x double> %54, ptr %6, align 16, !tbaa !13
  %55 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store double %40, ptr %55, align 16, !tbaa !13
  %56 = getelementptr inbounds nuw i8, ptr %6, i64 24
  store <2 x double> %41, ptr %56, align 8, !tbaa !13
  %57 = getelementptr inbounds nuw i8, ptr %6, i64 40
  store double %42, ptr %57, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #14
  store double %18, ptr %4, align 8, !tbaa !14
  %58 = getelementptr inbounds nuw i8, ptr %4, i64 8
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %58, i8 0, i64 24, i1 false)
  %59 = load ptr, ptr %2, align 8, !tbaa !16
  %60 = getelementptr inbounds nuw i8, ptr %59, i64 16
  %61 = load ptr, ptr %60, align 8
  %62 = call %"struct.std::pair" %61(ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(48) %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #14
  %63 = extractvalue %"struct.std::pair" %62, 0
  %64 = load double, ptr @infinity, align 8, !tbaa !13
  %65 = fcmp olt double %63, %64
  %66 = fneg double %28
  %67 = select i1 %65, double 0.000000e+00, double %66
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #14
  br label %68

68:                                               ; preds = %30, %20, %3
  %69 = phi double [ 0.000000e+00, %3 ], [ %67, %30 ], [ 0.000000e+00, %20 ]
  ret double %69
}

; Function Attrs: mustprogress uwtable
define dso_local noundef nonnull ptr @_Z6createiRK3Vecd(i32 noundef %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(24) %1, double noundef %2) local_unnamed_addr #3 personality ptr @__gxx_personality_v0 {
  %4 = alloca %"class.std::__cxx11::list", align 8
  %5 = alloca %struct.Vec, align 16
  %6 = alloca %"class.std::__cxx11::list", align 8
  %7 = tail call noalias noundef nonnull dereferenceable(40) ptr @_Znwm(i64 noundef 40) #15
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %10 = load double, ptr %9, align 8, !tbaa !13
  store ptr getelementptr inbounds nuw inrange(-16, 24) (i8, ptr @_ZTV6Sphere, i64 16), ptr %7, align 8, !tbaa !16
  %11 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %12 = load <2 x double>, ptr %1, align 8, !tbaa !13
  store <2 x double> %12, ptr %11, align 8, !tbaa !13
  %13 = getelementptr inbounds nuw i8, ptr %7, i64 24
  store double %10, ptr %13, align 8, !tbaa !13
  %14 = getelementptr inbounds nuw i8, ptr %7, i64 32
  store double %2, ptr %14, align 8, !tbaa !18
  %15 = icmp eq i32 %0, 1
  br i1 %15, label %179, label %16

16:                                               ; preds = %3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #14
  %17 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr %4, ptr %17, align 8, !tbaa !21
  store ptr %4, ptr %4, align 8, !tbaa !25
  %18 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i64 0, ptr %18, align 8, !tbaa !26
  %19 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #15
          to label %20 unwind label %37

20:                                               ; preds = %16
  %21 = getelementptr inbounds nuw i8, ptr %19, i64 16
  store ptr %7, ptr %21, align 8, !tbaa !29
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16) %19, ptr noundef nonnull align 8 dereferenceable(24) %4) #14
  %22 = load i64, ptr %18, align 8, !tbaa !31
  %23 = add i64 %22, 1
  store i64 %23, ptr %18, align 8, !tbaa !31
  %24 = fmul double %2, 3.000000e+00
  %25 = fdiv double %24, 0x400BB67AE8584CAA
  %26 = add nsw i32 %0, -1
  %27 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %28 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %29 = fmul double %2, 5.000000e-01
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #14
  %30 = load double, ptr %1, align 8, !tbaa !6
  %31 = fsub double %30, %25
  %32 = load double, ptr %8, align 8, !tbaa !11
  %33 = fadd double %25, %32
  %34 = load double, ptr %9, align 8, !tbaa !12
  %35 = fsub double %34, %25
  store double %31, ptr %5, align 16
  store double %33, ptr %27, align 8
  store double %35, ptr %28, align 16
  %36 = invoke noundef ptr @_Z6createiRK3Vecd(i32 noundef %26, ptr noundef nonnull align 8 dereferenceable(24) %5, double noundef %29)
          to label %39 unwind label %81

37:                                               ; preds = %16
  %38 = landingpad { ptr, i32 }
          cleanup
  br label %170

39:                                               ; preds = %20
  %40 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #15
          to label %41 unwind label %81

41:                                               ; preds = %39
  %42 = getelementptr inbounds nuw i8, ptr %40, i64 16
  store ptr %36, ptr %42, align 8, !tbaa !29
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16) %40, ptr noundef nonnull align 8 dereferenceable(24) %4) #14
  %43 = load i64, ptr %18, align 8, !tbaa !31
  %44 = add i64 %43, 1
  store i64 %44, ptr %18, align 8, !tbaa !31
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #14
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #14
  %45 = load double, ptr %9, align 8, !tbaa !12
  %46 = fsub double %45, %25
  %47 = load <2 x double>, ptr %1, align 8, !tbaa !13
  %48 = insertelement <2 x double> poison, double %25, i64 0
  %49 = shufflevector <2 x double> %48, <2 x double> poison, <2 x i32> zeroinitializer
  %50 = fadd <2 x double> %49, %47
  store <2 x double> %50, ptr %5, align 16
  store double %46, ptr %28, align 16
  %51 = invoke noundef ptr @_Z6createiRK3Vecd(i32 noundef %26, ptr noundef nonnull align 8 dereferenceable(24) %5, double noundef %29)
          to label %52 unwind label %81

52:                                               ; preds = %41
  %53 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #15
          to label %54 unwind label %81

54:                                               ; preds = %52
  %55 = getelementptr inbounds nuw i8, ptr %53, i64 16
  store ptr %51, ptr %55, align 8, !tbaa !29
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16) %53, ptr noundef nonnull align 8 dereferenceable(24) %4) #14
  %56 = load i64, ptr %18, align 8, !tbaa !31
  %57 = add i64 %56, 1
  store i64 %57, ptr %18, align 8, !tbaa !31
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #14
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #14
  %58 = load double, ptr %1, align 8, !tbaa !6
  %59 = fsub double %58, %25
  store double %59, ptr %5, align 16
  %60 = load <2 x double>, ptr %8, align 8, !tbaa !13
  %61 = fadd <2 x double> %49, %60
  store <2 x double> %61, ptr %27, align 8
  %62 = invoke noundef ptr @_Z6createiRK3Vecd(i32 noundef %26, ptr noundef nonnull align 8 dereferenceable(24) %5, double noundef %29)
          to label %63 unwind label %81

63:                                               ; preds = %54
  %64 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #15
          to label %65 unwind label %81

65:                                               ; preds = %63
  %66 = getelementptr inbounds nuw i8, ptr %64, i64 16
  store ptr %62, ptr %66, align 8, !tbaa !29
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16) %64, ptr noundef nonnull align 8 dereferenceable(24) %4) #14
  %67 = load i64, ptr %18, align 8, !tbaa !31
  %68 = add i64 %67, 1
  store i64 %68, ptr %18, align 8, !tbaa !31
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #14
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #14
  %69 = load double, ptr %9, align 8, !tbaa !12
  %70 = fadd double %25, %69
  %71 = load <2 x double>, ptr %1, align 8, !tbaa !13
  %72 = fadd <2 x double> %49, %71
  store <2 x double> %72, ptr %5, align 16
  store double %70, ptr %28, align 16
  %73 = invoke noundef ptr @_Z6createiRK3Vecd(i32 noundef %26, ptr noundef nonnull align 8 dereferenceable(24) %5, double noundef %29)
          to label %74 unwind label %81

74:                                               ; preds = %65
  %75 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #15
          to label %76 unwind label %81

76:                                               ; preds = %74
  %77 = getelementptr inbounds nuw i8, ptr %75, i64 16
  store ptr %73, ptr %77, align 8, !tbaa !29
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16) %75, ptr noundef nonnull align 8 dereferenceable(24) %4) #14
  %78 = load i64, ptr %18, align 8, !tbaa !31
  %79 = add i64 %78, 1
  store i64 %79, ptr %18, align 8, !tbaa !31
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #14
  %80 = invoke noalias noundef nonnull dereferenceable(72) ptr @_Znwm(i64 noundef 72) #15
          to label %83 unwind label %159

81:                                               ; preds = %74, %65, %63, %54, %52, %41, %39, %20
  %82 = landingpad { ptr, i32 }
          cleanup
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #14
  br label %170

83:                                               ; preds = %76
  %84 = load double, ptr %1, align 8, !tbaa !13
  %85 = load double, ptr %8, align 8, !tbaa !13
  %86 = load double, ptr %9, align 8, !tbaa !13
  %87 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store ptr %6, ptr %87, align 8, !tbaa !21
  store ptr %6, ptr %6, align 8, !tbaa !25
  %88 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store i64 0, ptr %88, align 8, !tbaa !26
  %89 = load ptr, ptr %4, align 8, !tbaa !25
  %90 = icmp eq ptr %89, %4
  br i1 %90, label %112, label %91

91:                                               ; preds = %83, %94
  %92 = phi ptr [ %100, %94 ], [ %89, %83 ]
  %93 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #15
          to label %94 unwind label %102

94:                                               ; preds = %91
  %95 = getelementptr inbounds nuw i8, ptr %92, i64 16
  %96 = getelementptr inbounds nuw i8, ptr %93, i64 16
  %97 = load ptr, ptr %95, align 8, !tbaa !29
  store ptr %97, ptr %96, align 8, !tbaa !29
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16) %93, ptr noundef nonnull align 8 dereferenceable(24) %6) #14
  %98 = load i64, ptr %88, align 8, !tbaa !31
  %99 = add i64 %98, 1
  store i64 %99, ptr %88, align 8, !tbaa !31
  %100 = load ptr, ptr %92, align 8, !tbaa !25
  %101 = icmp eq ptr %100, %4
  br i1 %101, label %110, label %91, !llvm.loop !34

102:                                              ; preds = %91
  %103 = landingpad { ptr, i32 }
          cleanup
  %104 = load ptr, ptr %6, align 8, !tbaa !25
  %105 = icmp eq ptr %104, %6
  br i1 %105, label %168, label %106

106:                                              ; preds = %102, %106
  %107 = phi ptr [ %108, %106 ], [ %104, %102 ]
  %108 = load ptr, ptr %107, align 8, !tbaa !25
  call void @_ZdlPvm(ptr noundef nonnull %107, i64 noundef 24) #16
  %109 = icmp eq ptr %108, %6
  br i1 %109, label %168, label %106, !llvm.loop !36

110:                                              ; preds = %94
  %111 = load ptr, ptr %6, align 8, !tbaa !25
  br label %112

112:                                              ; preds = %110, %83
  %113 = phi ptr [ %111, %110 ], [ %6, %83 ]
  store ptr getelementptr inbounds nuw inrange(-16, 24) (i8, ptr @_ZTV5Group, i64 16), ptr %80, align 8, !tbaa !16
  %114 = getelementptr inbounds nuw i8, ptr %80, i64 8
  store ptr getelementptr inbounds nuw inrange(-16, 24) (i8, ptr @_ZTV6Sphere, i64 16), ptr %114, align 8, !tbaa !16
  %115 = getelementptr inbounds nuw i8, ptr %80, i64 16
  store double %84, ptr %115, align 8
  %116 = getelementptr inbounds nuw i8, ptr %80, i64 24
  store double %85, ptr %116, align 8
  %117 = getelementptr inbounds nuw i8, ptr %80, i64 32
  store double %86, ptr %117, align 8
  %118 = getelementptr inbounds nuw i8, ptr %80, i64 40
  store double %24, ptr %118, align 8
  %119 = getelementptr inbounds nuw i8, ptr %80, i64 48
  %120 = getelementptr inbounds nuw i8, ptr %80, i64 56
  store ptr %119, ptr %120, align 8, !tbaa !21
  store ptr %119, ptr %119, align 8, !tbaa !25
  %121 = getelementptr inbounds nuw i8, ptr %80, i64 64
  store i64 0, ptr %121, align 8, !tbaa !26
  %122 = icmp eq ptr %113, %6
  br i1 %122, label %144, label %123

123:                                              ; preds = %112, %126
  %124 = phi ptr [ %132, %126 ], [ %113, %112 ]
  %125 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #15
          to label %126 unwind label %134

126:                                              ; preds = %123
  %127 = getelementptr inbounds nuw i8, ptr %124, i64 16
  %128 = getelementptr inbounds nuw i8, ptr %125, i64 16
  %129 = load ptr, ptr %127, align 8, !tbaa !29
  store ptr %129, ptr %128, align 8, !tbaa !29
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16) %125, ptr noundef nonnull align 8 dereferenceable(24) %119) #14
  %130 = load i64, ptr %121, align 8, !tbaa !31
  %131 = add i64 %130, 1
  store i64 %131, ptr %121, align 8, !tbaa !31
  %132 = load ptr, ptr %124, align 8, !tbaa !25
  %133 = icmp eq ptr %132, %6
  br i1 %133, label %142, label %123, !llvm.loop !34

134:                                              ; preds = %123
  %135 = landingpad { ptr, i32 }
          cleanup
  %136 = load ptr, ptr %119, align 8, !tbaa !25
  %137 = icmp eq ptr %136, %119
  br i1 %137, label %161, label %138

138:                                              ; preds = %134, %138
  %139 = phi ptr [ %140, %138 ], [ %136, %134 ]
  %140 = load ptr, ptr %139, align 8, !tbaa !25
  call void @_ZdlPvm(ptr noundef nonnull %139, i64 noundef 24) #16
  %141 = icmp eq ptr %140, %119
  br i1 %141, label %161, label %138, !llvm.loop !36

142:                                              ; preds = %126
  %143 = load ptr, ptr %6, align 8, !tbaa !25
  br label %144

144:                                              ; preds = %142, %112
  %145 = phi ptr [ %143, %142 ], [ %113, %112 ]
  %146 = icmp eq ptr %145, %6
  br i1 %146, label %151, label %147

147:                                              ; preds = %144, %147
  %148 = phi ptr [ %149, %147 ], [ %145, %144 ]
  %149 = load ptr, ptr %148, align 8, !tbaa !25
  call void @_ZdlPvm(ptr noundef nonnull %148, i64 noundef 24) #16
  %150 = icmp eq ptr %149, %6
  br i1 %150, label %151, label %147, !llvm.loop !36

151:                                              ; preds = %147, %144
  %152 = load ptr, ptr %4, align 8, !tbaa !25
  %153 = icmp eq ptr %152, %4
  br i1 %153, label %158, label %154

154:                                              ; preds = %151, %154
  %155 = phi ptr [ %156, %154 ], [ %152, %151 ]
  %156 = load ptr, ptr %155, align 8, !tbaa !25
  call void @_ZdlPvm(ptr noundef nonnull %155, i64 noundef 24) #16
  %157 = icmp eq ptr %156, %4
  br i1 %157, label %158, label %154, !llvm.loop !36

158:                                              ; preds = %154, %151
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #14
  br label %179

159:                                              ; preds = %76
  %160 = landingpad { ptr, i32 }
          cleanup
  br label %170

161:                                              ; preds = %138, %134
  %162 = load ptr, ptr %6, align 8, !tbaa !25
  %163 = icmp eq ptr %162, %6
  br i1 %163, label %168, label %164

164:                                              ; preds = %161, %164
  %165 = phi ptr [ %166, %164 ], [ %162, %161 ]
  %166 = load ptr, ptr %165, align 8, !tbaa !25
  call void @_ZdlPvm(ptr noundef nonnull %165, i64 noundef 24) #16
  %167 = icmp eq ptr %166, %6
  br i1 %167, label %168, label %164, !llvm.loop !36

168:                                              ; preds = %106, %164, %161, %102
  %169 = phi { ptr, i32 } [ %103, %102 ], [ %135, %161 ], [ %135, %164 ], [ %103, %106 ]
  call void @_ZdlPvm(ptr noundef nonnull %80, i64 noundef 72) #16
  br label %170

170:                                              ; preds = %81, %159, %168, %37
  %171 = phi { ptr, i32 } [ %38, %37 ], [ %82, %81 ], [ %169, %168 ], [ %160, %159 ]
  %172 = load ptr, ptr %4, align 8, !tbaa !25
  %173 = icmp eq ptr %172, %4
  br i1 %173, label %178, label %174

174:                                              ; preds = %170, %174
  %175 = phi ptr [ %176, %174 ], [ %172, %170 ]
  %176 = load ptr, ptr %175, align 8, !tbaa !25
  call void @_ZdlPvm(ptr noundef nonnull %175, i64 noundef 24) #16
  %177 = icmp eq ptr %176, %4
  br i1 %177, label %178, label %174, !llvm.loop !36

178:                                              ; preds = %174, %170
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #14
  resume { ptr, i32 } %171

179:                                              ; preds = %3, %158
  %180 = phi ptr [ %80, %158 ], [ %7, %3 ]
  ret ptr %180
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #5

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #6

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #7

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN5SceneD2Ev(ptr noundef nonnull align 8 dereferenceable(8) %0) unnamed_addr #8 comdat {
  ret void
}

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #9 {
  %3 = alloca %"struct.std::pair", align 8
  %4 = alloca %"struct.std::pair", align 8
  %5 = alloca %struct.Ray, align 16
  %6 = alloca i8, align 4
  %7 = alloca %struct.Vec, align 16
  %8 = alloca %struct.Ray, align 16
  %9 = icmp eq i32 %0, 2
  br i1 %9, label %10, label %15

10:                                               ; preds = %2
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %12 = load ptr, ptr %11, align 8, !tbaa !37
  %13 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %12, ptr noundef null, i32 noundef 10) #14
  %14 = trunc i64 %13 to i32
  br label %15

15:                                               ; preds = %10, %2
  %16 = phi i32 [ %14, %10 ], [ 6, %2 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #14
  store <2 x double> <double 0.000000e+00, double -1.000000e+00>, ptr %7, align 16, !tbaa !13
  %17 = getelementptr inbounds nuw i8, ptr %7, i64 16
  store double 0.000000e+00, ptr %17, align 16, !tbaa !12
  %18 = call noundef ptr @_Z6createiRK3Vecd(i32 noundef %16, ptr noundef nonnull align 8 dereferenceable(24) %7, double noundef 1.000000e+00)
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #14
  %19 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str, i64 noundef 3)
  %20 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i32 noundef 512)
  %21 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %20, ptr noundef nonnull @.str.1, i64 noundef 1)
  %22 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) %20, i32 noundef 512)
  %23 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %22, ptr noundef nonnull @.str.2, i64 noundef 5)
  %24 = getelementptr inbounds nuw i8, ptr %8, i64 16
  %25 = getelementptr inbounds nuw i8, ptr %8, i64 24
  %26 = getelementptr inbounds nuw i8, ptr %8, i64 32
  %27 = getelementptr inbounds nuw i8, ptr %8, i64 40
  %28 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %29 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %30 = getelementptr inbounds nuw i8, ptr %5, i64 24
  %31 = getelementptr inbounds nuw i8, ptr %5, i64 40
  %32 = getelementptr inbounds nuw i8, ptr %3, i64 8
  br label %33

33:                                               ; preds = %15, %55
  %34 = phi i32 [ 511, %15 ], [ %56, %55 ]
  %35 = uitofp nneg i32 %34 to double
  %36 = fadd double %35, -2.560000e+02
  %37 = fmul double %36, %36
  %38 = insertelement <2 x double> <double poison, double 5.120000e+02>, double %36, i64 0
  %39 = fadd double %35, 2.500000e-01
  %40 = fadd double %39, -2.560000e+02
  %41 = fmul double %40, %40
  %42 = insertelement <2 x double> <double poison, double 5.120000e+02>, double %40, i64 0
  %43 = fadd double %35, 5.000000e-01
  %44 = fadd double %43, -2.560000e+02
  %45 = fmul double %44, %44
  %46 = insertelement <2 x double> <double poison, double 5.120000e+02>, double %44, i64 0
  %47 = fadd double %35, 7.500000e-01
  %48 = fadd double %47, -2.560000e+02
  %49 = fmul double %48, %48
  %50 = insertelement <2 x double> <double poison, double 5.120000e+02>, double %48, i64 0
  br label %51

51:                                               ; preds = %33, %102
  %52 = phi i32 [ 0, %33 ], [ %103, %102 ]
  %53 = uitofp nneg i32 %52 to double
  %54 = load double, ptr @infinity, align 8, !tbaa !13
  br label %58

55:                                               ; preds = %102
  %56 = add nsw i32 %34, -1
  %57 = icmp eq i32 %34, 0
  br i1 %57, label %316, label %33, !llvm.loop !39

58:                                               ; preds = %51, %310
  %59 = phi double [ %54, %51 ], [ %311, %310 ]
  %60 = phi i32 [ 0, %51 ], [ %314, %310 ]
  %61 = phi double [ 0.000000e+00, %51 ], [ %313, %310 ]
  %62 = uitofp nneg i32 %60 to double
  %63 = fmul double %62, 2.500000e-01
  %64 = fadd double %63, %53
  %65 = fadd double %64, -2.560000e+02
  %66 = call double @llvm.fmuladd.f64(double %65, double %65, double %37)
  %67 = fadd double %66, 2.621440e+05
  %68 = call double @llvm.sqrt.f64(double %67)
  %69 = fdiv double 1.000000e+00, %68
  %70 = fmul double %65, %69
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #14
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(16) %8, i8 0, i64 16, i1 false)
  store double -4.000000e+00, ptr %24, align 16, !tbaa !13
  store double %70, ptr %25, align 8, !tbaa !13
  %71 = insertelement <2 x double> poison, double %69, i64 0
  %72 = shufflevector <2 x double> %71, <2 x double> poison, <2 x i32> zeroinitializer
  %73 = fmul <2 x double> %72, %38
  store <2 x double> %73, ptr %26, align 16, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #14
  store double %59, ptr %4, align 8, !tbaa !14
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %28, i8 0, i64 24, i1 false)
  %74 = load ptr, ptr %18, align 8, !tbaa !16
  %75 = getelementptr inbounds nuw i8, ptr %74, i64 16
  %76 = load ptr, ptr %75, align 8
  %77 = call %"struct.std::pair" %76(ptr noundef nonnull align 8 dereferenceable(8) %18, ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(48) %8)
  %78 = extractvalue %"struct.std::pair" %77, 1
  %79 = extractvalue %struct.Vec %78, 0
  %80 = extractvalue %struct.Vec %78, 1
  %81 = extractvalue %struct.Vec %78, 2
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #14
  %82 = extractvalue %"struct.std::pair" %77, 0
  %83 = load double, ptr @infinity, align 8, !tbaa !13
  %84 = fcmp oeq double %82, %83
  br i1 %84, label %139, label %105

85:                                               ; preds = %310
  %86 = fmul double %313, 2.550000e+02
  %87 = fmul double %86, 6.250000e-02
  %88 = fadd double %87, 5.000000e-01
  %89 = fptosi double %88 to i32
  %90 = trunc i32 %89 to i8
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  store i8 %90, ptr %6, align 4, !tbaa !40
  %91 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !16
  %92 = getelementptr i8, ptr %91, i64 -24
  %93 = load i64, ptr %92, align 8
  %94 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %93
  %95 = getelementptr inbounds nuw i8, ptr %94, i64 16
  %96 = load i64, ptr %95, align 8, !tbaa !41
  %97 = icmp eq i64 %96, 0
  br i1 %97, label %100, label %98

98:                                               ; preds = %85
  %99 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %6, i64 noundef 1)
  br label %102

100:                                              ; preds = %85
  %101 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %90)
  br label %102

102:                                              ; preds = %98, %100
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  %103 = add nuw nsw i32 %52, 1
  %104 = icmp eq i32 %103, 512
  br i1 %104, label %55, label %51, !llvm.loop !51

105:                                              ; preds = %58
  %106 = fmul double %80, 0xBFE9A8365810363F
  %107 = call double @llvm.fmuladd.f64(double %79, double 0xBFD11ACEE560242A, double %106)
  %108 = call noundef double @llvm.fmuladd.f64(double %81, double 0x3FE11ACEE560242A, double %107)
  %109 = fcmp ult double %108, 0.000000e+00
  br i1 %109, label %110, label %139

110:                                              ; preds = %105
  %111 = load double, ptr %27, align 8, !tbaa !12
  %112 = fmul double %82, %111
  %113 = load double, ptr %24, align 16, !tbaa !12
  %114 = fadd double %112, %113
  %115 = load double, ptr @delta, align 8, !tbaa !13
  %116 = fmul double %81, %115
  %117 = fadd double %114, %116
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #14
  %118 = load <2 x double>, ptr %25, align 8, !tbaa !13
  %119 = insertelement <2 x double> poison, double %82, i64 0
  %120 = shufflevector <2 x double> %119, <2 x double> poison, <2 x i32> zeroinitializer
  %121 = fmul <2 x double> %120, %118
  %122 = load <2 x double>, ptr %8, align 16, !tbaa !13
  %123 = fadd <2 x double> %121, %122
  %124 = insertelement <2 x double> poison, double %79, i64 0
  %125 = insertelement <2 x double> %124, double %80, i64 1
  %126 = insertelement <2 x double> poison, double %115, i64 0
  %127 = shufflevector <2 x double> %126, <2 x double> poison, <2 x i32> zeroinitializer
  %128 = fmul <2 x double> %125, %127
  %129 = fadd <2 x double> %123, %128
  store <2 x double> %129, ptr %5, align 16, !tbaa !13
  store double %117, ptr %29, align 16, !tbaa !13
  store <2 x double> <double 0x3FD11ACEE560242A, double 0x3FE9A8365810363F>, ptr %30, align 8, !tbaa !13
  store double 0xBFE11ACEE560242A, ptr %31, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #14
  store double %83, ptr %3, align 8, !tbaa !14
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %32, i8 0, i64 24, i1 false)
  %130 = load ptr, ptr %18, align 8, !tbaa !16
  %131 = getelementptr inbounds nuw i8, ptr %130, i64 16
  %132 = load ptr, ptr %131, align 8
  %133 = call %"struct.std::pair" %132(ptr noundef nonnull align 8 dereferenceable(8) %18, ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(48) %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #14
  %134 = extractvalue %"struct.std::pair" %133, 0
  %135 = load double, ptr @infinity, align 8, !tbaa !13
  %136 = fcmp olt double %134, %135
  %137 = fneg double %108
  %138 = select i1 %136, double 0.000000e+00, double %137
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #14
  br label %139

139:                                              ; preds = %58, %105, %110
  %140 = phi double [ %83, %58 ], [ %135, %110 ], [ %83, %105 ]
  %141 = phi double [ 0.000000e+00, %58 ], [ %138, %110 ], [ 0.000000e+00, %105 ]
  %142 = fadd double %61, %141
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #14
  %143 = call double @llvm.fmuladd.f64(double %65, double %65, double %41)
  %144 = fadd double %143, 2.621440e+05
  %145 = call double @llvm.sqrt.f64(double %144)
  %146 = fdiv double 1.000000e+00, %145
  %147 = fmul double %65, %146
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #14
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(16) %8, i8 0, i64 16, i1 false)
  store double -4.000000e+00, ptr %24, align 16, !tbaa !13
  store double %147, ptr %25, align 8, !tbaa !13
  %148 = insertelement <2 x double> poison, double %146, i64 0
  %149 = shufflevector <2 x double> %148, <2 x double> poison, <2 x i32> zeroinitializer
  %150 = fmul <2 x double> %149, %42
  store <2 x double> %150, ptr %26, align 16, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #14
  store double %140, ptr %4, align 8, !tbaa !14
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %28, i8 0, i64 24, i1 false)
  %151 = load ptr, ptr %18, align 8, !tbaa !16
  %152 = getelementptr inbounds nuw i8, ptr %151, i64 16
  %153 = load ptr, ptr %152, align 8
  %154 = call %"struct.std::pair" %153(ptr noundef nonnull align 8 dereferenceable(8) %18, ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(48) %8)
  %155 = extractvalue %"struct.std::pair" %154, 1
  %156 = extractvalue %struct.Vec %155, 0
  %157 = extractvalue %struct.Vec %155, 1
  %158 = extractvalue %struct.Vec %155, 2
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #14
  %159 = extractvalue %"struct.std::pair" %154, 0
  %160 = load double, ptr @infinity, align 8, !tbaa !13
  %161 = fcmp oeq double %159, %160
  br i1 %161, label %196, label %162

162:                                              ; preds = %139
  %163 = fmul double %157, 0xBFE9A8365810363F
  %164 = call double @llvm.fmuladd.f64(double %156, double 0xBFD11ACEE560242A, double %163)
  %165 = call noundef double @llvm.fmuladd.f64(double %158, double 0x3FE11ACEE560242A, double %164)
  %166 = fcmp ult double %165, 0.000000e+00
  br i1 %166, label %167, label %196

167:                                              ; preds = %162
  %168 = load double, ptr %27, align 8, !tbaa !12
  %169 = fmul double %159, %168
  %170 = load double, ptr %24, align 16, !tbaa !12
  %171 = fadd double %169, %170
  %172 = load double, ptr @delta, align 8, !tbaa !13
  %173 = fmul double %158, %172
  %174 = fadd double %171, %173
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #14
  %175 = load <2 x double>, ptr %25, align 8, !tbaa !13
  %176 = insertelement <2 x double> poison, double %159, i64 0
  %177 = shufflevector <2 x double> %176, <2 x double> poison, <2 x i32> zeroinitializer
  %178 = fmul <2 x double> %177, %175
  %179 = load <2 x double>, ptr %8, align 16, !tbaa !13
  %180 = fadd <2 x double> %178, %179
  %181 = insertelement <2 x double> poison, double %156, i64 0
  %182 = insertelement <2 x double> %181, double %157, i64 1
  %183 = insertelement <2 x double> poison, double %172, i64 0
  %184 = shufflevector <2 x double> %183, <2 x double> poison, <2 x i32> zeroinitializer
  %185 = fmul <2 x double> %182, %184
  %186 = fadd <2 x double> %180, %185
  store <2 x double> %186, ptr %5, align 16, !tbaa !13
  store double %174, ptr %29, align 16, !tbaa !13
  store <2 x double> <double 0x3FD11ACEE560242A, double 0x3FE9A8365810363F>, ptr %30, align 8, !tbaa !13
  store double 0xBFE11ACEE560242A, ptr %31, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #14
  store double %160, ptr %3, align 8, !tbaa !14
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %32, i8 0, i64 24, i1 false)
  %187 = load ptr, ptr %18, align 8, !tbaa !16
  %188 = getelementptr inbounds nuw i8, ptr %187, i64 16
  %189 = load ptr, ptr %188, align 8
  %190 = call %"struct.std::pair" %189(ptr noundef nonnull align 8 dereferenceable(8) %18, ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(48) %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #14
  %191 = extractvalue %"struct.std::pair" %190, 0
  %192 = load double, ptr @infinity, align 8, !tbaa !13
  %193 = fcmp olt double %191, %192
  %194 = fneg double %165
  %195 = select i1 %193, double 0.000000e+00, double %194
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #14
  br label %196

196:                                              ; preds = %167, %162, %139
  %197 = phi double [ %160, %139 ], [ %192, %167 ], [ %160, %162 ]
  %198 = phi double [ 0.000000e+00, %139 ], [ %195, %167 ], [ 0.000000e+00, %162 ]
  %199 = fadd double %142, %198
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #14
  %200 = call double @llvm.fmuladd.f64(double %65, double %65, double %45)
  %201 = fadd double %200, 2.621440e+05
  %202 = call double @llvm.sqrt.f64(double %201)
  %203 = fdiv double 1.000000e+00, %202
  %204 = fmul double %65, %203
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #14
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(16) %8, i8 0, i64 16, i1 false)
  store double -4.000000e+00, ptr %24, align 16, !tbaa !13
  store double %204, ptr %25, align 8, !tbaa !13
  %205 = insertelement <2 x double> poison, double %203, i64 0
  %206 = shufflevector <2 x double> %205, <2 x double> poison, <2 x i32> zeroinitializer
  %207 = fmul <2 x double> %206, %46
  store <2 x double> %207, ptr %26, align 16, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #14
  store double %197, ptr %4, align 8, !tbaa !14
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %28, i8 0, i64 24, i1 false)
  %208 = load ptr, ptr %18, align 8, !tbaa !16
  %209 = getelementptr inbounds nuw i8, ptr %208, i64 16
  %210 = load ptr, ptr %209, align 8
  %211 = call %"struct.std::pair" %210(ptr noundef nonnull align 8 dereferenceable(8) %18, ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(48) %8)
  %212 = extractvalue %"struct.std::pair" %211, 1
  %213 = extractvalue %struct.Vec %212, 0
  %214 = extractvalue %struct.Vec %212, 1
  %215 = extractvalue %struct.Vec %212, 2
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #14
  %216 = extractvalue %"struct.std::pair" %211, 0
  %217 = load double, ptr @infinity, align 8, !tbaa !13
  %218 = fcmp oeq double %216, %217
  br i1 %218, label %253, label %219

219:                                              ; preds = %196
  %220 = fmul double %214, 0xBFE9A8365810363F
  %221 = call double @llvm.fmuladd.f64(double %213, double 0xBFD11ACEE560242A, double %220)
  %222 = call noundef double @llvm.fmuladd.f64(double %215, double 0x3FE11ACEE560242A, double %221)
  %223 = fcmp ult double %222, 0.000000e+00
  br i1 %223, label %224, label %253

224:                                              ; preds = %219
  %225 = load double, ptr %27, align 8, !tbaa !12
  %226 = fmul double %216, %225
  %227 = load double, ptr %24, align 16, !tbaa !12
  %228 = fadd double %226, %227
  %229 = load double, ptr @delta, align 8, !tbaa !13
  %230 = fmul double %215, %229
  %231 = fadd double %228, %230
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #14
  %232 = load <2 x double>, ptr %25, align 8, !tbaa !13
  %233 = insertelement <2 x double> poison, double %216, i64 0
  %234 = shufflevector <2 x double> %233, <2 x double> poison, <2 x i32> zeroinitializer
  %235 = fmul <2 x double> %234, %232
  %236 = load <2 x double>, ptr %8, align 16, !tbaa !13
  %237 = fadd <2 x double> %235, %236
  %238 = insertelement <2 x double> poison, double %213, i64 0
  %239 = insertelement <2 x double> %238, double %214, i64 1
  %240 = insertelement <2 x double> poison, double %229, i64 0
  %241 = shufflevector <2 x double> %240, <2 x double> poison, <2 x i32> zeroinitializer
  %242 = fmul <2 x double> %239, %241
  %243 = fadd <2 x double> %237, %242
  store <2 x double> %243, ptr %5, align 16, !tbaa !13
  store double %231, ptr %29, align 16, !tbaa !13
  store <2 x double> <double 0x3FD11ACEE560242A, double 0x3FE9A8365810363F>, ptr %30, align 8, !tbaa !13
  store double 0xBFE11ACEE560242A, ptr %31, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #14
  store double %217, ptr %3, align 8, !tbaa !14
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %32, i8 0, i64 24, i1 false)
  %244 = load ptr, ptr %18, align 8, !tbaa !16
  %245 = getelementptr inbounds nuw i8, ptr %244, i64 16
  %246 = load ptr, ptr %245, align 8
  %247 = call %"struct.std::pair" %246(ptr noundef nonnull align 8 dereferenceable(8) %18, ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(48) %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #14
  %248 = extractvalue %"struct.std::pair" %247, 0
  %249 = load double, ptr @infinity, align 8, !tbaa !13
  %250 = fcmp olt double %248, %249
  %251 = fneg double %222
  %252 = select i1 %250, double 0.000000e+00, double %251
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #14
  br label %253

253:                                              ; preds = %224, %219, %196
  %254 = phi double [ %217, %196 ], [ %249, %224 ], [ %217, %219 ]
  %255 = phi double [ 0.000000e+00, %196 ], [ %252, %224 ], [ 0.000000e+00, %219 ]
  %256 = fadd double %199, %255
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #14
  %257 = call double @llvm.fmuladd.f64(double %65, double %65, double %49)
  %258 = fadd double %257, 2.621440e+05
  %259 = call double @llvm.sqrt.f64(double %258)
  %260 = fdiv double 1.000000e+00, %259
  %261 = fmul double %65, %260
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #14
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(16) %8, i8 0, i64 16, i1 false)
  store double -4.000000e+00, ptr %24, align 16, !tbaa !13
  store double %261, ptr %25, align 8, !tbaa !13
  %262 = insertelement <2 x double> poison, double %260, i64 0
  %263 = shufflevector <2 x double> %262, <2 x double> poison, <2 x i32> zeroinitializer
  %264 = fmul <2 x double> %263, %50
  store <2 x double> %264, ptr %26, align 16, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #14
  store double %254, ptr %4, align 8, !tbaa !14
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %28, i8 0, i64 24, i1 false)
  %265 = load ptr, ptr %18, align 8, !tbaa !16
  %266 = getelementptr inbounds nuw i8, ptr %265, i64 16
  %267 = load ptr, ptr %266, align 8
  %268 = call %"struct.std::pair" %267(ptr noundef nonnull align 8 dereferenceable(8) %18, ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(48) %8)
  %269 = extractvalue %"struct.std::pair" %268, 1
  %270 = extractvalue %struct.Vec %269, 0
  %271 = extractvalue %struct.Vec %269, 1
  %272 = extractvalue %struct.Vec %269, 2
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #14
  %273 = extractvalue %"struct.std::pair" %268, 0
  %274 = load double, ptr @infinity, align 8, !tbaa !13
  %275 = fcmp oeq double %273, %274
  br i1 %275, label %310, label %276

276:                                              ; preds = %253
  %277 = fmul double %271, 0xBFE9A8365810363F
  %278 = call double @llvm.fmuladd.f64(double %270, double 0xBFD11ACEE560242A, double %277)
  %279 = call noundef double @llvm.fmuladd.f64(double %272, double 0x3FE11ACEE560242A, double %278)
  %280 = fcmp ult double %279, 0.000000e+00
  br i1 %280, label %281, label %310

281:                                              ; preds = %276
  %282 = load double, ptr %27, align 8, !tbaa !12
  %283 = fmul double %273, %282
  %284 = load double, ptr %24, align 16, !tbaa !12
  %285 = fadd double %283, %284
  %286 = load double, ptr @delta, align 8, !tbaa !13
  %287 = fmul double %272, %286
  %288 = fadd double %285, %287
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #14
  %289 = load <2 x double>, ptr %25, align 8, !tbaa !13
  %290 = insertelement <2 x double> poison, double %273, i64 0
  %291 = shufflevector <2 x double> %290, <2 x double> poison, <2 x i32> zeroinitializer
  %292 = fmul <2 x double> %291, %289
  %293 = load <2 x double>, ptr %8, align 16, !tbaa !13
  %294 = fadd <2 x double> %292, %293
  %295 = insertelement <2 x double> poison, double %270, i64 0
  %296 = insertelement <2 x double> %295, double %271, i64 1
  %297 = insertelement <2 x double> poison, double %286, i64 0
  %298 = shufflevector <2 x double> %297, <2 x double> poison, <2 x i32> zeroinitializer
  %299 = fmul <2 x double> %296, %298
  %300 = fadd <2 x double> %294, %299
  store <2 x double> %300, ptr %5, align 16, !tbaa !13
  store double %288, ptr %29, align 16, !tbaa !13
  store <2 x double> <double 0x3FD11ACEE560242A, double 0x3FE9A8365810363F>, ptr %30, align 8, !tbaa !13
  store double 0xBFE11ACEE560242A, ptr %31, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #14
  store double %274, ptr %3, align 8, !tbaa !14
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %32, i8 0, i64 24, i1 false)
  %301 = load ptr, ptr %18, align 8, !tbaa !16
  %302 = getelementptr inbounds nuw i8, ptr %301, i64 16
  %303 = load ptr, ptr %302, align 8
  %304 = call %"struct.std::pair" %303(ptr noundef nonnull align 8 dereferenceable(8) %18, ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(48) %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #14
  %305 = extractvalue %"struct.std::pair" %304, 0
  %306 = load double, ptr @infinity, align 8, !tbaa !13
  %307 = fcmp olt double %305, %306
  %308 = fneg double %279
  %309 = select i1 %307, double 0.000000e+00, double %308
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #14
  br label %310

310:                                              ; preds = %281, %276, %253
  %311 = phi double [ %274, %253 ], [ %306, %281 ], [ %274, %276 ]
  %312 = phi double [ 0.000000e+00, %253 ], [ %309, %281 ], [ 0.000000e+00, %276 ]
  %313 = fadd double %256, %312
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #14
  %314 = add nuw nsw i32 %60, 1
  %315 = icmp eq i32 %314, 4
  br i1 %315, label %85, label %58, !llvm.loop !52

316:                                              ; preds = %55
  %317 = load ptr, ptr %18, align 8, !tbaa !16
  %318 = getelementptr inbounds nuw i8, ptr %317, i64 8
  %319 = load ptr, ptr %318, align 8
  call void %319(ptr noundef nonnull align 8 dereferenceable(8) %18) #14
  ret i32 0
}

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8), i32 noundef) local_unnamed_addr #10

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN6SphereD0Ev(ptr noundef nonnull align 8 dereferenceable(40) %0) unnamed_addr #8 comdat {
  tail call void @_ZdlPvm(ptr noundef nonnull %0, i64 noundef 40) #16
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local %"struct.std::pair" @_ZNK6Sphere9intersectERKSt4pairId3VecERK3Ray(ptr noundef nonnull align 8 dereferenceable(40) %0, ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(48) %2) unnamed_addr #3 comdat {
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %5 = load double, ptr %4, align 8, !tbaa !6
  %6 = load double, ptr %2, align 8, !tbaa !6
  %7 = fsub double %5, %6
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %9 = load double, ptr %8, align 8, !tbaa !11
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %11 = load double, ptr %10, align 8, !tbaa !11
  %12 = fsub double %9, %11
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %14 = load double, ptr %13, align 8, !tbaa !12
  %15 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %16 = load double, ptr %15, align 8, !tbaa !12
  %17 = fsub double %14, %16
  %18 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %19 = load double, ptr %18, align 8, !tbaa !6
  %20 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %21 = load double, ptr %20, align 8, !tbaa !11
  %22 = fmul double %12, %21
  %23 = tail call double @llvm.fmuladd.f64(double %7, double %19, double %22)
  %24 = getelementptr inbounds nuw i8, ptr %2, i64 40
  %25 = load double, ptr %24, align 8, !tbaa !12
  %26 = tail call noundef double @llvm.fmuladd.f64(double %17, double %25, double %23)
  %27 = fmul double %12, %12
  %28 = tail call double @llvm.fmuladd.f64(double %7, double %7, double %27)
  %29 = tail call noundef double @llvm.fmuladd.f64(double %17, double %17, double %28)
  %30 = fneg double %29
  %31 = tail call double @llvm.fmuladd.f64(double %26, double %26, double %30)
  %32 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %33 = load double, ptr %32, align 8, !tbaa !18
  %34 = tail call double @llvm.fmuladd.f64(double %33, double %33, double %31)
  %35 = fcmp olt double %34, 0.000000e+00
  br i1 %35, label %36, label %38

36:                                               ; preds = %3
  %37 = load double, ptr @infinity, align 8, !tbaa !13
  br label %47

38:                                               ; preds = %3
  %39 = tail call double @sqrt(double noundef %34) #14, !tbaa !53
  %40 = fadd double %26, %39
  %41 = fcmp olt double %40, 0.000000e+00
  %42 = load double, ptr @infinity, align 8
  %43 = fsub double %26, %39
  %44 = fcmp ogt double %43, 0.000000e+00
  %45 = select i1 %44, double %43, double %40
  %46 = select i1 %41, double %42, double %45
  br label %47

47:                                               ; preds = %36, %38
  %48 = phi double [ %37, %36 ], [ %46, %38 ]
  %49 = load double, ptr %1, align 8, !tbaa !14
  %50 = fcmp ult double %48, %49
  br i1 %50, label %58, label %51

51:                                               ; preds = %47
  %52 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %53 = load double, ptr %52, align 8
  %54 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %55 = load double, ptr %54, align 8
  %56 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %57 = load double, ptr %56, align 8
  br label %76

58:                                               ; preds = %47
  %59 = fmul double %19, %48
  %60 = fmul double %48, %21
  %61 = fmul double %48, %25
  %62 = fadd double %6, %59
  %63 = fadd double %11, %60
  %64 = fadd double %16, %61
  %65 = fsub double %62, %5
  %66 = fsub double %63, %9
  %67 = fsub double %64, %14
  %68 = fmul double %66, %66
  %69 = tail call double @llvm.fmuladd.f64(double %65, double %65, double %68)
  %70 = tail call noundef double @llvm.fmuladd.f64(double %67, double %67, double %69)
  %71 = tail call double @llvm.sqrt.f64(double %70)
  %72 = fdiv double 1.000000e+00, %71
  %73 = fmul double %65, %72
  %74 = fmul double %66, %72
  %75 = fmul double %67, %72
  br label %76

76:                                               ; preds = %58, %51
  %77 = phi double [ %48, %58 ], [ %49, %51 ]
  %78 = phi double [ %73, %58 ], [ %53, %51 ]
  %79 = phi double [ %74, %58 ], [ %55, %51 ]
  %80 = phi double [ %75, %58 ], [ %57, %51 ]
  %81 = insertvalue %"struct.std::pair" poison, double %77, 0
  %82 = insertvalue %"struct.std::pair" %81, double %78, 1, 0
  %83 = insertvalue %"struct.std::pair" %82, double %79, 1, 1
  %84 = insertvalue %"struct.std::pair" %83, double %80, 1, 2
  ret %"struct.std::pair" %84
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN5GroupD2Ev(ptr noundef nonnull align 8 dereferenceable(72) %0) unnamed_addr #8 comdat personality ptr @__gxx_personality_v0 {
  store ptr getelementptr inbounds nuw inrange(-16, 24) (i8, ptr @_ZTV5Group, i64 16), ptr %0, align 8, !tbaa !16
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %3 = load ptr, ptr %2, align 8, !tbaa !25
  %4 = icmp eq ptr %3, %2
  br i1 %4, label %7, label %15

5:                                                ; preds = %24
  %6 = load ptr, ptr %2, align 8, !tbaa !25
  br label %7

7:                                                ; preds = %5, %1
  %8 = phi ptr [ %6, %5 ], [ %3, %1 ]
  %9 = icmp eq ptr %8, %2
  br i1 %9, label %14, label %10

10:                                               ; preds = %7, %10
  %11 = phi ptr [ %12, %10 ], [ %8, %7 ]
  %12 = load ptr, ptr %11, align 8, !tbaa !25
  tail call void @_ZdlPvm(ptr noundef nonnull %11, i64 noundef 24) #16
  %13 = icmp eq ptr %12, %2
  br i1 %13, label %14, label %10, !llvm.loop !36

14:                                               ; preds = %10, %7
  ret void

15:                                               ; preds = %1, %24
  %16 = phi ptr [ %25, %24 ], [ %3, %1 ]
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 16
  %18 = load ptr, ptr %17, align 8, !tbaa !29
  %19 = icmp eq ptr %18, null
  br i1 %19, label %24, label %20

20:                                               ; preds = %15
  %21 = load ptr, ptr %18, align 8, !tbaa !16
  %22 = getelementptr inbounds nuw i8, ptr %21, i64 8
  %23 = load ptr, ptr %22, align 8
  tail call void %23(ptr noundef nonnull align 8 dereferenceable(8) %18) #14
  br label %24

24:                                               ; preds = %15, %20
  %25 = load ptr, ptr %16, align 8, !tbaa !25
  %26 = icmp eq ptr %25, %2
  br i1 %26, label %5, label %15, !llvm.loop !54
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN5GroupD0Ev(ptr noundef nonnull align 8 dereferenceable(72) %0) unnamed_addr #8 comdat personality ptr @__gxx_personality_v0 {
  store ptr getelementptr inbounds nuw inrange(-16, 24) (i8, ptr @_ZTV5Group, i64 16), ptr %0, align 8, !tbaa !16
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %3 = load ptr, ptr %2, align 8, !tbaa !25
  %4 = icmp eq ptr %3, %2
  br i1 %4, label %7, label %14

5:                                                ; preds = %23
  %6 = load ptr, ptr %2, align 8, !tbaa !25
  br label %7

7:                                                ; preds = %5, %1
  %8 = phi ptr [ %6, %5 ], [ %3, %1 ]
  %9 = icmp eq ptr %8, %2
  br i1 %9, label %26, label %10

10:                                               ; preds = %7, %10
  %11 = phi ptr [ %12, %10 ], [ %8, %7 ]
  %12 = load ptr, ptr %11, align 8, !tbaa !25
  tail call void @_ZdlPvm(ptr noundef nonnull %11, i64 noundef 24) #16
  %13 = icmp eq ptr %12, %2
  br i1 %13, label %26, label %10, !llvm.loop !36

14:                                               ; preds = %1, %23
  %15 = phi ptr [ %24, %23 ], [ %3, %1 ]
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 16
  %17 = load ptr, ptr %16, align 8, !tbaa !29
  %18 = icmp eq ptr %17, null
  br i1 %18, label %23, label %19

19:                                               ; preds = %14
  %20 = load ptr, ptr %17, align 8, !tbaa !16
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 8
  %22 = load ptr, ptr %21, align 8
  tail call void %22(ptr noundef nonnull align 8 dereferenceable(8) %17) #14
  br label %23

23:                                               ; preds = %19, %14
  %24 = load ptr, ptr %15, align 8, !tbaa !25
  %25 = icmp eq ptr %24, %2
  br i1 %25, label %5, label %14, !llvm.loop !54

26:                                               ; preds = %10, %7
  tail call void @_ZdlPvm(ptr noundef nonnull %0, i64 noundef 72) #16
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local %"struct.std::pair" @_ZNK5Group9intersectERKSt4pairId3VecERK3Ray(ptr noundef nonnull align 8 dereferenceable(72) %0, ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(48) %2) unnamed_addr #3 comdat personality ptr @__gxx_personality_v0 {
  %4 = alloca %"struct.std::pair", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #14
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) %1, i64 32, i1 false)
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %6 = load double, ptr %5, align 8, !tbaa !6
  %7 = load double, ptr %2, align 8, !tbaa !6
  %8 = fsub double %6, %7
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %10 = load double, ptr %9, align 8, !tbaa !11
  %11 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %12 = load double, ptr %11, align 8, !tbaa !11
  %13 = fsub double %10, %12
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %15 = load double, ptr %14, align 8, !tbaa !12
  %16 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %17 = load double, ptr %16, align 8, !tbaa !12
  %18 = fsub double %15, %17
  %19 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %20 = load double, ptr %19, align 8, !tbaa !6
  %21 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %22 = load double, ptr %21, align 8, !tbaa !11
  %23 = fmul double %13, %22
  %24 = tail call double @llvm.fmuladd.f64(double %8, double %20, double %23)
  %25 = getelementptr inbounds nuw i8, ptr %2, i64 40
  %26 = load double, ptr %25, align 8, !tbaa !12
  %27 = tail call noundef double @llvm.fmuladd.f64(double %18, double %26, double %24)
  %28 = fmul double %13, %13
  %29 = tail call double @llvm.fmuladd.f64(double %8, double %8, double %28)
  %30 = tail call noundef double @llvm.fmuladd.f64(double %18, double %18, double %29)
  %31 = fneg double %30
  %32 = tail call double @llvm.fmuladd.f64(double %27, double %27, double %31)
  %33 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %34 = load double, ptr %33, align 8, !tbaa !18
  %35 = tail call double @llvm.fmuladd.f64(double %34, double %34, double %32)
  %36 = fcmp olt double %35, 0.000000e+00
  br i1 %36, label %37, label %39

37:                                               ; preds = %3
  %38 = load double, ptr @infinity, align 8, !tbaa !13
  br label %48

39:                                               ; preds = %3
  %40 = tail call double @sqrt(double noundef %35) #14, !tbaa !53
  %41 = fadd double %27, %40
  %42 = fcmp olt double %41, 0.000000e+00
  %43 = load double, ptr @infinity, align 8
  %44 = fsub double %27, %40
  %45 = fcmp ogt double %44, 0.000000e+00
  %46 = select i1 %45, double %44, double %41
  %47 = select i1 %42, double %43, double %46
  br label %48

48:                                               ; preds = %37, %39
  %49 = phi double [ %38, %37 ], [ %47, %39 ]
  %50 = load double, ptr %1, align 8, !tbaa !14
  %51 = fcmp ult double %49, %50
  br i1 %51, label %59, label %52

52:                                               ; preds = %48
  %53 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %54 = load double, ptr %53, align 8
  %55 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %56 = load double, ptr %55, align 8
  %57 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %58 = load double, ptr %57, align 8
  br label %90

59:                                               ; preds = %48
  %60 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %61 = load ptr, ptr %60, align 8, !tbaa !25
  %62 = icmp eq ptr %61, %60
  br i1 %62, label %63, label %71

63:                                               ; preds = %59
  %64 = load double, ptr %4, align 8
  %65 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %66 = load double, ptr %65, align 8
  %67 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %68 = load double, ptr %67, align 8
  %69 = getelementptr inbounds nuw i8, ptr %4, i64 24
  %70 = load double, ptr %69, align 8
  br label %90

71:                                               ; preds = %59
  %72 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %73 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %74 = getelementptr inbounds nuw i8, ptr %4, i64 24
  br label %75

75:                                               ; preds = %71, %75
  %76 = phi ptr [ %61, %71 ], [ %88, %75 ]
  %77 = getelementptr inbounds nuw i8, ptr %76, i64 16
  %78 = load ptr, ptr %77, align 8, !tbaa !29
  %79 = load ptr, ptr %78, align 8, !tbaa !16
  %80 = getelementptr inbounds nuw i8, ptr %79, i64 16
  %81 = load ptr, ptr %80, align 8
  %82 = call %"struct.std::pair" %81(ptr noundef nonnull align 8 dereferenceable(8) %78, ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(48) %2)
  %83 = extractvalue %"struct.std::pair" %82, 0
  %84 = extractvalue %"struct.std::pair" %82, 1
  %85 = extractvalue %struct.Vec %84, 0
  %86 = extractvalue %struct.Vec %84, 1
  %87 = extractvalue %struct.Vec %84, 2
  store double %83, ptr %4, align 8, !tbaa !14
  store double %85, ptr %72, align 8, !tbaa !13
  store double %86, ptr %73, align 8, !tbaa !13
  store double %87, ptr %74, align 8, !tbaa !13
  %88 = load ptr, ptr %76, align 8, !tbaa !25
  %89 = icmp eq ptr %88, %60
  br i1 %89, label %90, label %75, !llvm.loop !55

90:                                               ; preds = %75, %63, %52
  %91 = phi double [ %50, %52 ], [ %64, %63 ], [ %83, %75 ]
  %92 = phi double [ %54, %52 ], [ %66, %63 ], [ %85, %75 ]
  %93 = phi double [ %56, %52 ], [ %68, %63 ], [ %86, %75 ]
  %94 = phi double [ %58, %52 ], [ %70, %63 ], [ %87, %75 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #14
  %95 = insertvalue %"struct.std::pair" poison, double %91, 0
  %96 = insertvalue %"struct.std::pair" %95, double %92, 1, 0
  %97 = insertvalue %"struct.std::pair" %96, double %93, 1, 1
  %98 = insertvalue %"struct.std::pair" %97, double %94, 1, 2
  ret %"struct.std::pair" %98
}

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #11

; Function Attrs: nounwind
declare void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr noundef nonnull align 8 dereferenceable(16), ptr noundef) local_unnamed_addr #11

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #10

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #10

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.sqrt.f64(double) #12

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #13

attributes #0 = { mustprogress nocallback nofree nounwind willreturn memory(errnomem: write) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #12 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #13 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #14 = { nounwind }
attributes #15 = { builtin allocsize(0) }
attributes #16 = { builtin nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"_ZTS3Vec", !8, i64 0, !8, i64 8, !8, i64 16}
!8 = !{!"double", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!7, !8, i64 8}
!12 = !{!7, !8, i64 16}
!13 = !{!8, !8, i64 0}
!14 = !{!15, !8, i64 0}
!15 = !{!"_ZTSSt4pairId3VecE", !8, i64 0, !7, i64 8}
!16 = !{!17, !17, i64 0}
!17 = !{!"vtable pointer", !10, i64 0}
!18 = !{!19, !8, i64 32}
!19 = !{!"_ZTS6Sphere", !20, i64 0, !7, i64 8, !8, i64 32}
!20 = !{!"_ZTS5Scene"}
!21 = !{!22, !23, i64 8}
!22 = !{!"_ZTSNSt8__detail15_List_node_baseE", !23, i64 0, !23, i64 8}
!23 = !{!"p1 _ZTSNSt8__detail15_List_node_baseE", !24, i64 0}
!24 = !{!"any pointer", !9, i64 0}
!25 = !{!22, !23, i64 0}
!26 = !{!27, !28, i64 16}
!27 = !{!"_ZTSNSt8__detail17_List_node_headerE", !22, i64 0, !28, i64 16}
!28 = !{!"long", !9, i64 0}
!29 = !{!30, !30, i64 0}
!30 = !{!"p1 _ZTS5Scene", !24, i64 0}
!31 = !{!32, !28, i64 16}
!32 = !{!"_ZTSNSt7__cxx1110_List_baseIP5SceneSaIS2_EEE", !33, i64 0}
!33 = !{!"_ZTSNSt7__cxx1110_List_baseIP5SceneSaIS2_EE10_List_implE", !27, i64 0}
!34 = distinct !{!34, !35}
!35 = !{!"llvm.loop.mustprogress"}
!36 = distinct !{!36, !35}
!37 = !{!38, !38, i64 0}
!38 = !{!"p1 omnipotent char", !24, i64 0}
!39 = distinct !{!39, !35}
!40 = !{!9, !9, i64 0}
!41 = !{!42, !28, i64 16}
!42 = !{!"_ZTSSt8ios_base", !28, i64 8, !28, i64 16, !43, i64 24, !44, i64 28, !44, i64 32, !45, i64 40, !46, i64 48, !9, i64 64, !47, i64 192, !48, i64 200, !49, i64 208}
!43 = !{!"_ZTSSt13_Ios_Fmtflags", !9, i64 0}
!44 = !{!"_ZTSSt12_Ios_Iostate", !9, i64 0}
!45 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !24, i64 0}
!46 = !{!"_ZTSNSt8ios_base6_WordsE", !24, i64 0, !28, i64 8}
!47 = !{!"int", !9, i64 0}
!48 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !24, i64 0}
!49 = !{!"_ZTSSt6locale", !50, i64 0}
!50 = !{!"p1 _ZTSNSt6locale5_ImplE", !24, i64 0}
!51 = distinct !{!51, !35}
!52 = distinct !{!52, !35}
!53 = !{!47, !47, i64 0}
!54 = distinct !{!54, !35}
!55 = distinct !{!55, !35}
