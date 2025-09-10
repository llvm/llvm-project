; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/ctor_dtor_count.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/ctor_dtor_count.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

$_ZN1AD2Ev = comdat any

$_ZTI1A = comdat any

$_ZTS1A = comdat any

$_ZTIP1A = comdat any

$_ZTSP1A = comdat any

@_ZL1c = internal unnamed_addr global i32 0, align 4
@_ZTI1A = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1A }, comdat, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@_ZTS1A = linkonce_odr dso_local constant [3 x i8] c"1A\00", comdat, align 1
@_ZTIP1A = linkonce_odr dso_local constant { ptr, ptr, i32, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 2), ptr @_ZTSP1A, i32 0, ptr @_ZTI1A }, comdat, align 8
@_ZTVN10__cxxabiv119__pointer_type_infoE = external global [0 x ptr]
@_ZTSP1A = linkonce_odr dso_local constant [4 x i8] c"P1A\00", comdat, align 1
@_ZL1k = internal unnamed_addr global i32 0, align 4
@_ZTIi = external constant ptr
@str = private unnamed_addr constant [10 x i8] c"Deriv ok!\00", align 4
@str.11 = private unnamed_addr constant [18 x i8] c"Deriv pointer ok!\00", align 4
@str.12 = private unnamed_addr constant [9 x i8] c"Base ok!\00", align 4
@str.13 = private unnamed_addr constant [17 x i8] c"Base pointer ok!\00", align 4
@str.14 = private unnamed_addr constant [9 x i8] c"Copy ok!\00", align 4
@str.15 = private unnamed_addr constant [17 x i8] c"Copy pointer ok!\00", align 4
@str.17 = private unnamed_addr constant [20 x i8] c"Member positive ok!\00", align 4
@str.19 = private unnamed_addr constant [16 x i8] c"Member zero ok!\00", align 4
@str.20 = private unnamed_addr constant [18 x i8] c"caught negative T\00", align 4
@str.21 = private unnamed_addr constant [20 x i8] c"Member negative ok!\00", align 4

; Function Attrs: cold mustprogress uwtable
define dso_local noundef i32 @_Z6simplev() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %1 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %2 = add nsw i32 %1, 2
  store i32 %2, ptr @_ZL1c, align 4, !tbaa !6
  %3 = tail call ptr @__cxa_allocate_exception(i64 4) #9
  store i32 1, ptr %3, align 16, !tbaa !6
  invoke void @__cxa_throw(ptr nonnull %3, ptr nonnull @_ZTIi, ptr null) #10
          to label %12 unwind label %4

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          catch ptr null
  %6 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %7 = add nsw i32 %6, -2
  store i32 %7, ptr @_ZL1c, align 4, !tbaa !6
  %8 = extractvalue { ptr, i32 } %5, 0
  %9 = tail call ptr @__cxa_begin_catch(ptr %8) #9
  tail call void @__cxa_end_catch()
  %10 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %11 = icmp eq i32 %10, 0
  br i1 %11, label %13, label %16

12:                                               ; preds = %0
  unreachable

13:                                               ; preds = %4
  %14 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  %15 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  br label %16

16:                                               ; preds = %4, %13
  %17 = phi i32 [ %10, %4 ], [ %15, %13 ]
  %18 = add nsw i32 %17, 2
  store i32 %18, ptr @_ZL1c, align 4, !tbaa !6
  %19 = tail call ptr @__cxa_allocate_exception(i64 4) #9
  store i32 1, ptr %19, align 16, !tbaa !6
  invoke void @__cxa_throw(ptr nonnull %19, ptr nonnull @_ZTIi, ptr null) #10
          to label %28 unwind label %20

20:                                               ; preds = %16
  %21 = landingpad { ptr, i32 }
          catch ptr null
  %22 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %23 = add nsw i32 %22, -2
  store i32 %23, ptr @_ZL1c, align 4, !tbaa !6
  %24 = extractvalue { ptr, i32 } %21, 0
  %25 = tail call ptr @__cxa_begin_catch(ptr %24) #9
  tail call void @__cxa_end_catch()
  %26 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %29, label %31

28:                                               ; preds = %16
  unreachable

29:                                               ; preds = %20
  %30 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.11)
  br label %31

31:                                               ; preds = %29, %20
  %32 = tail call ptr @__cxa_allocate_exception(i64 1) #9
  %33 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %34 = add nsw i32 %33, 1
  store i32 %34, ptr @_ZL1c, align 4, !tbaa !6
  invoke void @__cxa_throw(ptr nonnull %32, ptr nonnull @_ZTI1A, ptr nonnull @_ZN1AD2Ev) #10
          to label %127 unwind label %35

35:                                               ; preds = %31
  %36 = landingpad { ptr, i32 }
          catch ptr null
  %37 = extractvalue { ptr, i32 } %36, 0
  %38 = tail call ptr @__cxa_begin_catch(ptr %37) #9
  tail call void @__cxa_end_catch()
  %39 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %40 = icmp eq i32 %39, 0
  br i1 %40, label %41, label %43

41:                                               ; preds = %35
  %42 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.12)
  br label %43

43:                                               ; preds = %41, %35
  %44 = tail call ptr @__cxa_allocate_exception(i64 8) #9
  %45 = invoke noalias noundef nonnull dereferenceable(1) ptr @_Znwm(i64 noundef 1) #11
          to label %46 unwind label %49

46:                                               ; preds = %43
  %47 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %48 = add nsw i32 %47, 1
  store i32 %48, ptr @_ZL1c, align 4, !tbaa !6
  store ptr %45, ptr %44, align 16, !tbaa !10
  invoke void @__cxa_throw(ptr nonnull %44, ptr nonnull @_ZTIP1A, ptr null) #10
          to label %127 unwind label %51

49:                                               ; preds = %43
  %50 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIP1A
  tail call void @__cxa_free_exception(ptr %44) #9
  br label %53

51:                                               ; preds = %46
  %52 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1A
  br label %53

53:                                               ; preds = %51, %49
  %54 = phi { ptr, i32 } [ %52, %51 ], [ %50, %49 ]
  %55 = extractvalue { ptr, i32 } %54, 1
  %56 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIP1A) #9
  %57 = icmp eq i32 %55, %56
  br i1 %57, label %58, label %125

58:                                               ; preds = %53
  %59 = extractvalue { ptr, i32 } %54, 0
  %60 = tail call ptr @__cxa_begin_catch(ptr %59) #9
  %61 = icmp eq ptr %60, null
  br i1 %61, label %65, label %62

62:                                               ; preds = %58
  %63 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %64 = add nsw i32 %63, -1
  store i32 %64, ptr @_ZL1c, align 4, !tbaa !6
  tail call void @_ZdlPvm(ptr noundef nonnull %60, i64 noundef 1) #12
  br label %65

65:                                               ; preds = %62, %58
  tail call void @__cxa_end_catch() #9
  %66 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %67 = icmp eq i32 %66, 0
  br i1 %67, label %68, label %71

68:                                               ; preds = %65
  %69 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.13)
  %70 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  br label %71

71:                                               ; preds = %68, %65
  %72 = phi i32 [ %70, %68 ], [ %66, %65 ]
  %73 = add nsw i32 %72, 1
  store i32 %73, ptr @_ZL1c, align 4, !tbaa !6
  %74 = tail call ptr @__cxa_allocate_exception(i64 1) #9
  %75 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %76 = add nsw i32 %75, 1
  store i32 %76, ptr @_ZL1c, align 4, !tbaa !6
  invoke void @__cxa_throw(ptr nonnull %74, ptr nonnull @_ZTI1A, ptr nonnull @_ZN1AD2Ev) #10
          to label %127 unwind label %77

77:                                               ; preds = %71
  %78 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTI1A
  %79 = extractvalue { ptr, i32 } %78, 1
  %80 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %81 = add nsw i32 %80, -1
  store i32 %81, ptr @_ZL1c, align 4, !tbaa !6
  %82 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI1A) #9
  %83 = icmp eq i32 %79, %82
  br i1 %83, label %84, label %125

84:                                               ; preds = %77
  %85 = extractvalue { ptr, i32 } %78, 0
  %86 = tail call ptr @__cxa_begin_catch(ptr %85) #9
  tail call void @__cxa_end_catch()
  %87 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %88 = icmp eq i32 %87, 0
  br i1 %88, label %89, label %92

89:                                               ; preds = %84
  %90 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.14)
  %91 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  br label %92

92:                                               ; preds = %89, %84
  %93 = phi i32 [ %91, %89 ], [ %87, %84 ]
  %94 = add nsw i32 %93, 1
  store i32 %94, ptr @_ZL1c, align 4, !tbaa !6
  %95 = tail call ptr @__cxa_allocate_exception(i64 8) #9
  %96 = invoke noalias noundef nonnull dereferenceable(1) ptr @_Znwm(i64 noundef 1) #11
          to label %97 unwind label %100

97:                                               ; preds = %92
  %98 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %99 = add nsw i32 %98, 1
  store i32 %99, ptr @_ZL1c, align 4, !tbaa !6
  store ptr %96, ptr %95, align 16, !tbaa !10
  invoke void @__cxa_throw(ptr nonnull %95, ptr nonnull @_ZTIP1A, ptr null) #10
          to label %127 unwind label %102

100:                                              ; preds = %92
  %101 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIP1A
  tail call void @__cxa_free_exception(ptr %95) #9
  br label %104

102:                                              ; preds = %97
  %103 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIP1A
  br label %104

104:                                              ; preds = %102, %100
  %105 = phi { ptr, i32 } [ %103, %102 ], [ %101, %100 ]
  %106 = extractvalue { ptr, i32 } %105, 1
  %107 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %108 = add nsw i32 %107, -1
  store i32 %108, ptr @_ZL1c, align 4, !tbaa !6
  %109 = icmp eq i32 %106, %55
  br i1 %109, label %110, label %125

110:                                              ; preds = %104
  %111 = extractvalue { ptr, i32 } %105, 0
  %112 = tail call ptr @__cxa_begin_catch(ptr %111) #9
  %113 = icmp eq ptr %112, null
  br i1 %113, label %117, label %114

114:                                              ; preds = %110
  %115 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %116 = add nsw i32 %115, -1
  store i32 %116, ptr @_ZL1c, align 4, !tbaa !6
  tail call void @_ZdlPvm(ptr noundef nonnull %112, i64 noundef 1) #12
  br label %117

117:                                              ; preds = %114, %110
  tail call void @__cxa_end_catch() #9
  %118 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %119 = icmp eq i32 %118, 0
  br i1 %119, label %120, label %123

120:                                              ; preds = %117
  %121 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.15)
  %122 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  br label %123

123:                                              ; preds = %120, %117
  %124 = phi i32 [ %122, %120 ], [ %118, %117 ]
  ret i32 %124

125:                                              ; preds = %104, %77, %53
  %126 = phi { ptr, i32 } [ %105, %104 ], [ %78, %77 ], [ %54, %53 ]
  resume { ptr, i32 } %126

127:                                              ; preds = %97, %71, %46, %31
  unreachable
}

declare i32 @__gxx_personality_v0(...)

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #1

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #2

declare ptr @__cxa_allocate_exception(i64) local_unnamed_addr

declare void @__cxa_free_exception(ptr) local_unnamed_addr

; Function Attrs: cold noreturn
declare void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr #3

; Function Attrs: nofree nosync nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #4

; Function Attrs: mustprogress uwtable
define dso_local noundef i32 @_Z6memberv() local_unnamed_addr #5 personality ptr @__gxx_personality_v0 {
  %1 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %2 = load i32, ptr @_ZL1k, align 4, !tbaa !6
  %3 = add nsw i32 %2, 1
  store i32 %3, ptr @_ZL1k, align 4, !tbaa !6
  %4 = icmp eq i32 %1, 0
  %5 = icmp eq i32 %2, 0
  %6 = select i1 %4, i1 %5, i1 false
  br i1 %6, label %7, label %11

7:                                                ; preds = %0
  %8 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.17)
  %9 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %10 = load i32, ptr @_ZL1k, align 4, !tbaa !6
  br label %11

11:                                               ; preds = %0, %7
  %12 = phi i32 [ %3, %0 ], [ %10, %7 ]
  %13 = phi i32 [ %1, %0 ], [ %9, %7 ]
  %14 = add nsw i32 %12, 1
  store i32 %14, ptr @_ZL1k, align 4, !tbaa !6
  store i32 %13, ptr @_ZL1c, align 4, !tbaa !6
  %15 = icmp eq i32 %13, 0
  %16 = icmp eq i32 %14, 2
  %17 = select i1 %15, i1 %16, i1 false
  br i1 %17, label %18, label %21

18:                                               ; preds = %11
  %19 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.19)
  %20 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  br label %21

21:                                               ; preds = %18, %11
  %22 = phi i32 [ %20, %18 ], [ %13, %11 ]
  %23 = add nsw i32 %22, 1
  store i32 %23, ptr @_ZL1c, align 4, !tbaa !6
  %24 = tail call ptr @__cxa_allocate_exception(i64 4) #9
  store i32 -10, ptr %24, align 16, !tbaa !6
  invoke void @__cxa_throw(ptr nonnull %24, ptr nonnull @_ZTIi, ptr null) #10
          to label %25 unwind label %26

25:                                               ; preds = %21
  unreachable

26:                                               ; preds = %21
  %27 = landingpad { ptr, i32 }
          catch ptr null
  %28 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %29 = add nsw i32 %28, -1
  store i32 %29, ptr @_ZL1c, align 4, !tbaa !6
  %30 = extractvalue { ptr, i32 } %27, 0
  %31 = tail call ptr @__cxa_begin_catch(ptr %30) #9
  %32 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.20)
  tail call void @__cxa_end_catch()
  %33 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %34 = icmp eq i32 %33, 0
  %35 = load i32, ptr @_ZL1k, align 4
  %36 = icmp eq i32 %35, 2
  %37 = select i1 %34, i1 %36, i1 false
  br i1 %37, label %38, label %41

38:                                               ; preds = %26
  %39 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.21)
  %40 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  br label %41

41:                                               ; preds = %38, %26
  %42 = phi i32 [ %40, %38 ], [ %33, %26 ]
  ret i32 %42
}

; Function Attrs: cold mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #6 {
  %1 = tail call noundef i32 @_Z6simplev()
  %2 = tail call noundef i32 @_Z6memberv()
  %3 = add nsw i32 %2, %1
  ret i32 %3
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN1AD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #7 comdat {
  %2 = load i32, ptr @_ZL1c, align 4, !tbaa !6
  %3 = add nsw i32 %2, -1
  store i32 %3, ptr @_ZL1c, align 4, !tbaa !6
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #8

attributes #0 = { cold mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold noreturn }
attributes #4 = { nofree nosync nounwind memory(none) }
attributes #5 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { cold mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nofree nounwind }
attributes #9 = { nounwind }
attributes #10 = { noreturn }
attributes #11 = { builtin allocsize(0) }
attributes #12 = { builtin nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"p1 _ZTS1A", !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
