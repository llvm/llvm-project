; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/recursive-throw.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/recursive-throw.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

$_ZTI13TestException = comdat any

$_ZTS13TestException = comdat any

@thrown = dso_local local_unnamed_addr global i8 0, align 4
@caught = dso_local local_unnamed_addr global i8 0, align 4
@_ZTI13TestException = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS13TestException }, comdat, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@_ZTS13TestException = linkonce_odr dso_local constant [16 x i8] c"13TestException\00", comdat, align 1

; Function Attrs: cold mustprogress noreturn uwtable
define dso_local void @_Z3thri(i32 %0) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  store i8 1, ptr @thrown, align 4, !tbaa !6
  %2 = tail call ptr @__cxa_allocate_exception(i64 1) #6
  tail call void @__cxa_throw(ptr nonnull %2, ptr nonnull @_ZTI13TestException, ptr null) #7
  unreachable
}

declare ptr @__cxa_allocate_exception(i64) local_unnamed_addr

declare i32 @__gxx_personality_v0(...)

; Function Attrs: cold noreturn
declare void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr #1

; Function Attrs: cold mustprogress noinline uwtable
define dso_local void @_Z3runv() local_unnamed_addr #2 personality ptr @__gxx_personality_v0 {
  invoke void @_Z3thri(i32 poison)
          to label %1 unwind label %2

1:                                                ; preds = %0
  unreachable

2:                                                ; preds = %0
  %3 = landingpad { ptr, i32 }
          catch ptr @_ZTI13TestException
  %4 = extractvalue { ptr, i32 } %3, 1
  %5 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI13TestException) #6
  %6 = icmp eq i32 %4, %5
  br i1 %6, label %7, label %10

7:                                                ; preds = %2
  %8 = extractvalue { ptr, i32 } %3, 0
  %9 = tail call ptr @__cxa_begin_catch(ptr %8) #6
  store i8 1, ptr @caught, align 4, !tbaa !6
  tail call void @__cxa_end_catch()
  ret void

10:                                               ; preds = %2
  resume { ptr, i32 } %3
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nofree nosync nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #4

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: cold mustprogress norecurse nounwind uwtable
define dso_local noundef range(i32 0, 2) i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #5 personality ptr @__gxx_personality_v0 {
  invoke void @_Z3runv()
          to label %7 unwind label %3

3:                                                ; preds = %2
  %4 = landingpad { ptr, i32 }
          catch ptr null
  %5 = extractvalue { ptr, i32 } %4, 0
  %6 = tail call ptr @__cxa_begin_catch(ptr %5) #6
  tail call void @abort() #8
  unreachable

7:                                                ; preds = %2
  %8 = load i8, ptr @thrown, align 4, !tbaa !6, !range !10, !noundef !11
  %9 = trunc nuw i8 %8 to i1
  %10 = load i8, ptr @caught, align 4, !range !10
  %11 = xor i8 %10, 1
  %12 = zext nneg i8 %11 to i32
  %13 = select i1 %9, i32 %12, i32 1
  ret i32 %13
}

attributes #0 = { cold mustprogress noreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold noreturn }
attributes #2 = { cold mustprogress noinline uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nosync nounwind memory(none) }
attributes #5 = { cold mustprogress norecurse nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nounwind }
attributes #7 = { noreturn }
attributes #8 = { cold noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"bool", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{i8 0, i8 2}
!11 = !{}
