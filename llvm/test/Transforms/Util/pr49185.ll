; REQUIRES: asserts
; RUN: opt -S -passes='default<O3>' %s
%struct0 = type { i64, i64, i32, i64, i32 }
%union0 = type { i32 }

@g_6 = external dso_local global i32, align 1
@g_60 = external dso_local global i16, align 1
@g_79 = external dso_local global { i16, i16 }, align 1
@g_315 = external dso_local global %struct0, align 1
@g_359 = external dso_local global %struct0, align 1

define dso_local i16 @main(i16 %argc, ptr %argv) #0 {
entry:
  %call2 = call i16 @func_1()
  unreachable
}

define internal i16 @func_1() #0 {
entry:
  %call = call i16 @func_21(ptr undef, i32 undef, ptr undef)
  ret i16 undef
}

define internal i16 @func_21(ptr %p_22, i32 %p_23, ptr %p_24) #0 {
entry:
  call void @func_34(ptr align 1 undef, i32 undef, i32 undef, ptr @g_6, ptr byval(%union0) align 1 undef)
  unreachable
}

define internal void @func_34(ptr %agg.result, i32 %p_35, i32 %p_36, ptr %p_37, ptr %p_38) #0 {
entry:
  %p_37.addr = alloca ptr, align 1
  %cleanup.dest.slot = alloca i32, align 1
  store ptr %p_37, ptr %p_37.addr, align 1
  br label %lbl_898

lbl_898:                                          ; preds = %cleanup3097, %entry
  br label %lbl_1111

lbl_1111:                                         ; preds = %cleanup3097, %lbl_898
  %0 = load i32, ptr getelementptr inbounds (%struct0, ptr @g_359, i32 0, i32 4), align 1
  %tobool1833 = icmp ne i32 %0, 0
  br i1 %tobool1833, label %land.rhs1834, label %land.end1851

land.rhs1834:                                     ; preds = %lbl_1111
  store i16 0, ptr @g_60, align 1
  br label %land.end1851

land.end1851:                                     ; preds = %land.rhs1834, %lbl_1111
  %1 = load ptr, ptr %p_37.addr, align 1
  %2 = load i32, ptr %1, align 1
  %tobool2351 = icmp ne i32 %2, 0
  br i1 %tobool2351, label %if.then2352, label %if.else3029

if.then2352:                                      ; preds = %land.end1851
  %3 = load i16, ptr getelementptr inbounds ({ i16, i16 }, ptr @g_79, i32 0, i32 0), align 1, !tbaa !1
  %tobool3011 = icmp ne i16 %3, 0
  call void @llvm.assume(i1 %tobool3011)
  store i32 11, ptr %cleanup.dest.slot, align 1
  br label %cleanup3097

if.else3029:                                      ; preds = %land.end1851
  store i32 3, ptr getelementptr inbounds (%struct0, ptr @g_315, i32 0, i32 4), align 1
  store i32 132, ptr %cleanup.dest.slot, align 1
  br label %cleanup3097

cleanup3097:                                      ; preds = %if.else3029, %if.then2352
  %cleanup.dest3113 = load i32, ptr %cleanup.dest.slot, align 1
  switch i32 %cleanup.dest3113, label %cleanup3402 [
    i32 132, label %lbl_1111
    i32 11, label %lbl_898
  ]

cleanup3402:                                      ; preds = %cleanup3097
  ret void
}

; Function Attrs: nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #1

attributes #0 = { "use-soft-float"="false" }
attributes #1 = { nofree nosync nounwind willreturn }

!llvm.ident = !{!0}

!0 = !{!"clang version 13.0.0"}
!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
