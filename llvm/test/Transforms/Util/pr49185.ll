; REQUIRES: asserts
; RUN: opt -S -passes='default<O3>' %s
%struct0 = type { i64, i64, i32, i64, i32 }
%struct1 = type { i32 }
%union0 = type { i32 }
%union1 = type { i16 }
%union2 = type { i32 }

@g_6 = external dso_local global i32, align 1
@g_60 = external dso_local global i16, align 1
@g_79 = external dso_local global { i16, i16 }, align 1
@g_315 = external dso_local global %struct0, align 1
@g_359 = external dso_local global %struct0, align 1

define dso_local i16 @main(i16 %argc, i16** %argv) #0 {
entry:
  %call2 = call i16 @func_1()
  unreachable
}

define internal i16 @func_1() #0 {
entry:
  %call = call i16 @func_21(i32* undef, i32 undef, i32* undef)
  ret i16 undef
}

define internal i16 @func_21(i32* %p_22, i32 %p_23, i32* %p_24) #0 {
entry:
  call void @func_34(%struct0* align 1 undef, i32 undef, i32 undef, i32* @g_6, %union0* byval(%union0) align 1 undef)
  unreachable
}

define internal void @func_34(%struct0* %agg.result, i32 %p_35, i32 %p_36, i32* %p_37, %union0* %p_38) #0 {
entry:
  %p_37.addr = alloca i32*, align 1
  %cleanup.dest.slot = alloca i32, align 1
  store i32* %p_37, i32** %p_37.addr, align 1
  br label %lbl_898

lbl_898:                                          ; preds = %cleanup3097, %entry
  br label %lbl_1111

lbl_1111:                                         ; preds = %cleanup3097, %lbl_898
  %0 = load i32, i32* getelementptr inbounds (%struct0, %struct0* @g_359, i32 0, i32 4), align 1
  %tobool1833 = icmp ne i32 %0, 0
  br i1 %tobool1833, label %land.rhs1834, label %land.end1851

land.rhs1834:                                     ; preds = %lbl_1111
  store i16 0, i16* @g_60, align 1
  br label %land.end1851

land.end1851:                                     ; preds = %land.rhs1834, %lbl_1111
  %1 = load i32*, i32** %p_37.addr, align 1
  %2 = load i32, i32* %1, align 1
  %tobool2351 = icmp ne i32 %2, 0
  br i1 %tobool2351, label %if.then2352, label %if.else3029

if.then2352:                                      ; preds = %land.end1851
  %3 = load i16, i16* getelementptr inbounds ({ i16, i16 }, { i16, i16 }* @g_79, i32 0, i32 0), align 1, !tbaa !1
  %tobool3011 = icmp ne i16 %3, 0
  call void @llvm.assume(i1 %tobool3011)
  store i32 11, i32* %cleanup.dest.slot, align 1
  br label %cleanup3097

if.else3029:                                      ; preds = %land.end1851
  store i32 3, i32* getelementptr inbounds (%struct0, %struct0* @g_315, i32 0, i32 4), align 1
  store i32 132, i32* %cleanup.dest.slot, align 1
  br label %cleanup3097

cleanup3097:                                      ; preds = %if.else3029, %if.then2352
  %cleanup.dest3113 = load i32, i32* %cleanup.dest.slot, align 1
  switch i32 %cleanup.dest3113, label %cleanup3402 [
    i32 132, label %lbl_1111
    i32 11, label %lbl_898
  ]

cleanup3402:                                      ; preds = %cleanup3097
  ret void
}

; Function Attrs: nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #4

attributes #0 = { "use-soft-float"="false" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { noinline }
attributes #3 = { argmemonly nofree nosync nounwind willreturn writeonly }
attributes #4 = { nofree nosync nounwind willreturn }

!llvm.ident = !{!0}

!0 = !{!"clang version 13.0.0"}
!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
