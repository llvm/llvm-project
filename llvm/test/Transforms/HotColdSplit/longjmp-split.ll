; RUN:  opt -profile-summary-cold-count=0 -hotcoldsplit -S < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.__jmp_buf_tag = type { [8 x i64], i32, %struct.__sigset_t }
%struct.__sigset_t = type { [16 x i64] }

@c = dso_local global i32 1, align 4
@buf = dso_local global [20 x i8*] zeroinitializer, align 16

; CHECK-LABEL: @f
; CHECK: f.cold.1
define dso_local void @f() #0 !prof !31 {
entry:
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  %0 = load i32, i32* @c, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.then, label %if.else, !prof !32

if.then:                                          ; preds = %entry
  ret void

if.else:                                          ; preds = %entry
  %1 = load i32, i32* @c, align 4
  %inc = add  i32 %1, 1
  store i32 %inc, i32* @c, align 4
  %2 = load i32, i32* @c, align 4
  %inc1 = add  i32 %2, 1
  store i32 %inc1, i32* @c, align 4
  %3 = load i32, i32* @c, align 4
  %inc2 = add  i32 %3, 1
  store i32 %inc2, i32* @c, align 4
  %4 = load i32, i32* @c, align 4
  %inc3 = add  i32 %4, 1
  store i32 %inc3, i32* @c, align 4
  %5 = load i32, i32* @c, align 4
  %dec = add  i32 %5, -1
  store i32 %dec, i32* @c, align 4
  %6 = load i32, i32* @c, align 4
  %dec4 = add  i32 %6, -1
  store i32 %dec4, i32* @c, align 4
  %7 = load i32, i32* @c, align 4
  %inc5 = add  i32 %7, 1
  store i32 %inc5, i32* @c, align 4
  %8 = load i32, i32* @c, align 4
  %inc6 = add  i32 %8, 1
  store i32 %inc6, i32* @c, align 4
  %9 = load i32, i32* @c, align 4
  %add = add  i32 %9, 1
  store i32 %add, i32* %i, align 4
  %10 = load i32, i32* %i, align 4
  %sub = sub  i32 %10, 1
  store i32 %sub, i32* %j, align 4
  %11 = load i32, i32* %i, align 4
  %add7 = add  i32 %11, 2
  store i32 %add7, i32* %k, align 4
  call void @longjmp(%struct.__jmp_buf_tag* bitcast ([20 x i8*]* @buf to %struct.__jmp_buf_tag*), i32 1) #3
  unreachable
}

declare dso_local void @longjmp(%struct.__jmp_buf_tag*, i32) #1

define dso_local i32 @main() #0 !prof !31 {
entry:
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  store i32 0, i32* %i, align 4
  %call = call i32 @_setjmp(%struct.__jmp_buf_tag* bitcast ([20 x i8*]* @buf to %struct.__jmp_buf_tag*)) #4
  %tobool = icmp ne i32 %call, 0
  br i1 %tobool, label %if.then, label %if.end, !prof !33

if.then:                                          ; preds = %entry
  store i32 1, i32* %retval, align 4
  br label %return

if.end:                                           ; preds = %entry
  call void @f()
  store i32 0, i32* %retval, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %0 = load i32, i32* %retval, align 4
  ret i32 %0
}

declare dso_local i32 @_setjmp(%struct.__jmp_buf_tag*) #2

attributes #0 = { inlinehint nounwind uwtable }
attributes #1 = { noreturn nounwind }
attributes #2 = { nounwind returns_twice }
attributes #3 = { noreturn nounwind }
attributes #4 = { nounwind returns_twice }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10, !11, !12}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 2}
!5 = !{!"MaxCount", i64 1}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1}
!8 = !{!"NumCounts", i64 4}
!9 = !{!"NumFunctions", i64 2}
!10 = !{!"IsPartialProfile", i64 0}
!11 = !{!"PartialProfileRatio", double 0.000000e+00}
!12 = !{!"DetailedSummary", !13}
!13 = !{!14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29}
!14 = !{i32 10000, i64 0, i32 0}
!15 = !{i32 100000, i64 0, i32 0}
!16 = !{i32 200000, i64 0, i32 0}
!17 = !{i32 300000, i64 0, i32 0}
!18 = !{i32 400000, i64 0, i32 0}
!19 = !{i32 500000, i64 1, i32 2}
!20 = !{i32 600000, i64 1, i32 2}
!21 = !{i32 700000, i64 1, i32 2}
!22 = !{i32 800000, i64 1, i32 2}
!23 = !{i32 900000, i64 1, i32 2}
!24 = !{i32 950000, i64 1, i32 2}
!25 = !{i32 990000, i64 1, i32 2}
!26 = !{i32 999000, i64 1, i32 2}
!27 = !{i32 999900, i64 1, i32 2}
!28 = !{i32 999990, i64 1, i32 2}
!29 = !{i32 999999, i64 1, i32 2}
!31 = !{!"function_entry_count", i64 1}
!32 = !{!"branch_weights", i32 1, i32 0}
!33 = !{!"branch_weights", i32 0, i32 1}
