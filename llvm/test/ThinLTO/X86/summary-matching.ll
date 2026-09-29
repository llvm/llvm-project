;; Test to make sure that function's definiton and summary matches.
; RUN: split-file %s %t
; RUN: opt -thinlto-bc %t/main.ll >%t/main.o
; RUN: opt -thinlto-bc %t/b.ll >%t/b.o
; RUN: opt -thinlto-bc %t/c.ll >%t/c.o

; RUN: llvm-lto2 run %t/b.o %t/c.o %t/main.o -enable-memprof-context-disambiguation \
; RUN: -supports-hot-cold-new -o %t/a.out \
; RUN: -r=%t/main.o,main,plx \
; RUN: -r=%t/b.o,_Z1bv,plx \
; RUN: -r=%t/b.o,_Z3fooIiET_S0_S0_,plx \
; RUN: -r=%t/b.o,_Znwm \
; RUN: -r=%t/c.o,_Z1cv,plx \
; RUN: -r=%t/c.o,_Z3fooIiET_S0_S0_ \
; RUN: -r=%t/c.o,_Z3barIiET_S0_S0_,plx \
; RUN: -r=%t/c.o,_Znwm \
; RUN: -r=%t/main.o,_Z1bv \
; RUN: -r=%t/main.o,_Z1cv \
; RUN: -r=%t/main.o,_Z3fooIiET_S0_S0_ 

;; foo has two copys:
;; foo in b.ll is prevailing and inlines bar.
;; foo in c.ll isn't prevailing and doesn't inline bar.
;; main will import foo in c.ll and foo's summary in b.ll default.

;--- main.ll
; ModuleID = 'main.cc'
source_filename = "main.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  %call = call noundef i32 @_Z1bv(), !callsite !6
  %call1 = call noundef i32 @_Z1cv(), !callsite !7
  %add = add nsw i32 %call, %call1
  %call2 = call noundef i32 @_Z3fooIiET_S0_S0_(i32 noundef 1, i32 noundef 2), !callsite !8
  %add3 = add nsw i32 %add, %call2
  ret i32 %add3
}

declare noundef i32 @_Z1bv() #1

declare noundef i32 @_Z1cv() #1

declare noundef i32 @_Z3fooIiET_S0_S0_(i32 noundef, i32 noundef) #1

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 19.0.0"}
!6 = !{i64 1}
!7 = !{i64 5}
!8 = !{i64 7}

;--- c.ll
; ModuleID = 'c.cc'
source_filename = "c.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_Z3fooIiET_S0_S0_ = comdat any

$_Z3barIiET_S0_S0_ = comdat any

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local noundef i32 @_Z1cv() #0 {
entry:
  %num1 = alloca i32, align 4
  %num2 = alloca i32, align 4
  store i32 1, ptr %num1, align 4
  store i32 1, ptr %num2, align 4
  %0 = load i32, ptr %num1, align 4
  %1 = load i32, ptr %num2, align 4
  %call = call noundef i32 @_Z3fooIiET_S0_S0_(i32 noundef %0, i32 noundef %1), !callsite !6
  ret i32 %call
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i32 @_Z3fooIiET_S0_S0_(i32 noundef %a, i32 noundef %b) #3 comdat {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %rtn = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %b.addr, align 4
  %call = call noundef i32 @_Z3barIiET_S0_S0_(i32 noundef %0, i32 noundef %1), !callsite !7
  store i32 %call, ptr %rtn, align 4
  %2 = load i32, ptr %rtn, align 4
  ret i32 %2
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local noundef i32 @_Z3barIiET_S0_S0_(i32 noundef %a, i32 noundef %b) #0 comdat {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %c = alloca ptr, align 8
  %d = alloca ptr, align 8
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, ptr %a.addr, align 4
  %1 = load i32, ptr %b.addr, align 4
  %add1 = add nsw i32 %1, 1
  store i32 %add1, ptr %b.addr, align 4
  %2 = load i32, ptr %a.addr, align 4
  %add2 = add nsw i32 %2, 1
  store i32 %add2, ptr %a.addr, align 4
  %3 = load i32, ptr %b.addr, align 4
  %add3 = add nsw i32 %3, 1
  store i32 %add3, ptr %b.addr, align 4
  %4 = load i32, ptr %a.addr, align 4
  %add4 = add nsw i32 %4, 1
  store i32 %add4, ptr %a.addr, align 4
  %5 = load i32, ptr %b.addr, align 4
  %add5 = add nsw i32 %5, 1
  store i32 %add5, ptr %b.addr, align 4
  %6 = load i32, ptr %a.addr, align 4
  %add6 = add nsw i32 %6, 1
  store i32 %add6, ptr %a.addr, align 4
  %7 = load i32, ptr %b.addr, align 4
  %add7 = add nsw i32 %7, 1
  store i32 %add7, ptr %b.addr, align 4
  %8 = load i32, ptr %a.addr, align 4
  %add8 = add nsw i32 %8, 1
  store i32 %add8, ptr %a.addr, align 4
  %9 = load i32, ptr %b.addr, align 4
  %add9 = add nsw i32 %9, 1
  store i32 %add9, ptr %b.addr, align 4
  %10 = load i32, ptr %a.addr, align 4
  %add10 = add nsw i32 %10, 1
  store i32 %add10, ptr %a.addr, align 4
  %11 = load i32, ptr %b.addr, align 4
  %add11 = add nsw i32 %11, 1
  store i32 %add11, ptr %b.addr, align 4
  %12 = load i32, ptr %a.addr, align 4
  %add12 = add nsw i32 %12, 1
  store i32 %add12, ptr %a.addr, align 4
  %13 = load i32, ptr %b.addr, align 4
  %add13 = add nsw i32 %13, 1
  store i32 %add13, ptr %b.addr, align 4
  %14 = load i32, ptr %a.addr, align 4
  %add14 = add nsw i32 %14, 1
  store i32 %add14, ptr %a.addr, align 4
  %15 = load i32, ptr %b.addr, align 4
  %add15 = add nsw i32 %15, 1
  store i32 %add15, ptr %b.addr, align 4
  %16 = load i32, ptr %a.addr, align 4
  %add16 = add nsw i32 %16, 1
  store i32 %add16, ptr %a.addr, align 4
  %17 = load i32, ptr %b.addr, align 4
  %add17 = add nsw i32 %17, 1
  store i32 %add17, ptr %b.addr, align 4
  %18 = load i32, ptr %a.addr, align 4
  %add18 = add nsw i32 %18, 1
  store i32 %add18, ptr %a.addr, align 4
  %19 = load i32, ptr %b.addr, align 4
  %add19 = add nsw i32 %19, 1
  store i32 %add19, ptr %b.addr, align 4
  %20 = load i32, ptr %a.addr, align 4
  %add20 = add nsw i32 %20, 1
  store i32 %add20, ptr %a.addr, align 4
  %21 = load i32, ptr %b.addr, align 4
  %add21 = add nsw i32 %21, 1
  store i32 %add21, ptr %b.addr, align 4
  %22 = load i32, ptr %a.addr, align 4
  %add22 = add nsw i32 %22, 1
  store i32 %add22, ptr %a.addr, align 4
  %23 = load i32, ptr %b.addr, align 4
  %add23 = add nsw i32 %23, 1
  store i32 %add23, ptr %b.addr, align 4
  %call = call noalias noundef nonnull ptr @_Znwm(i64 noundef 4) #2, !callsite !8
  store i32 1, ptr %call, align 4
  store ptr %call, ptr %c, align 8
  %call24 = call noalias noundef nonnull ptr @_Znwm(i64 noundef 4) #2, !callsite !9
  store i32 1, ptr %call24, align 4
  store ptr %call24, ptr %d, align 8
  %24 = load i32, ptr %a.addr, align 4
  %25 = load i32, ptr %b.addr, align 4
  %cmp = icmp sgt i32 %24, %25
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  %26 = load i32, ptr %a.addr, align 4
  br label %cond.end

cond.false:                                       ; preds = %entry
  %27 = load i32, ptr %b.addr, align 4
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %26, %cond.true ], [ %27, %cond.false ]
  ret i32 %cond
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) #1

attributes #0 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nobuiltin allocsize(0) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { builtin allocsize(0) }
attributes #3 = { mustprogress uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 19.0.0"}
!6 = !{i64 6}
!7 = !{i64 3}
!8 = !{i64 4}
!9 = !{i64 9}

;--- b.ll
; ModuleID = 'b.cc'
source_filename = "b.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_Z3fooIiET_S0_S0_ = comdat any

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local noundef i32 @_Z1bv() #0 {
entry:
  %num1 = alloca i32, align 4
  %num2 = alloca i32, align 4
  store i32 0, ptr %num1, align 4
  store i32 0, ptr %num2, align 4
  %0 = load i32, ptr %num1, align 4
  %1 = load i32, ptr %num2, align 4
  %call = call noundef i32 @_Z3fooIiET_S0_S0_(i32 noundef %0, i32 noundef %1), !callsite !6
  ret i32 %call
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i32 @_Z3fooIiET_S0_S0_(i32 noundef %a, i32 noundef %b) #3 comdat {
entry:
  %a.addr.i = alloca i32, align 4
  %b.addr.i = alloca i32, align 4
  %c.i = alloca ptr, align 8
  %d.i = alloca ptr, align 8
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %rtn = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %b.addr, align 4
  store i32 %0, ptr %a.addr.i, align 4
  store i32 %1, ptr %b.addr.i, align 4
  %2 = load i32, ptr %a.addr.i, align 4
  %add.i = add nsw i32 %2, 1
  store i32 %add.i, ptr %a.addr.i, align 4
  %3 = load i32, ptr %b.addr.i, align 4
  %add1.i = add nsw i32 %3, 1
  store i32 %add1.i, ptr %b.addr.i, align 4
  %4 = load i32, ptr %a.addr.i, align 4
  %add2.i = add nsw i32 %4, 1
  store i32 %add2.i, ptr %a.addr.i, align 4
  %5 = load i32, ptr %b.addr.i, align 4
  %add3.i = add nsw i32 %5, 1
  store i32 %add3.i, ptr %b.addr.i, align 4
  %6 = load i32, ptr %a.addr.i, align 4
  %add4.i = add nsw i32 %6, 1
  store i32 %add4.i, ptr %a.addr.i, align 4
  %7 = load i32, ptr %b.addr.i, align 4
  %add5.i = add nsw i32 %7, 1
  store i32 %add5.i, ptr %b.addr.i, align 4
  %8 = load i32, ptr %a.addr.i, align 4
  %add6.i = add nsw i32 %8, 1
  store i32 %add6.i, ptr %a.addr.i, align 4
  %9 = load i32, ptr %b.addr.i, align 4
  %add7.i = add nsw i32 %9, 1
  store i32 %add7.i, ptr %b.addr.i, align 4
  %10 = load i32, ptr %a.addr.i, align 4
  %add8.i = add nsw i32 %10, 1
  store i32 %add8.i, ptr %a.addr.i, align 4
  %11 = load i32, ptr %b.addr.i, align 4
  %add9.i = add nsw i32 %11, 1
  store i32 %add9.i, ptr %b.addr.i, align 4
  %12 = load i32, ptr %a.addr.i, align 4
  %add10.i = add nsw i32 %12, 1
  store i32 %add10.i, ptr %a.addr.i, align 4
  %13 = load i32, ptr %b.addr.i, align 4
  %add11.i = add nsw i32 %13, 1
  store i32 %add11.i, ptr %b.addr.i, align 4
  %14 = load i32, ptr %a.addr.i, align 4
  %add12.i = add nsw i32 %14, 1
  store i32 %add12.i, ptr %a.addr.i, align 4
  %15 = load i32, ptr %b.addr.i, align 4
  %add13.i = add nsw i32 %15, 1
  store i32 %add13.i, ptr %b.addr.i, align 4
  %16 = load i32, ptr %a.addr.i, align 4
  %add14.i = add nsw i32 %16, 1
  store i32 %add14.i, ptr %a.addr.i, align 4
  %17 = load i32, ptr %b.addr.i, align 4
  %add15.i = add nsw i32 %17, 1
  store i32 %add15.i, ptr %b.addr.i, align 4
  %18 = load i32, ptr %a.addr.i, align 4
  %add16.i = add nsw i32 %18, 1
  store i32 %add16.i, ptr %a.addr.i, align 4
  %19 = load i32, ptr %b.addr.i, align 4
  %add17.i = add nsw i32 %19, 1
  store i32 %add17.i, ptr %b.addr.i, align 4
  %20 = load i32, ptr %a.addr.i, align 4
  %add18.i = add nsw i32 %20, 1
  store i32 %add18.i, ptr %a.addr.i, align 4
  %21 = load i32, ptr %b.addr.i, align 4
  %add19.i = add nsw i32 %21, 1
  store i32 %add19.i, ptr %b.addr.i, align 4
  %22 = load i32, ptr %a.addr.i, align 4
  %add20.i = add nsw i32 %22, 1
  store i32 %add20.i, ptr %a.addr.i, align 4
  %23 = load i32, ptr %b.addr.i, align 4
  %add21.i = add nsw i32 %23, 1
  store i32 %add21.i, ptr %b.addr.i, align 4
  %24 = load i32, ptr %a.addr.i, align 4
  %add22.i = add nsw i32 %24, 1
  store i32 %add22.i, ptr %a.addr.i, align 4
  %25 = load i32, ptr %b.addr.i, align 4
  %add23.i = add nsw i32 %25, 1
  store i32 %add23.i, ptr %b.addr.i, align 4
  %call.i = call noalias noundef nonnull ptr @_Znwm(i64 noundef 4) #2, !callsite !7
  store i32 1, ptr %call.i, align 4
  store ptr %call.i, ptr %c.i, align 8
  %call24.i = call noalias noundef nonnull ptr @_Znwm(i64 noundef 4) #2, !callsite !8
  store i32 1, ptr %call24.i, align 4
  store ptr %call24.i, ptr %d.i, align 8
  %26 = load i32, ptr %a.addr.i, align 4
  %27 = load i32, ptr %b.addr.i, align 4
  %cmp.i = icmp sgt i32 %26, %27
  br i1 %cmp.i, label %cond.true.i, label %cond.false.i

cond.true.i:                                      ; preds = %entry
  %28 = load i32, ptr %a.addr.i, align 4
  br label %_Z3barIiET_S0_S0_.exit

cond.false.i:                                     ; preds = %entry
  %29 = load i32, ptr %b.addr.i, align 4
  br label %_Z3barIiET_S0_S0_.exit

_Z3barIiET_S0_S0_.exit:                           ; preds = %cond.true.i, %cond.false.i
  %cond.i = phi i32 [ %28, %cond.true.i ], [ %29, %cond.false.i ]
  store i32 %cond.i, ptr %rtn, align 4
  %30 = load i32, ptr %rtn, align 4
  ret i32 %30
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) #1

attributes #0 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nobuiltin allocsize(0) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { builtin allocsize(0) }
attributes #3 = { mustprogress uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 19.0.0"}
!6 = !{i64 2}
!7 = !{i64 4, i64 3}
!8 = !{i64 9, i64 3}
