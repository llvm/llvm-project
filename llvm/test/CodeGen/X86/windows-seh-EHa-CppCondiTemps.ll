; RUN: llc -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: $ip2state$main:
; CHECK-NEXT:	.long	.Lfunc_begin4@IMGREL
; CHECK-NEXT:	.long	-1                              # ToState
; CHECK-NEXT:	.long	.Ltmp
; CHECK-NEXT:	.long	0                               # ToState
; CHECK-NEXT:	.long	.Ltmp
; CHECK-NEXT:	.long	4                               # ToState
; CHECK-NEXT:   .long	.Ltmp
; CHECK:     	.long	5                               # ToState
; CHECK-NEXT:	.long	.Ltmp
; CHECK-NEXT:	.long	4                               # ToState
; CHECK-NEXT:	.long	.Ltmp
; CHECK-NEXT:	.long	0                               # ToState
; CHECK-NEXT:	.long	.Ltmp
; CHECK-NEXT:	.long	2                               # ToState
; CHECK-NEXT:	.long	.Ltmp
; CHECK-NEXT:	.long	3                               # ToState
; CHECK-NEXT:	.long	.Ltmp
; CHECK-NEXT:	.long	0                               # ToState
; CHECK-NEXT:	.long	.Ltmp
; CHECK-NEXT:	.long	1                               # ToState
; CHECK-NEXT:	.long	.Ltmp
; CHECK-NEXT:	.long	0                               # ToState
; CHECK-NEXT:	.long	.Ltmp
; CHECK-NEXT:	.long	-1                              # ToState

; ModuleID = 'windows-seh-EHa-CppCondiTemps.cpp'
source_filename = "windows-seh-EHa-CppCondiTemps.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc"

%class.B1 = type { i32 }
%class.B2 = type { %class.B1 }
%class.B3 = type { %class.B2 }

$"??1B1@@QEAA@XZ" = comdat any

$"??1B2@@QEAA@XZ" = comdat any

$"??0B2@@QEAA@XZ" = comdat any

$"??0B3@@QEAA@XZ" = comdat any

$"??1B3@@QEAA@XZ" = comdat any

$"??0B1@@QEAA@XZ" = comdat any

$"??_C@_0N@FMGAAAAM@in?5B1?5Dtor?5?6?$AA@" = comdat any

$"??_C@_0N@GFONDMMJ@in?5B2?5Dtor?5?6?$AA@" = comdat any

$"??_C@_0N@HCJGCIIK@in?5B3?5Dtor?5?6?$AA@" = comdat any

@"?xxxx@@3HA" = dso_local global i32 0, align 4
@"?ptr@@3PEAHEA" = dso_local global ptr null, align 8
@"??_C@_0N@FMGAAAAM@in?5B1?5Dtor?5?6?$AA@" = linkonce_odr dso_local unnamed_addr constant [13 x i8] c"in B1 Dtor \0A\00", comdat, align 1
@"??_C@_0N@GFONDMMJ@in?5B2?5Dtor?5?6?$AA@" = linkonce_odr dso_local unnamed_addr constant [13 x i8] c"in B2 Dtor \0A\00", comdat, align 1
@"??_C@_0N@HCJGCIIK@in?5B3?5Dtor?5?6?$AA@" = linkonce_odr dso_local unnamed_addr constant [13 x i8] c"in B3 Dtor \0A\00", comdat, align 1

; Function Attrs: noinline nounwind optnone mustprogress
define dso_local i32 @"?foo@@YAHH@Z"(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  %0 = load i32, ptr @"?xxxx@@3HA", align 4
  %1 = load i32, ptr %a.addr, align 4
  %add = add nsw i32 %0, %1
  ret i32 %add
}

; Function Attrs: noinline optnone mustprogress
define dso_local i32 @"?bar@@YAHHVB1@@VB2@@@Z"(i32 %j, i32 %b1Bar.coerce, i32 %b2Bar.coerce) #1 personality ptr @__CxxFrameHandler3 {
entry:
  %b1Bar = alloca %class.B1, align 4
  %b2Bar = alloca %class.B2, align 4
  %j.addr = alloca i32, align 4
  %ww = alloca i32, align 4
  %coerce.dive = getelementptr inbounds %class.B1, ptr %b1Bar, i32 0, i32 0
  store i32 %b1Bar.coerce, ptr %coerce.dive, align 4
  %coerce.dive1 = getelementptr inbounds %class.B2, ptr %b2Bar, i32 0, i32 0
  %coerce.dive2 = getelementptr inbounds %class.B1, ptr %coerce.dive1, i32 0, i32 0
  store i32 %b2Bar.coerce, ptr %coerce.dive2, align 4
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont unwind label %ehcleanup7

invoke.cont:                                      ; preds = %entry
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont3 unwind label %ehcleanup

invoke.cont3:                                     ; preds = %invoke.cont
  store i32 %j, ptr %j.addr, align 4
  %0 = load i32, ptr %j.addr, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %invoke.cont3
  %data = getelementptr inbounds %class.B1, ptr %b1Bar, i32 0, i32 0
  %1 = load i32, ptr %data, align 4
  store i32 %1, ptr %ww, align 4
  br label %if.end

if.else:                                          ; preds = %invoke.cont3
  %2 = bitcast ptr %b2Bar to ptr
  %data4 = getelementptr inbounds %class.B1, ptr %2, i32 0, i32 0
  %3 = load i32, ptr %data4, align 4
  store i32 %3, ptr %ww, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %4 = load i32, ptr %ww, align 4
  %5 = load ptr, ptr @"?ptr@@3PEAHEA", align 8
  %6 = load i32, ptr %5, align 4
  %add = add nsw i32 %4, %6
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont5 unwind label %ehcleanup

invoke.cont5:                                     ; preds = %if.end
  call void @"??1B1@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %b1Bar) #8
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont6 unwind label %ehcleanup7

ehcleanup:                                        ; preds = %if.end, %invoke.cont
  %7 = cleanuppad within none []
  call void @"??1B1@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %b1Bar) #8 [ "funclet"(token %7) ]
  cleanupret from %7 unwind label %ehcleanup7

invoke.cont6:                                     ; preds = %invoke.cont5
  call void @"??1B2@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %b2Bar) #8
  ret i32 %add

ehcleanup7:                                       ; preds = %invoke.cont5, %ehcleanup, %entry
  %8 = cleanuppad within none []
  call void @"??1B2@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %b2Bar) #8 [ "funclet"(token %8) ]
  cleanupret from %8 unwind to caller
}

; Function Attrs: nounwind readnone
declare dso_local void @llvm.seh.scope.begin() #2

declare dso_local i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind readnone
declare dso_local void @llvm.seh.scope.end() #2

; Function Attrs: noinline nounwind optnone
define linkonce_odr dso_local void @"??1B1@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %this) unnamed_addr #3 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  call void (...) @"?printf@@YAXZZ"(ptr @"??_C@_0N@FMGAAAAM@in?5B1?5Dtor?5?6?$AA@")
  ret void
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr dso_local void @"??1B2@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %this) unnamed_addr #3 comdat align 2 personality ptr @__CxxFrameHandler3 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  invoke void (...) @"?printf@@YAXZZ"(ptr @"??_C@_0N@GFONDMMJ@in?5B2?5Dtor?5?6?$AA@")
          to label %invoke.cont2 unwind label %ehcleanup

invoke.cont2:                                     ; preds = %invoke.cont
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont3 unwind label %ehcleanup

invoke.cont3:                                     ; preds = %invoke.cont2
  %0 = bitcast ptr %this1 to ptr
  call void @"??1B1@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %0) #8
  ret void

ehcleanup:                                        ; preds = %invoke.cont2, %invoke.cont, %entry
  %1 = cleanuppad within none []
  %2 = bitcast ptr %this1 to ptr
  call void @"??1B1@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %2) #8 [ "funclet"(token %1) ]
  cleanupret from %1 unwind to caller
}

; Function Attrs: noinline optnone mustprogress
define dso_local void @"?goo@@YA?AVB1@@H@Z"(ptr noalias sret(%class.B1) align 4 %agg.result, i32 %w) #1 personality ptr @__CxxFrameHandler3 {
entry:
  %result.ptr = alloca ptr, align 8
  %w.addr = alloca i32, align 4
  %b2ingoo = alloca %class.B2, align 4
  %0 = bitcast ptr %agg.result to ptr
  store ptr %0, ptr %result.ptr, align 8
  store i32 %w, ptr %w.addr, align 4
  %call = call ptr @"??0B2@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %b2ingoo)
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %1 = load i32, ptr %w.addr, align 4
  %2 = bitcast ptr %b2ingoo to ptr
  %data = getelementptr inbounds %class.B1, ptr %2, i32 0, i32 0
  %3 = load i32, ptr %data, align 4
  %add = add nsw i32 %3, %1
  store i32 %add, ptr %data, align 4
  %4 = bitcast ptr %b2ingoo to ptr
  %5 = bitcast ptr %agg.result to ptr
  %6 = bitcast ptr %4 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %5, ptr align 4 %6, i64 4, i1 false)
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont1 unwind label %ehcleanup

invoke.cont1:                                     ; preds = %invoke.cont
  call void @"??1B2@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %b2ingoo) #8
  ret void

ehcleanup:                                        ; preds = %invoke.cont, %entry
  %7 = cleanuppad within none []
  call void @"??1B2@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %b2ingoo) #8 [ "funclet"(token %7) ]
  cleanupret from %7 unwind to caller
}

; Function Attrs: noinline optnone
define linkonce_odr dso_local ptr @"??0B2@@QEAA@XZ"(ptr nonnull returned align 4 dereferenceable(4) %this) unnamed_addr #4 comdat align 2 personality ptr @__CxxFrameHandler3 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  %0 = bitcast ptr %this1 to ptr
  %call = call ptr @"??0B1@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %0)
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %1 = bitcast ptr %this1 to ptr
  %data = getelementptr inbounds %class.B1, ptr %1, i32 0, i32 0
  %2 = load i32, ptr %data, align 4
  %add = add nsw i32 %2, 222
  %call2 = call i32 @"?foo@@YAHH@Z"(i32 %add)
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont3 unwind label %ehcleanup

invoke.cont3:                                     ; preds = %invoke.cont
  ret ptr %this1

ehcleanup:                                        ; preds = %invoke.cont, %entry
  %3 = cleanuppad within none []
  %4 = bitcast ptr %this1 to ptr
  call void @"??1B1@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %4) #8 [ "funclet"(token %3) ]
  cleanupret from %3 unwind to caller
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #5

; Function Attrs: noinline norecurse optnone mustprogress
define dso_local i32 @main() #6 personality ptr @__CxxFrameHandler3 {
entry:
  %retval = alloca i32, align 4
  %b3inmain = alloca %class.B3, align 4
  %m = alloca i32, align 4
  %ref.tmp = alloca %class.B2, align 4
  %cleanup.cond = alloca i1, align 1
  %ref.tmp5 = alloca %class.B3, align 4
  %cleanup.cond9 = alloca i1, align 1
  %i = alloca i32, align 4
  %agg.tmp = alloca %class.B2, align 4
  %agg.tmp28 = alloca %class.B1, align 4
  %b1fromgoo = alloca %class.B1, align 4
  store i32 0, ptr %retval, align 4
  %call = call ptr @"??0B3@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %b3inmain)
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont unwind label %ehcleanup50

invoke.cont:                                      ; preds = %entry
  %0 = load i32, ptr @"?xxxx@@3HA", align 4
  %cmp = icmp sgt i32 %0, 1
  store i1 false, ptr %cleanup.cond, align 1
  store i1 false, ptr %cleanup.cond9, align 1
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %invoke.cont
  %call2 = invoke ptr @"??0B2@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %ref.tmp)
          to label %invoke.cont1 unwind label %ehcleanup50

invoke.cont1:                                     ; preds = %cond.true
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont3 unwind label %ehcleanup21

invoke.cont3:                                     ; preds = %invoke.cont1
  store i1 true, ptr %cleanup.cond, align 1
  %1 = bitcast ptr %ref.tmp to ptr
  %data = getelementptr inbounds %class.B1, ptr %1, i32 0, i32 0
  %2 = load i32, ptr %data, align 4
  %call4 = call i32 @"?foo@@YAHH@Z"(i32 99)
  %add = add nsw i32 %2, %call4
  br label %cond.end

cond.false:                                       ; preds = %invoke.cont
  %call7 = invoke ptr @"??0B3@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %ref.tmp5)
          to label %invoke.cont6 unwind label %ehcleanup21

invoke.cont6:                                     ; preds = %cond.false
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont8 unwind label %ehcleanup

invoke.cont8:                                     ; preds = %invoke.cont6
  store i1 true, ptr %cleanup.cond9, align 1
  %3 = bitcast ptr %ref.tmp5 to ptr
  %data10 = getelementptr inbounds %class.B1, ptr %3, i32 0, i32 0
  %4 = load i32, ptr %data10, align 4
  %call11 = call i32 @"?foo@@YAHH@Z"(i32 88)
  %add12 = add nsw i32 %4, %call11
  br label %cond.end

cond.end:                                         ; preds = %invoke.cont8, %invoke.cont3
  %cond = phi i32 [ %add, %invoke.cont3 ], [ %add12, %invoke.cont8 ]
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont13 unwind label %ehcleanup

invoke.cont13:                                    ; preds = %cond.end
  %cleanup.is_active = load i1, ptr %cleanup.cond9, align 1
  br i1 %cleanup.is_active, label %cleanup.action, label %cleanup.done

cleanup.action:                                   ; preds = %invoke.cont13
  call void @"??1B3@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %ref.tmp5) #8
  br label %cleanup.done

cleanup.done:                                     ; preds = %cleanup.action, %invoke.cont13
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont17 unwind label %ehcleanup21

invoke.cont17:                                    ; preds = %cleanup.done
  %cleanup.is_active18 = load i1, ptr %cleanup.cond, align 1
  br i1 %cleanup.is_active18, label %cleanup.action19, label %cleanup.done20

cleanup.action19:                                 ; preds = %invoke.cont17
  call void @"??1B2@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %ref.tmp) #8
  br label %cleanup.done20

cleanup.done20:                                   ; preds = %cleanup.action19, %invoke.cont17
  store i32 %cond, ptr %m, align 4
  %call26 = invoke ptr @"??0B2@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %agg.tmp)
          to label %invoke.cont25 unwind label %ehcleanup50

invoke.cont25:                                    ; preds = %cleanup.done20
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont27 unwind label %ehcleanup38

invoke.cont27:                                    ; preds = %invoke.cont25
  %call30 = invoke ptr @"??0B1@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %agg.tmp28)
          to label %invoke.cont29 unwind label %ehcleanup38

invoke.cont29:                                    ; preds = %invoke.cont27
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont31 unwind label %ehcleanup36

invoke.cont31:                                    ; preds = %invoke.cont29
  %call32 = call i32 @"?foo@@YAHH@Z"(i32 0)
  %coerce.dive = getelementptr inbounds %class.B1, ptr %agg.tmp28, i32 0, i32 0
  %5 = load i32, ptr %coerce.dive, align 4
  %coerce.dive33 = getelementptr inbounds %class.B2, ptr %agg.tmp, i32 0, i32 0
  %coerce.dive34 = getelementptr inbounds %class.B1, ptr %coerce.dive33, i32 0, i32 0
  %6 = load i32, ptr %coerce.dive34, align 4
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont35 unwind label %ehcleanup36

invoke.cont35:                                    ; preds = %invoke.cont31
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont37 unwind label %ehcleanup38

invoke.cont37:                                    ; preds = %invoke.cont35
  %call40 = invoke i32 @"?bar@@YAHHVB1@@VB2@@@Z"(i32 %call32, i32 %5, i32 %6)
          to label %invoke.cont39 unwind label %ehcleanup50

invoke.cont39:                                    ; preds = %invoke.cont37
  store i32 %call40, ptr %i, align 4
  %7 = load i32, ptr %i, align 4
  invoke void @"?goo@@YA?AVB1@@H@Z"(ptr sret(%class.B1) align 4 %b1fromgoo, i32 %7)
          to label %invoke.cont41 unwind label %ehcleanup50

invoke.cont41:                                    ; preds = %invoke.cont39
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont42 unwind label %ehcleanup48

invoke.cont42:                                    ; preds = %invoke.cont41
  %8 = load i32, ptr %m, align 4
  %data43 = getelementptr inbounds %class.B1, ptr %b1fromgoo, i32 0, i32 0
  %9 = load i32, ptr %data43, align 4
  %add44 = add nsw i32 %8, %9
  %10 = bitcast ptr %b3inmain to ptr
  %data45 = getelementptr inbounds %class.B1, ptr %10, i32 0, i32 0
  %11 = load i32, ptr %data45, align 4
  %add46 = add nsw i32 %add44, %11
  store i32 %add46, ptr %retval, align 4
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont47 unwind label %ehcleanup48

ehcleanup:                                        ; preds = %cond.end, %invoke.cont6
  %12 = cleanuppad within none []
  %cleanup.is_active14 = load i1, ptr %cleanup.cond9, align 1
  br i1 %cleanup.is_active14, label %cleanup.action15, label %cleanup.done16

cleanup.action15:                                 ; preds = %ehcleanup
  call void @"??1B3@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %ref.tmp5) #8 [ "funclet"(token %12) ]
  br label %cleanup.done16

cleanup.done16:                                   ; preds = %cleanup.action15, %ehcleanup
  cleanupret from %12 unwind label %ehcleanup21

ehcleanup21:                                      ; preds = %cleanup.done, %cleanup.done16, %cond.false, %invoke.cont1
  %13 = cleanuppad within none []
  %cleanup.is_active22 = load i1, ptr %cleanup.cond, align 1
  br i1 %cleanup.is_active22, label %cleanup.action23, label %cleanup.done24

cleanup.action23:                                 ; preds = %ehcleanup21
  call void @"??1B2@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %ref.tmp) #8 [ "funclet"(token %13) ]
  br label %cleanup.done24

cleanup.done24:                                   ; preds = %cleanup.action23, %ehcleanup21
  cleanupret from %13 unwind label %ehcleanup50

ehcleanup36:                                      ; preds = %invoke.cont31, %invoke.cont29
  %14 = cleanuppad within none []
  call void @"??1B1@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %agg.tmp28) #8 [ "funclet"(token %14) ]
  cleanupret from %14 unwind label %ehcleanup38

ehcleanup38:                                      ; preds = %invoke.cont35, %ehcleanup36, %invoke.cont27, %invoke.cont25
  %15 = cleanuppad within none []
  call void @"??1B2@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %agg.tmp) #8 [ "funclet"(token %15) ]
  cleanupret from %15 unwind label %ehcleanup50

invoke.cont47:                                    ; preds = %invoke.cont42
  call void @"??1B1@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %b1fromgoo) #8
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont49 unwind label %ehcleanup50

ehcleanup48:                                      ; preds = %invoke.cont42, %invoke.cont41
  %16 = cleanuppad within none []
  call void @"??1B1@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %b1fromgoo) #8 [ "funclet"(token %16) ]
  cleanupret from %16 unwind label %ehcleanup50

invoke.cont49:                                    ; preds = %invoke.cont47
  call void @"??1B3@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %b3inmain) #8
  %17 = load i32, ptr %retval, align 4
  ret i32 %17

ehcleanup50:                                      ; preds = %invoke.cont47, %ehcleanup48, %invoke.cont39, %invoke.cont37, %ehcleanup38, %cleanup.done20, %cleanup.done24, %cond.true, %entry
  %18 = cleanuppad within none []
  call void @"??1B3@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %b3inmain) #8 [ "funclet"(token %18) ]
  cleanupret from %18 unwind to caller
}

; Function Attrs: noinline optnone
define linkonce_odr dso_local ptr @"??0B3@@QEAA@XZ"(ptr nonnull returned align 4 dereferenceable(4) %this) unnamed_addr #4 comdat align 2 personality ptr @__CxxFrameHandler3 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  %0 = bitcast ptr %this1 to ptr
  %call = call ptr @"??0B2@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %0)
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %1 = bitcast ptr %this1 to ptr
  %data = getelementptr inbounds %class.B1, ptr %1, i32 0, i32 0
  %2 = load i32, ptr %data, align 4
  %add = add nsw i32 %2, 333
  %call2 = call i32 @"?foo@@YAHH@Z"(i32 %add)
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont3 unwind label %ehcleanup

invoke.cont3:                                     ; preds = %invoke.cont
  ret ptr %this1

ehcleanup:                                        ; preds = %invoke.cont, %entry
  %3 = cleanuppad within none []
  %4 = bitcast ptr %this1 to ptr
  call void @"??1B2@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %4) #8 [ "funclet"(token %3) ]
  cleanupret from %3 unwind to caller
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr dso_local void @"??1B3@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %this) unnamed_addr #3 comdat align 2 personality ptr @__CxxFrameHandler3 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  invoke void (...) @"?printf@@YAXZZ"(ptr @"??_C@_0N@HCJGCIIK@in?5B3?5Dtor?5?6?$AA@")
          to label %invoke.cont2 unwind label %ehcleanup

invoke.cont2:                                     ; preds = %invoke.cont
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont3 unwind label %ehcleanup

invoke.cont3:                                     ; preds = %invoke.cont2
  %0 = bitcast ptr %this1 to ptr
  call void @"??1B2@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %0) #8
  ret void

ehcleanup:                                        ; preds = %invoke.cont2, %invoke.cont, %entry
  %1 = cleanuppad within none []
  %2 = bitcast ptr %this1 to ptr
  call void @"??1B2@@QEAA@XZ"(ptr nonnull align 4 dereferenceable(4) %2) #8 [ "funclet"(token %1) ]
  cleanupret from %1 unwind to caller
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr dso_local ptr @"??0B1@@QEAA@XZ"(ptr nonnull returned align 4 dereferenceable(4) %this) unnamed_addr #3 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  %data = getelementptr inbounds %class.B1, ptr %this1, i32 0, i32 0
  store i32 90, ptr %data, align 4
  %data2 = getelementptr inbounds %class.B1, ptr %this1, i32 0, i32 0
  %0 = load i32, ptr %data2, align 4
  %add = add nsw i32 %0, 111
  %call = call i32 @"?foo@@YAHH@Z"(i32 %add)
  ret ptr %this1
}

declare dso_local void @"?printf@@YAXZZ"(...) #7

attributes #0 = { noinline nounwind optnone mustprogress "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { noinline optnone mustprogress "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #2 = { nounwind readnone }
attributes #3 = { noinline nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #4 = { noinline optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #5 = { argmemonly nofree nosync nounwind willreturn }
attributes #6 = { noinline norecurse optnone mustprogress "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #7 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #8 = { nounwind }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 2, !"eh-asynch", i32 1}
