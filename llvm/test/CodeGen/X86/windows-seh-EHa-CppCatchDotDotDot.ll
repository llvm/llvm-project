; RUN: llc -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: "$cppxdata$?crash@@YAXH@Z":
; CHECK:	.long	("$stateUnwindMap$?crash@@YAXH@Z")
; CHECK:        .long   ("$tryMap$?crash@@YAXH@Z")@IMGREL # TryBlockMap
; CHECK-NEXT:   .long   6                       # IPMapEntries
; CHECK-NEXT:	.long	("$ip2state$?crash@@YAXH@Z")

; CHECK-LABEL: "$stateUnwindMap$?crash@@YAXH@Z":
; CHECK-NEXT:        .long   -1
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   "?dtor$
; CHECK-NEXT:        .long   -1
; CHECK-NEXT:        .long   0

; CHECK-LABEL: "$tryMap$?crash@@YAXH@Z":
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   ("$handlerMap$

; CHECK:       "$handlerMap$0$?crash@@YAXH@Z"
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   "?catch$

; CHECK-LABEL: "$ip2state$?crash@@YAXH@Z":
; CHECK-NEXT:	.long	.Lfunc_begin0@IMGREL
; CHECK-NEXT:	.long	-1
; CHECK-NEXT:	.long	.Ltmp
; CHECK-NEXT:	.long	0
; CHECK-NEXT:	.long	.Ltmp
; CHECK-NEXT:	.long	1
; CHECK-NEXT:	.long	.Ltmp
; CHECK-NEXT:	.long	0
; CHECK-NEXT:	.long	.Ltmp
; CHECK-NEXT:	.long	-1
; CHECK-NEXT:	.long	"?catch$
; CHECK-NEXT:	.long	2

; ModuleID = 'windows-seh-EHa-CppCatchDotDotDot.cpp'
source_filename = "windows-seh-EHa-CppCatchDotDotDot.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc"

%rtti.TypeDescriptor2 = type { ptr, ptr, [3 x i8] }
%eh.CatchableType = type { i32, i32, i32, i32, i32, i32, i32 }
%eh.CatchableTypeArray.1 = type { i32, [1 x i32] }
%eh.ThrowInfo = type { i32, i32, i32, i32 }
%struct.A = type { i8 }

$"??_C@_0BJ@EIKFKKLB@?5in?5catch?$CI?4?4?4?$CJ?5funclet?5?6?$AA@" = comdat any

$"??_R0H@8" = comdat any

$"_CT??_R0H@84" = comdat any

$_CTA1H = comdat any

$_TI1H = comdat any

$"??_C@_0CN@MKCAOFNA@?5Test?5CPP?5unwind?3?5in?5except?5hand@" = comdat any

$"??_C@_0N@LJHFFAKD@?5in?5A?5ctor?5?6?$AA@" = comdat any

$"??_C@_0N@HMNCGOCN@?5in?5A?5dtor?5?6?$AA@" = comdat any

@"?pt1@@3PEAHEA" = dso_local global ptr null, align 8
@"?pt2@@3PEAHEA" = dso_local global ptr null, align 8
@"?pt3@@3PEAHEA" = dso_local global ptr null, align 8
@"?g@@3HA" = dso_local global i32 0, align 4
@"??_C@_0BJ@EIKFKKLB@?5in?5catch?$CI?4?4?4?$CJ?5funclet?5?6?$AA@" = linkonce_odr dso_local unnamed_addr constant [25 x i8] c" in catch(...) funclet \0A\00", comdat, align 1
@"??_7type_info@@6B@" = external constant ptr
@"??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { ptr @"??_7type_info@@6B@", ptr null, [3 x i8] c".H\00" }, comdat
@__ImageBase = external dso_local constant i8
@"_CT??_R0H@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0H@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 4, i32 0 }, section ".xdata", comdat
@_CTA1H = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0H@84" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@_TI1H = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @_CTA1H to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat
@"??_C@_0CN@MKCAOFNA@?5Test?5CPP?5unwind?3?5in?5except?5hand@" = linkonce_odr dso_local unnamed_addr constant [45 x i8] c" Test CPP unwind: in except handler i = %d \0A\00", comdat, align 1
@"??_C@_0N@LJHFFAKD@?5in?5A?5ctor?5?6?$AA@" = linkonce_odr dso_local unnamed_addr constant [13 x i8] c" in A ctor \0A\00", comdat, align 1
@"??_C@_0N@HMNCGOCN@?5in?5A?5dtor?5?6?$AA@" = linkonce_odr dso_local unnamed_addr constant [13 x i8] c" in A dtor \0A\00", comdat, align 1

; Function Attrs: noinline nounwind optnone
define dso_local void @"?foo@@YAXXZ"() #0 {
entry:
  store volatile i32 0, ptr inttoptr (i64 17 to ptr), align 4
  ret void
}

; Function Attrs: noinline optnone
define dso_local void @"?crash@@YAXH@Z"(i32 %i) #1 personality ptr @__CxxFrameHandler3 {
entry:
  %i.addr = alloca i32, align 4
  %ObjA = alloca %struct.A, align 1
  %tmp = alloca i32, align 4
  store i32 %i, ptr %i.addr, align 4
  %0 = load i32, ptr %i.addr, align 4
  store i32 %0, ptr @"?g@@3HA", align 4
  invoke void @llvm.seh.try.begin()
          to label %invoke.cont unwind label %catch.dispatch

invoke.cont:                                      ; preds = %entry
  %call = invoke ptr @"??0A@?1??crash@@YAXH@Z@QEAA@XZ"(ptr %ObjA)
          to label %invoke.cont1 unwind label %catch.dispatch

invoke.cont1:                                     ; preds = %invoke.cont
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont2 unwind label %ehcleanup

invoke.cont2:                                     ; preds = %invoke.cont1
  %1 = load i32, ptr %i.addr, align 4
  %cmp = icmp eq i32 %1, 1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %invoke.cont2
  store volatile i32 0, ptr inttoptr (i64 17 to ptr), align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %invoke.cont2
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont3 unwind label %ehcleanup

invoke.cont3:                                     ; preds = %if.end
  call void @"??1A@?1??crash@@YAXH@Z@QEAA@XZ"(ptr %ObjA) #6
  br label %try.cont

ehcleanup:                                        ; preds = %if.end, %invoke.cont1
  %2 = cleanuppad within none []
  call void @"??1A@?1??crash@@YAXH@Z@QEAA@XZ"(ptr %ObjA) #6 [ "funclet"(token %2) ]
  cleanupret from %2 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %ehcleanup, %invoke.cont, %entry
  %3 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %4 = catchpad within %3 [ptr null, i32 0, ptr null]
  call void (...) @"?printf@@YAXZZ"(ptr @"??_C@_0BJ@EIKFKKLB@?5in?5catch?$CI?4?4?4?$CJ?5funclet?5?6?$AA@") [ "funclet"(token %4) ]
  %5 = load i32, ptr %i.addr, align 4
  %cmp4 = icmp eq i32 %5, 1
  br i1 %cmp4, label %if.then5, label %if.end6

if.then5:                                         ; preds = %catch
  %6 = load i32, ptr %i.addr, align 4
  store i32 %6, ptr %tmp, align 4
  %7 = bitcast ptr %tmp to ptr
  call void @_CxxThrowException(ptr %7, ptr @_TI1H) #7 [ "funclet"(token %4) ]
  unreachable

if.end6:                                          ; preds = %catch
  catchret from %4 to label %catchret.dest

catchret.dest:                                    ; preds = %if.end6
  br label %try.cont

try.cont:                                         ; preds = %catchret.dest, %invoke.cont3
  ret void
}

; Function Attrs: nounwind willreturn
declare dso_local void @llvm.seh.try.begin() #2

declare dso_local i32 @__CxxFrameHandler3(...)

; Function Attrs: noinline optnone
define internal ptr @"??0A@?1??crash@@YAXH@Z@QEAA@XZ"(ptr returned %this) unnamed_addr #1 align 2 {
entry:
  %retval = alloca ptr, align 8
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  store ptr %this1, ptr %retval, align 8
  call void (...) @"?printf@@YAXZZ"(ptr @"??_C@_0N@LJHFFAKD@?5in?5A?5ctor?5?6?$AA@")
  %0 = load i32, ptr @"?g@@3HA", align 4
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store volatile i32 0, ptr inttoptr (i64 17 to ptr), align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %1 = load ptr, ptr %retval, align 8
  ret ptr %1
}

; Function Attrs: nounwind readnone
declare dso_local void @llvm.seh.scope.begin() #3

; Function Attrs: nounwind readnone
declare dso_local void @llvm.seh.scope.end() #3

; Function Attrs: noinline nounwind optnone
define internal void @"??1A@?1??crash@@YAXH@Z@QEAA@XZ"(ptr %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  call void (...) @"?printf@@YAXZZ"(ptr @"??_C@_0N@HMNCGOCN@?5in?5A?5dtor?5?6?$AA@")
  ret void
}

declare dso_local void @"?printf@@YAXZZ"(...) #4

declare dso_local void @_CxxThrowException(ptr, ptr)

; Function Attrs: noinline norecurse optnone
define dso_local i32 @main() #5 personality ptr @__C_specific_handler {
entry:
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  %__exception_code = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 2
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  invoke void @llvm.seh.try.begin()
          to label %invoke.cont unwind label %catch.dispatch

invoke.cont:                                      ; preds = %for.body
  %1 = load volatile i32, ptr %i, align 4
  invoke void @"?crash@@YAXH@Z"(i32 %1) #8
          to label %invoke.cont1 unwind label %catch.dispatch

invoke.cont1:                                     ; preds = %invoke.cont
  invoke void @llvm.seh.try.end()
          to label %invoke.cont2 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %invoke.cont1, %invoke.cont, %for.body
  %2 = catchswitch within none [label %__except] unwind to caller

__except:                                         ; preds = %catch.dispatch
  %3 = catchpad within %2 [ptr null]
  catchret from %3 to label %__except3

__except3:                                        ; preds = %__except
  %4 = call i32 @llvm.eh.exceptioncode(token %3)
  store i32 %4, ptr %__exception_code, align 4
  %5 = load i32, ptr %i, align 4
  call void (...) @"?printf@@YAXZZ"(ptr @"??_C@_0CN@MKCAOFNA@?5Test?5CPP?5unwind?3?5in?5except?5hand@", i32 %5)
  br label %__try.cont

__try.cont:                                       ; preds = %__except3, %invoke.cont2
  br label %for.inc

for.inc:                                          ; preds = %__try.cont
  %6 = load i32, ptr %i, align 4
  %inc = add nsw i32 %6, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

invoke.cont2:                                     ; preds = %invoke.cont1
  br label %__try.cont

for.end:                                          ; preds = %for.cond
  ret i32 0
}

declare dso_local i32 @__C_specific_handler(...)

; Function Attrs: nounwind willreturn
declare dso_local void @llvm.seh.try.end() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.exceptioncode(token) #3

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind willreturn }
attributes #3 = { nounwind readnone }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noinline norecurse optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind }
attributes #7 = { noreturn }
attributes #8 = { noinline }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 2, !"eh-asynch", i32 1}


