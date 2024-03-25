; RUN: llc -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: "$cppxdata$?crash@@YAXH@Z":
; CHECK:	.long	("$stateUnwindMap$?crash@@YAXH@Z")
; CHECK:	.long	("$ip2state$?crash@@YAXH@Z")

; CHECK-LABEL: "$stateUnwindMap$?crash@@YAXH@Z":
; CHECK:	.long	-1 
; CHECK:	.long	"?dtor$
; CHECK:	.long	0 
; CHECK:	.long	"?dtor$
; CHECK:	.long	1
; CHECK:	.long	"?dtor$

; CHECK-LABEL: "$ip2state$?crash@@YAXH@Z":
; CHECK-NEXT:	.long	.Lfunc_begin0@IMGREL
; CHECK-NEXT:	.long	-1                  
; CHECK-NEXT:	.long	.Ltmp     
; CHECK-NEXT:	.long	0                   
; CHECK-NEXT:	.long	.Ltmp     
; CHECK-NEXT:	.long	1                   
; CHECK-NEXT:	.long	.Ltmp
; CHECK-NEXT:	.long	2                   
; CHECK-NEXT:	.long	.Ltmp
; CHECK-NEXT:	.long	1                   
; CHECK-NEXT:	.long	.Ltmp
; CHECK-NEXT:	.long	0                   
; CHECK-NEXT:	.long	.Ltmp
; CHECK-NEXT:	.long	-1                  

; ModuleID = 'windows-seh-EHa-CppDtors01.cpp'
source_filename = "windows-seh-EHa-CppDtors01.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc"

%struct.A = type { i8 }
%struct.B = type { i8 }
%struct.C = type { i8 }

$"??_C@_0CM@KAOHJHDK@?5Test?5CPP?5unwind?3?5in?5catch?5handl@" = comdat any

$"??_C@_0N@FCCEEGKL@?5in?5C?5dtor?5?6?$AA@" = comdat any

$"??_C@_0N@EFFPFCOI@?5in?5B?5dtor?5?6?$AA@" = comdat any

$"??_C@_0N@HMNCGOCN@?5in?5A?5dtor?5?6?$AA@" = comdat any

@"?g@@3HA" = dso_local global i32 0, align 4
@"??_C@_0CM@KAOHJHDK@?5Test?5CPP?5unwind?3?5in?5catch?5handl@" = linkonce_odr dso_local unnamed_addr constant [44 x i8] c" Test CPP unwind: in catch handler i = %d \0A\00", comdat, align 1
@"??_C@_0N@FCCEEGKL@?5in?5C?5dtor?5?6?$AA@" = linkonce_odr dso_local unnamed_addr constant [13 x i8] c" in C dtor \0A\00", comdat, align 1
@"??_C@_0N@EFFPFCOI@?5in?5B?5dtor?5?6?$AA@" = linkonce_odr dso_local unnamed_addr constant [13 x i8] c" in B dtor \0A\00", comdat, align 1
@"??_C@_0N@HMNCGOCN@?5in?5A?5dtor?5?6?$AA@" = linkonce_odr dso_local unnamed_addr constant [13 x i8] c" in A dtor \0A\00", comdat, align 1

; Function Attrs: noinline optnone
define dso_local void @"?crash@@YAXH@Z"(i32 %i) #0 personality ptr @__CxxFrameHandler3 {
entry:
  %i.addr = alloca i32, align 4
  %ObjA = alloca %struct.A, align 1
  %ObjB = alloca %struct.B, align 1
  %ObjC = alloca %struct.C, align 1
  store i32 %i, ptr %i.addr, align 4
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont unwind label %ehcleanup13

invoke.cont:                                      ; preds = %entry
  %0 = load i32, ptr %i.addr, align 4
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %invoke.cont
  store volatile i32 0, ptr inttoptr (i64 17 to ptr), align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %invoke.cont
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont1 unwind label %ehcleanup11

invoke.cont1:                                     ; preds = %if.end
  %1 = load i32, ptr %i.addr, align 4
  %cmp2 = icmp eq i32 %1, 1
  br i1 %cmp2, label %if.then3, label %if.end4

if.then3:                                         ; preds = %invoke.cont1
  store volatile i32 0, ptr inttoptr (i64 17 to ptr), align 4
  br label %if.end4

if.end4:                                          ; preds = %if.then3, %invoke.cont1
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont5 unwind label %ehcleanup

invoke.cont5:                                     ; preds = %if.end4
  %2 = load i32, ptr %i.addr, align 4
  %cmp6 = icmp eq i32 %2, 2
  br i1 %cmp6, label %if.then7, label %if.end8

if.then7:                                         ; preds = %invoke.cont5
  store volatile i32 0, ptr inttoptr (i64 17 to ptr), align 4
  br label %if.end8

if.end8:                                          ; preds = %if.then7, %invoke.cont5
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont9 unwind label %ehcleanup

invoke.cont9:                                     ; preds = %if.end8
  call void @"??1C@?1??crash@@YAXH@Z@QEAA@XZ"(ptr %ObjC) #6
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont10 unwind label %ehcleanup11

invoke.cont10:                                    ; preds = %invoke.cont9
  call void @"??1B@?1??crash@@YAXH@Z@QEAA@XZ"(ptr %ObjB) #6
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont12 unwind label %ehcleanup13

invoke.cont12:                                    ; preds = %invoke.cont10
  call void @"??1A@?1??crash@@YAXH@Z@QEAA@XZ"(ptr %ObjA) #6
  ret void

ehcleanup:                                        ; preds = %if.end8, %if.end4
  %3 = cleanuppad within none []
  call void @"??1C@?1??crash@@YAXH@Z@QEAA@XZ"(ptr %ObjC) #6 [ "funclet"(token %3) ]
  cleanupret from %3 unwind label %ehcleanup11

ehcleanup11:                                      ; preds = %invoke.cont9, %ehcleanup, %if.end
  %4 = cleanuppad within none []
  call void @"??1B@?1??crash@@YAXH@Z@QEAA@XZ"(ptr %ObjB) #6 [ "funclet"(token %4) ]
  cleanupret from %4 unwind label %ehcleanup13

ehcleanup13:                                      ; preds = %invoke.cont10, %ehcleanup11, %entry
  %5 = cleanuppad within none []
  call void @"??1A@?1??crash@@YAXH@Z@QEAA@XZ"(ptr %ObjA) #6 [ "funclet"(token %5) ]
  cleanupret from %5 unwind to caller
}

; Function Attrs: nounwind readnone
declare dso_local void @llvm.seh.scope.begin() #1

declare dso_local i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind readnone
declare dso_local void @llvm.seh.scope.end() #1

; Function Attrs: noinline nounwind optnone
define internal void @"??1C@?1??crash@@YAXH@Z@QEAA@XZ"(ptr %this) unnamed_addr #2 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  call void (...) @"?printf@@YAXZZ"(ptr @"??_C@_0N@FCCEEGKL@?5in?5C?5dtor?5?6?$AA@")
  ret void
}

; Function Attrs: noinline nounwind optnone
define internal void @"??1B@?1??crash@@YAXH@Z@QEAA@XZ"(ptr %this) unnamed_addr #2 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  call void (...) @"?printf@@YAXZZ"(ptr @"??_C@_0N@EFFPFCOI@?5in?5B?5dtor?5?6?$AA@")
  ret void
}

; Function Attrs: noinline nounwind optnone
define internal void @"??1A@?1??crash@@YAXH@Z@QEAA@XZ"(ptr %this) unnamed_addr #2 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  call void (...) @"?printf@@YAXZZ"(ptr @"??_C@_0N@HMNCGOCN@?5in?5A?5dtor?5?6?$AA@")
  ret void
}

; Function Attrs: noinline norecurse optnone
define dso_local i32 @main() #3 personality ptr @__C_specific_handler {
entry:
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  %__exception_code = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 3
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  invoke void @llvm.seh.try.begin()
          to label %invoke.cont unwind label %catch.dispatch

invoke.cont:                                      ; preds = %for.body
  %1 = load volatile i32, ptr %i, align 4
  invoke void @"?crash@@YAXH@Z"(i32 %1) #7
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
  call void (...) @"?printf@@YAXZZ"(ptr @"??_C@_0CM@KAOHJHDK@?5Test?5CPP?5unwind?3?5in?5catch?5handl@", i32 %5)
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

; Function Attrs: nounwind willreturn
declare dso_local void @llvm.seh.try.begin() #4

declare dso_local i32 @__C_specific_handler(...)

; Function Attrs: nounwind willreturn
declare dso_local void @llvm.seh.try.end() #4

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.exceptioncode(token) #1

declare dso_local void @"?printf@@YAXZZ"(...) #5

attributes #0 = { noinline optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noinline norecurse optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind willreturn }
attributes #5 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind }
attributes #7 = { noinline }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 2, !"eh-asynch", i32 1}
