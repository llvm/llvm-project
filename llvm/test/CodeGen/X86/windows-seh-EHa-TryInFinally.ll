; RUN: llc -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: "?fin$0@0@main@@"
; CHECK:      .seh_handlerdata
; CHECK:      .set ".L?fin$0@0@main@@$parent_frame_offset", 48
; CHECK-NEXT:        .long   (.Llsda_end1-.Llsda_begin1)/16 
; CHECK-NEXT: .Llsda_begin1:
; CHECK-NEXT:        .long   .Ltmp
; CHECK-NEXT:        .long   .Ltmp
; CHECK-NEXT:        .long   "?dtor$
; CHECK-NEXT:        .long   0
; CHECK-NEXT: .Llsda_end1:

; ModuleID = 'windows-seh-EHa-TryInFinally.cpp'
source_filename = "windows-seh-EHa-TryInFinally.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc"

$"??_C@_0CI@MDFPIOJJ@?5?9?9?9?5Test?5_Try?5in?5_finally?5?9?9?9?5i@" = comdat any

$"??_C@_0BN@HHKJHLBE@?5?5In?5Inner?5_finally?5i?5?$DN?5?$CFd?5?6?$AA@" = comdat any

$"??_C@_0BN@HAIIIOKI@?5?5In?5outer?5_finally?5i?5?$DN?5?$CFd?5?6?$AA@" = comdat any

$"??_C@_0BJ@OJMMAGCD@?5?5In?5outer?5_try?5i?5?$DN?5?$CFd?5?6?$AA@" = comdat any

$"??_C@_0CG@ENDJHCGA@?5?9?9?9?5In?5outer?5except?5handler?5i?5?$DN@" = comdat any

@"??_C@_0CI@MDFPIOJJ@?5?9?9?9?5Test?5_Try?5in?5_finally?5?9?9?9?5i@" = linkonce_odr dso_local unnamed_addr constant [40 x i8] c" --- Test _Try in _finally --- i = %d \0A\00", comdat, align 1
@"??_C@_0BN@HHKJHLBE@?5?5In?5Inner?5_finally?5i?5?$DN?5?$CFd?5?6?$AA@" = linkonce_odr dso_local unnamed_addr constant [29 x i8] c"  In Inner _finally i = %d \0A\00", comdat, align 1
@"??_C@_0BN@HAIIIOKI@?5?5In?5outer?5_finally?5i?5?$DN?5?$CFd?5?6?$AA@" = linkonce_odr dso_local unnamed_addr constant [29 x i8] c"  In outer _finally i = %d \0A\00", comdat, align 1
@"??_C@_0BJ@OJMMAGCD@?5?5In?5outer?5_try?5i?5?$DN?5?$CFd?5?6?$AA@" = linkonce_odr dso_local unnamed_addr constant [25 x i8] c"  In outer _try i = %d \0A\00", comdat, align 1
@"??_C@_0CG@ENDJHCGA@?5?9?9?9?5In?5outer?5except?5handler?5i?5?$DN@" = linkonce_odr dso_local unnamed_addr constant [38 x i8] c" --- In outer except handler i = %d \0A\00", comdat, align 1

; Function Attrs: noinline norecurse optnone
define dso_local i32 @main() #0 personality ptr @__C_specific_handler {
entry:
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  %__exception_code = alloca i32, align 4
  call void (...) @llvm.localescape(ptr %i)
  store i32 0, ptr %retval, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 3
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, ptr %i, align 4
  call void (...) @"?printf@@YAXZZ"(ptr @"??_C@_0CI@MDFPIOJJ@?5?9?9?9?5Test?5_Try?5in?5_finally?5?9?9?9?5i@", i32 %1)
  invoke void @llvm.seh.try.begin()
          to label %invoke.cont unwind label %catch.dispatch

invoke.cont:                                      ; preds = %for.body
  invoke void @llvm.seh.try.begin()
          to label %invoke.cont1 unwind label %ehcleanup

invoke.cont1:                                     ; preds = %invoke.cont
  %2 = load volatile i32, ptr %i, align 4
  invoke void (...) @"?printf@@YAXZZ"(ptr @"??_C@_0BJ@OJMMAGCD@?5?5In?5outer?5_try?5i?5?$DN?5?$CFd?5?6?$AA@", i32 %2) #6
          to label %invoke.cont2 unwind label %ehcleanup

invoke.cont2:                                     ; preds = %invoke.cont1
  %3 = load volatile i32, ptr %i, align 4
  %cmp3 = icmp eq i32 %3, 0
  br i1 %cmp3, label %if.then, label %if.end

if.then:                                          ; preds = %invoke.cont2
  store volatile i32 0, ptr inttoptr (i64 17 to ptr), align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %invoke.cont2
  invoke void @llvm.seh.try.end()
          to label %invoke.cont4 unwind label %ehcleanup

invoke.cont4:                                     ; preds = %if.end
  %4 = call ptr @llvm.localaddress()
  invoke void @"?fin$0@0@main@@"(i8 0, ptr %4) #6
          to label %invoke.cont5 unwind label %catch.dispatch

invoke.cont5:                                     ; preds = %invoke.cont4
  invoke void @llvm.seh.try.end()
          to label %invoke.cont7 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %invoke.cont5, %invoke.cont6, %ehcleanup, %invoke.cont4, %for.body
  %5 = catchswitch within none [label %__except] unwind to caller

__except:                                         ; preds = %catch.dispatch
  %6 = catchpad within %5 [ptr null]
  catchret from %6 to label %__except8

__except8:                                        ; preds = %__except
  %7 = call i32 @llvm.eh.exceptioncode(token %6)
  store i32 %7, ptr %__exception_code, align 4
  %8 = load i32, ptr %i, align 4
  call void (...) @"?printf@@YAXZZ"(ptr @"??_C@_0CG@ENDJHCGA@?5?9?9?9?5In?5outer?5except?5handler?5i?5?$DN@", i32 %8)
  br label %__try.cont

__try.cont:                                       ; preds = %__except8, %invoke.cont7
  br label %for.inc

for.inc:                                          ; preds = %__try.cont
  %9 = load i32, ptr %i, align 4
  %inc = add nsw i32 %9, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

invoke.cont7:                                     ; preds = %invoke.cont5
  br label %__try.cont

ehcleanup:                                        ; preds = %if.end, %invoke.cont1, %invoke.cont
  %10 = cleanuppad within none []
  %11 = call ptr @llvm.localaddress()
  invoke void @"?fin$0@0@main@@"(i8 1, ptr %11) #6 [ "funclet"(token %10) ]
          to label %invoke.cont6 unwind label %catch.dispatch

invoke.cont6:                                     ; preds = %ehcleanup
  cleanupret from %10 unwind label %catch.dispatch

for.end:                                          ; preds = %for.cond
  ret i32 0
}

declare dso_local void @"?printf@@YAXZZ"(...) #1

; Function Attrs: nounwind willreturn
declare dso_local void @llvm.seh.try.begin() #2

declare dso_local i32 @__C_specific_handler(...)

; Function Attrs: noinline
define internal void @"?fin$0@0@main@@"(i8 %abnormal_termination, ptr %frame_pointer) #3 personality ptr @__C_specific_handler {
entry:
  %frame_pointer.addr = alloca ptr, align 8
  %abnormal_termination.addr = alloca i8, align 1
  %0 = call ptr @llvm.localrecover(ptr @main, ptr %frame_pointer, i32 0)
  %i = bitcast ptr %0 to ptr
  store ptr %frame_pointer, ptr %frame_pointer.addr, align 8
  store i8 %abnormal_termination, ptr %abnormal_termination.addr, align 1
  invoke void @llvm.seh.try.begin()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %1 = load volatile i32, ptr %i, align 4
  invoke void (...) @"?printf@@YAXZZ"(ptr @"??_C@_0BN@HAIIIOKI@?5?5In?5outer?5_finally?5i?5?$DN?5?$CFd?5?6?$AA@", i32 %1) #6
          to label %invoke.cont1 unwind label %ehcleanup

invoke.cont1:                                     ; preds = %invoke.cont
  %2 = load volatile i32, ptr %i, align 4
  %cmp = icmp eq i32 %2, 1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %invoke.cont1
  store volatile i32 0, ptr inttoptr (i64 17 to ptr), align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %invoke.cont1
  invoke void @llvm.seh.try.end()
          to label %invoke.cont2 unwind label %ehcleanup

invoke.cont2:                                     ; preds = %if.end
  call void @"?fin$1@0@main@@"(i8 0, ptr %frame_pointer)
  ret void

ehcleanup:                                        ; preds = %if.end, %invoke.cont, %entry
  %3 = cleanuppad within none []
  call void @"?fin$1@0@main@@"(i8 1, ptr %frame_pointer) [ "funclet"(token %3) ]
  cleanupret from %3 unwind to caller
}

; Function Attrs: nounwind readnone
declare ptr @llvm.localrecover(ptr, ptr, i32 immarg) #4

; Function Attrs: noinline
define internal void @"?fin$1@0@main@@"(i8 %abnormal_termination, ptr %frame_pointer) #3 {
entry:
  %frame_pointer.addr = alloca ptr, align 8
  %abnormal_termination.addr = alloca i8, align 1
  %0 = call ptr @llvm.localrecover(ptr @main, ptr %frame_pointer, i32 0)
  %i = bitcast ptr %0 to ptr
  store ptr %frame_pointer, ptr %frame_pointer.addr, align 8
  store i8 %abnormal_termination, ptr %abnormal_termination.addr, align 1
  %1 = load i32, ptr %i, align 4
  call void (...) @"?printf@@YAXZZ"(ptr @"??_C@_0BN@HHKJHLBE@?5?5In?5Inner?5_finally?5i?5?$DN?5?$CFd?5?6?$AA@", i32 %1)
  %2 = load i32, ptr %i, align 4
  %cmp = icmp eq i32 %2, 2
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store volatile i32 0, ptr inttoptr (i64 17 to ptr), align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: nounwind willreturn
declare dso_local void @llvm.seh.try.end() #2

; Function Attrs: nounwind readnone
declare ptr @llvm.localaddress() #4

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.exceptioncode(token) #4

; Function Attrs: nounwind
declare void @llvm.localescape(...) #5

attributes #0 = { noinline norecurse optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind willreturn }
attributes #3 = { noinline "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind readnone }
attributes #5 = { nounwind }
attributes #6 = { noinline }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 2, !"eh-asynch", i32 1}

