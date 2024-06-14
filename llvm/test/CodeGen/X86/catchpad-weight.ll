; RUN: llc -print-after=finalize-isel %s -o /dev/null 2>&1 | FileCheck %s

; Check if the edge weight to the catchpad is calculated correctly.

; CHECK: successors: %bb.2(0x7ffff100), %bb.1(0x00000800), %bb.3(0x00000400), %bb.4(0x00000200), %bb.5(0x00000100)

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--windows-msvc18.0.0"

%rtti.TypeDescriptor7 = type { ptr, ptr, [8 x i8] }
%struct.HasDtor = type { i8 }

$"\01??_R0?AUA@@@8" = comdat any

$"\01??_R0?AUB@@@8" = comdat any

$"\01??_R0?AUC@@@8" = comdat any

@"\01??_7type_info@@6B@" = external constant ptr
@"\01??_R0?AUA@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { ptr @"\01??_7type_info@@6B@", ptr null, [8 x i8] c".?AUA@@\00" }, comdat
@"\01??_R0?AUB@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { ptr @"\01??_7type_info@@6B@", ptr null, [8 x i8] c".?AUB@@\00" }, comdat
@"\01??_R0?AUC@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { ptr @"\01??_7type_info@@6B@", ptr null, [8 x i8] c".?AUC@@\00" }, comdat

; Function Attrs: uwtable
define i32 @main() #0 personality ptr @__CxxFrameHandler3 {
entry:
  %o = alloca %struct.HasDtor, align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr %o) #4
  invoke void @"\01?may_throw@@YAXXZ"()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %cs1 = catchswitch within none [label %catch.5] unwind label %catch.dispatch.1

catch.5:                                          ; preds = %catch.dispatch
  %0 = catchpad within %cs1 [ptr @"\01??_R0?AUA@@@8", i32 0, ptr null]
  catchret from %0 to label %try.cont

try.cont:                                         ; preds = %entry, %catch, %catch.3, %catch.5
  call void @"\01??1HasDtor@@QEAA@XZ"(ptr nonnull %o) #4
  call void @llvm.lifetime.end.p0(i64 1, ptr %o) #4
  ret i32 0

catch.dispatch.1:                                 ; preds = %catch.dispatch
  %cs2 = catchswitch within none [label %catch.3] unwind label %catch.dispatch.2

catch.3:                                          ; preds = %catch.dispatch.1
  %1 = catchpad within %cs2 [ptr @"\01??_R0?AUB@@@8", i32 0, ptr null]
  catchret from %1 to label %try.cont

catch.dispatch.2:                                 ; preds = %catch.dispatch.1
  %cs3 = catchswitch within none [label %catch] unwind label %ehcleanup

catch:                                            ; preds = %catch.dispatch.2
  %2 = catchpad within %cs3 [ptr @"\01??_R0?AUC@@@8", i32 0, ptr null]
  catchret from %2 to label %try.cont

ehcleanup:                                        ; preds = %catchendblock
  %3 = cleanuppad within none []
  call void @"\01??1HasDtor@@QEAA@XZ"(ptr nonnull %o) #4 [ "funclet"(token %3) ]
  cleanupret from %3 unwind to caller
}

; Function Attrs: nounwind argmemonly
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1

declare void @"\01?may_throw@@YAXXZ"() #2

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind
declare void @"\01??1HasDtor@@QEAA@XZ"(ptr) #3

; Function Attrs: nounwind argmemonly
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind argmemonly }
attributes #2 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }
