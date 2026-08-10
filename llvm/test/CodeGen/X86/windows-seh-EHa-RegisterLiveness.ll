; XFAIL: *
; RUN: llc --verify-machineinstrs < %s
source_filename = "test.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.12.0"

$"?test@Test@@Plugin@@Host@@@Z" = comdat any

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind memory(none)
declare void @llvm.seh.scope.begin() #1

; Function Attrs: nobuiltin allocsize(0)
declare ptr @"??2@Test@Z"(i64) #1

; Function Attrs: nounwind memory(none)
declare void @llvm.seh.scope.end() #0

; Function Attrs: nobuiltin nounwind
declare void @"??3@YAXPEAX@Z"(ptr) #2

; Function Attrs: mustprogress uwtable
define ptr @"?test@Test@@Plugin@@Host@@@Z"(ptr %this, ptr %host) #3 comdat align 2 personality ptr @__CxxFrameHandler3 {
entry:
  %host.addr = alloca ptr, align 8
  %this.addr = alloca ptr, align 8
  store ptr %host, ptr %host.addr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  %call = call noalias ptr @"??2@Test@Z"(i64 152) #5
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %call3 = invoke ptr @"??Test@?A0x2749C4FD@@QEAA@Test@Test@@@Z"(ptr %call, ptr %this1)
          to label %invoke.cont2 unwind label %ehcleanup

invoke.cont2:                                     ; preds = %invoke.cont
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont4 unwind label %ehcleanup

invoke.cont4:                                     ; preds = %invoke.cont2
  ret ptr %call

ehcleanup:                                        ; preds = %invoke.cont2, %invoke.cont, %entry
  %0 = cleanuppad within none []
  call void @"??3@YAXPEAX@Z"(ptr %call) #6 [ "funclet"(token %0) ]
  cleanupret from %0 unwind to caller
}

; Function Attrs: uwtable
declare hidden ptr @"??Test@?A0x2749C4FD@@QEAA@Test@Test@@@Z"(ptr, ptr) #4 align 2

attributes #0 = { nounwind memory(none) }
attributes #1 = { nobuiltin allocsize(0) "target-cpu"="x86-64" "target-features"="+cmov,+crc32,+cx8,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87" "tune-cpu"="generic" }
attributes #2 = { nobuiltin nounwind "target-cpu"="x86-64" "target-features"="+cmov,+crc32,+cx8,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87" "tune-cpu"="generic" }
attributes #3 = { mustprogress uwtable "target-cpu"="x86-64" "target-features"="+cmov,+crc32,+cx8,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87" "tune-cpu"="generic" }
attributes #4 = { uwtable "target-cpu"="x86-64" "target-features"="+cmov,+crc32,+cx8,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87" "tune-cpu"="generic" }
attributes #5 = { builtin allocsize(0) }
attributes #6 = { builtin nounwind }

!llvm.module.flags = !{!1, !2}

!1 = !{i32 2, !"eh-asynch", i32 1}
!2 = !{i32 7, !"uwtable", i32 2}
