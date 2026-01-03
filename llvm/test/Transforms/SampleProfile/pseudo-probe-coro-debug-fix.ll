; RUN: opt < %s -passes='pseudo-probe,cgscc(coro-split),coro-cleanup,early-cse' -mtriple=x86_64 -S -o %t
; RUN: llc -mtriple=x86_64-- < %t | FileCheck %s

; CHECK:        .section	.pseudo_probe_desc,"",@progbits
; CHECK-NEXT:   .quad	9191153033785521275
; CHECK-NEXT:   .quad	3378390095374084
; CHECK-NEXT:   .byte	7
; CHECK-NEXT:   .ascii	"_Z3foov"
; CHECK-NEXT:   .quad	1080103601501470593
; CHECK-NEXT:   .quad	562954248388607
; CHECK-NEXT:   .byte	22
; CHECK-NEXT:   .ascii	"__clang_call_terminate"
; CHECK-NEXT:   .quad	-5983175768063907623
; CHECK-NEXT:   .quad	4294967295
; CHECK-NEXT:   .byte	43
; CHECK-NEXT:   .ascii	"_ZN4Task12promise_type17get_return_objectEv"
; CHECK-NEXT:   .quad	9223044513579825966
; CHECK-NEXT:   .quad	4294967295
; CHECK-NEXT:   .byte	41
; CHECK-NEXT:   .ascii	"_ZN4Task12promise_type15initial_suspendEv"
; CHECK-NEXT:   .quad	3637514160624257742
; CHECK-NEXT:   .quad	4294967295
; CHECK-NEXT:   .byte	44
; CHECK-NEXT:   .ascii	"_ZNKSt7__n486113suspend_never11await_readyEv"
; CHECK-NEXT:   .quad	4804537283846854890
; CHECK-NEXT:   .quad	844429225099263
; CHECK-NEXT:   .byte	37
; CHECK-NEXT:   .ascii	"_Z3foov.__await_suspend_wrapper__init"
; CHECK-NEXT:   .quad	7163718281371254139
; CHECK-NEXT:   .quad	4294967295
; CHECK-NEXT:   .byte	70
; CHECK-NEXT:   .ascii	"_ZNKSt7__n486113suspend_never13await_suspendENS_16coroutine_handleIvEE"
; CHECK-NEXT:   .quad	-8352551662853914866
; CHECK-NEXT:   .quad	281479271677951
; CHECK-NEXT:   .byte	71
; CHECK-NEXT:   .ascii	"_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEE12from_addressEPv"
; CHECK-NEXT:   .quad	-4287562419115964352
; CHECK-NEXT:   .quad	562954248388607
; CHECK-NEXT:   .byte	67
; CHECK-NEXT:   .ascii	"_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEEcvNS0_IvEEEv"
; CHECK-NEXT:   .quad	-3595869838865620152
; CHECK-NEXT:   .quad	4294967295
; CHECK-NEXT:   .byte	45
; CHECK-NEXT:   .ascii	"_ZNKSt7__n486113suspend_never12await_resumeEv"
; CHECK-NEXT:   .quad	-4360651593791024226
; CHECK-NEXT:   .quad	4294967295
; CHECK-NEXT:   .byte	16
; CHECK-NEXT:   .ascii	"_ZN8co_sleepC2Ei"
; CHECK-NEXT:   .quad	194070738489204334
; CHECK-NEXT:   .quad	4294967295
; CHECK-NEXT:   .byte	28
; CHECK-NEXT:   .ascii	"_ZNK8co_sleep11await_readyEv"
; CHECK-NEXT:   .quad	-6207857456115168182
; CHECK-NEXT:   .quad	844429225099263
; CHECK-NEXT:   .byte	38
; CHECK-NEXT:   .ascii	"_Z3foov.__await_suspend_wrapper__await"
; CHECK-NEXT:   .quad	-707552474449578794
; CHECK-NEXT:   .quad	4294967295
; CHECK-NEXT:   .byte	62
; CHECK-NEXT:   .ascii	"_ZNK8co_sleep13await_suspendENSt7__n486116coroutine_handleIvEE"
; CHECK-NEXT:   .quad	2482723163529722706
; CHECK-NEXT:   .quad	4294967295
; CHECK-NEXT:   .byte	29
; CHECK-NEXT:   .ascii	"_ZNK8co_sleep12await_resumeEv"
; CHECK-NEXT:   .quad	2751305985409977170
; CHECK-NEXT:   .quad	4294967295
; CHECK-NEXT:   .byte	45
; CHECK-NEXT:   .ascii	"_ZN4Task12promise_type19unhandled_exceptionEv"
; CHECK-NEXT:   .quad	5392659810511438807
; CHECK-NEXT:   .quad	4294967295
; CHECK-NEXT:   .byte	39
; CHECK-NEXT:   .ascii	"_ZN4Task12promise_type13final_suspendEv"
; CHECK-NEXT:   .quad	-5299639869497052884
; CHECK-NEXT:   .quad	4294967295
; CHECK-NEXT:   .byte	45
; CHECK-NEXT:   .ascii	"_ZNKSt7__n486114suspend_always11await_readyEv"
; CHECK-NEXT:   .quad	-3505192329872532516
; CHECK-NEXT:   .quad	844429225099263
; CHECK-NEXT:   .byte	38
; CHECK-NEXT:   .ascii	"_Z3foov.__await_suspend_wrapper__final"
; CHECK-NEXT:   .quad	-7204990471824719232
; CHECK-NEXT:   .quad	4294967295
; CHECK-NEXT:   .byte	71
; CHECK-NEXT:   .ascii	"_ZNKSt7__n486114suspend_always13await_suspendENS_16coroutine_handleIvEE"
; CHECK-NEXT:   .quad	1383777007208917084
; CHECK-NEXT:   .quad	4294967295
; CHECK-NEXT:   .byte	46
; CHECK-NEXT:   .ascii	"_ZNKSt7__n486114suspend_always12await_resumeEv"
; CHECK-NEXT:   .quad	-2624081020897602054
; CHECK-NEXT:   .quad	281479271677951
; CHECK-NEXT:   .byte	4
; CHECK-NEXT:   .ascii	"main"
; CHECK-NEXT:   .quad	8552724234615586455
; CHECK-NEXT:   .quad	4294967295
; CHECK-NEXT:   .byte	58
; CHECK-NEXT:   .ascii	"_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEEC2Ev"
; CHECK-NEXT:   .quad	8637232519489628998
; CHECK-NEXT:   .quad	281479271677951
; CHECK-NEXT:   .byte	51
; CHECK-NEXT:   .ascii	"_ZNSt7__n486116coroutine_handleIvE12from_addressEPv"
; CHECK-NEXT:   .quad	8181587454597930979
; CHECK-NEXT:   .quad	4294967295
; CHECK-NEXT:   .byte	65
; CHECK-NEXT:   .ascii	"_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEE7addressEv"
; CHECK-NEXT:   .quad	2426770966769951745
; CHECK-NEXT:   .quad	4294967295
; CHECK-NEXT:   .byte	38
; CHECK-NEXT:   .ascii	"_ZNSt7__n486116coroutine_handleIvEC2Ev"

; ModuleID = 'SampleProfile/Inputs/pseudo-probe-coro-debug-fix.cpp'
source_filename = "SampleProfile/Inputs/pseudo-probe-coro-debug-fix.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

%"struct.Task::promise_type" = type { i8 }
%struct.Task = type { i8 }
%"struct.std::__n4861::suspend_never" = type { i8 }
%struct.co_sleep = type { i32 }
%"struct.std::__n4861::suspend_always" = type { i8 }
%"struct.std::__n4861::coroutine_handle" = type { ptr }
%"struct.std::__n4861::coroutine_handle.0" = type { ptr }

$__clang_call_terminate = comdat any

$_ZN4Task12promise_type17get_return_objectEv = comdat any

$_ZN4Task12promise_type15initial_suspendEv = comdat any

$_ZNKSt7__n486113suspend_never11await_readyEv = comdat any

$_ZNKSt7__n486113suspend_never13await_suspendENS_16coroutine_handleIvEE = comdat any

$_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEE12from_addressEPv = comdat any

$_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEEcvNS0_IvEEEv = comdat any

$_ZNKSt7__n486113suspend_never12await_resumeEv = comdat any

$_ZN8co_sleepC2Ei = comdat any

$_ZNK8co_sleep11await_readyEv = comdat any

$_ZNK8co_sleep13await_suspendENSt7__n486116coroutine_handleIvEE = comdat any

$_ZNK8co_sleep12await_resumeEv = comdat any

$_ZN4Task12promise_type19unhandled_exceptionEv = comdat any

$_ZN4Task12promise_type13final_suspendEv = comdat any

$_ZNKSt7__n486114suspend_always11await_readyEv = comdat any

$_ZNKSt7__n486114suspend_always13await_suspendENS_16coroutine_handleIvEE = comdat any

$_ZNKSt7__n486114suspend_always12await_resumeEv = comdat any

$_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEEC2Ev = comdat any

$_ZNSt7__n486116coroutine_handleIvE12from_addressEPv = comdat any

$_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEE7addressEv = comdat any

$_ZNSt7__n486116coroutine_handleIvEC2Ev = comdat any

; Function Attrs: mustprogress noinline nounwind optnone presplitcoroutine uwtable
define dso_local void @_Z3foov() #0 personality ptr @__gxx_personality_v0 {
entry:
  %__promise = alloca %"struct.Task::promise_type", align 1
  %undef.agg.tmp = alloca %struct.Task, align 1
  %ref.tmp = alloca %"struct.std::__n4861::suspend_never", align 1
  %undef.agg.tmp3 = alloca %"struct.std::__n4861::suspend_never", align 1
  %ref.tmp5 = alloca %struct.co_sleep, align 4
  %exn.slot = alloca ptr, align 8
  %ehselector.slot = alloca i32, align 4
  %ref.tmp13 = alloca %"struct.std::__n4861::suspend_always", align 1
  %undef.agg.tmp14 = alloca %"struct.std::__n4861::suspend_always", align 1
  %0 = call token @llvm.coro.id(i32 16, ptr %__promise, ptr null, ptr null)
  %1 = call i1 @llvm.coro.alloc(token %0)
  br i1 %1, label %coro.alloc, label %coro.init

coro.alloc:                                       ; preds = %entry
  %2 = call i64 @llvm.coro.size.i64()
  %call = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %2) #13
          to label %invoke.cont unwind label %terminate.lpad

invoke.cont:                                      ; preds = %coro.alloc
  br label %coro.init

coro.init:                                        ; preds = %invoke.cont, %entry
  %3 = phi ptr [ null, %entry ], [ %call, %invoke.cont ]
  %4 = call ptr @llvm.coro.begin(token %0, ptr %3)
  call void @llvm.lifetime.start.p0(ptr %__promise) #2
  invoke void @_ZN4Task12promise_type17get_return_objectEv(ptr noundef nonnull align 1 dereferenceable(1) %__promise)
          to label %invoke.cont1 unwind label %terminate.lpad

invoke.cont1:                                     ; preds = %coro.init
  call void @llvm.lifetime.start.p0(ptr %ref.tmp) #2
  invoke void @_ZN4Task12promise_type15initial_suspendEv(ptr noundef nonnull align 1 dereferenceable(1) %__promise)
          to label %invoke.cont2 unwind label %terminate.lpad

invoke.cont2:                                     ; preds = %invoke.cont1
  %call4 = call noundef zeroext i1 @_ZNKSt7__n486113suspend_never11await_readyEv(ptr noundef nonnull align 1 dereferenceable(1) %ref.tmp) #2
  br i1 %call4, label %init.ready, label %init.suspend

init.suspend:                                     ; preds = %invoke.cont2
  %5 = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.void(ptr %ref.tmp, ptr %4, ptr @_Z3foov.__await_suspend_wrapper__init) #2
  %6 = call i8 @llvm.coro.suspend(token %5, i1 false)
  switch i8 %6, label %coro.ret [
    i8 0, label %init.ready
    i8 1, label %init.cleanup
  ]

init.cleanup:                                     ; preds = %init.suspend
  br label %cleanup

init.ready:                                       ; preds = %init.suspend, %invoke.cont2
  call void @_ZNKSt7__n486113suspend_never12await_resumeEv(ptr noundef nonnull align 1 dereferenceable(1) %ref.tmp) #2
  br label %cleanup

cleanup:                                          ; preds = %init.ready, %init.cleanup
  %cleanup.dest.slot.0 = phi i32 [ 0, %init.ready ], [ 2, %init.cleanup ]
  call void @llvm.lifetime.end.p0(ptr %ref.tmp) #2
  switch i32 %cleanup.dest.slot.0, label %cleanup19 [
    i32 0, label %cleanup.cont
  ]

cleanup.cont:                                     ; preds = %cleanup
  call void @llvm.lifetime.start.p0(ptr %ref.tmp5) #2
  invoke void @_ZN8co_sleepC2Ei(ptr noundef nonnull align 4 dereferenceable(4) %ref.tmp5, i32 noundef 10)
          to label %invoke.cont6 unwind label %lpad

invoke.cont6:                                     ; preds = %cleanup.cont
  %call7 = call noundef zeroext i1 @_ZNK8co_sleep11await_readyEv(ptr noundef nonnull align 4 dereferenceable(4) %ref.tmp5) #2
  br i1 %call7, label %await.ready, label %await.suspend

await.suspend:                                    ; preds = %invoke.cont6
  %7 = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.void(ptr %ref.tmp5, ptr %4, ptr @_Z3foov.__await_suspend_wrapper__await) #2
  %8 = call i8 @llvm.coro.suspend(token %7, i1 false)
  switch i8 %8, label %coro.ret [
    i8 0, label %await.ready
    i8 1, label %await.cleanup
  ]

await.cleanup:                                    ; preds = %await.suspend
  br label %cleanup8

lpad:                                             ; preds = %cleanup.cont
  %9 = landingpad { ptr, i32 }
          catch ptr null
  %10 = extractvalue { ptr, i32 } %9, 0
  store ptr %10, ptr %exn.slot, align 8
  %11 = extractvalue { ptr, i32 } %9, 1
  store i32 %11, ptr %ehselector.slot, align 4
  call void @llvm.lifetime.end.p0(ptr %ref.tmp5) #2
  br label %catch

catch:                                            ; preds = %lpad
  %exn = load ptr, ptr %exn.slot, align 8
  %12 = call ptr @__cxa_begin_catch(ptr %exn) #2
  invoke void @_ZN4Task12promise_type19unhandled_exceptionEv(ptr noundef nonnull align 1 dereferenceable(1) %__promise)
          to label %invoke.cont11 unwind label %terminate.lpad

invoke.cont11:                                    ; preds = %catch
  invoke void @__cxa_end_catch()
          to label %invoke.cont12 unwind label %terminate.lpad

invoke.cont12:                                    ; preds = %invoke.cont11
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont12, %cleanup.cont10
  br label %coro.final

coro.final:                                       ; preds = %try.cont
  call void @llvm.lifetime.start.p0(ptr %ref.tmp13) #2
  call void @_ZN4Task12promise_type13final_suspendEv(ptr noundef nonnull align 1 dereferenceable(1) %__promise) #2
  %call15 = call noundef zeroext i1 @_ZNKSt7__n486114suspend_always11await_readyEv(ptr noundef nonnull align 1 dereferenceable(1) %ref.tmp13) #2
  br i1 %call15, label %final.ready, label %final.suspend

final.suspend:                                    ; preds = %coro.final
  %13 = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.void(ptr %ref.tmp13, ptr %4, ptr @_Z3foov.__await_suspend_wrapper__final) #2
  %14 = call i8 @llvm.coro.suspend(token %13, i1 true)
  switch i8 %14, label %coro.ret [
    i8 0, label %final.ready
    i8 1, label %final.cleanup
  ]

final.cleanup:                                    ; preds = %final.suspend
  br label %cleanup16

await.ready:                                      ; preds = %await.suspend, %invoke.cont6
  call void @_ZNK8co_sleep12await_resumeEv(ptr noundef nonnull align 4 dereferenceable(4) %ref.tmp5) #2
  br label %cleanup8

cleanup8:                                         ; preds = %await.ready, %await.cleanup
  %cleanup.dest.slot.1 = phi i32 [ 0, %await.ready ], [ 2, %await.cleanup ]
  call void @llvm.lifetime.end.p0(ptr %ref.tmp5) #2
  switch i32 %cleanup.dest.slot.1, label %cleanup19 [
    i32 0, label %cleanup.cont10
  ]

cleanup.cont10:                                   ; preds = %cleanup8
  br label %try.cont

final.ready:                                      ; preds = %final.suspend, %coro.final
  call void @_ZNKSt7__n486114suspend_always12await_resumeEv(ptr noundef nonnull align 1 dereferenceable(1) %ref.tmp13) #2
  br label %cleanup16

cleanup16:                                        ; preds = %final.ready, %final.cleanup
  %cleanup.dest.slot.2 = phi i32 [ 0, %final.ready ], [ 2, %final.cleanup ]
  call void @llvm.lifetime.end.p0(ptr %ref.tmp13) #2
  switch i32 %cleanup.dest.slot.2, label %cleanup19 [
    i32 0, label %cleanup.cont18
  ]

cleanup.cont18:                                   ; preds = %cleanup16
  br label %cleanup19

cleanup19:                                        ; preds = %cleanup.cont18, %cleanup16, %cleanup8, %cleanup
  %cleanup.dest.slot.3 = phi i32 [ %cleanup.dest.slot.0, %cleanup ], [ %cleanup.dest.slot.1, %cleanup8 ], [ %cleanup.dest.slot.2, %cleanup16 ], [ 0, %cleanup.cont18 ]
  call void @llvm.lifetime.end.p0(ptr %__promise) #2
  %15 = call ptr @llvm.coro.free(token %0, ptr %4)
  %16 = icmp ne ptr %15, null
  br i1 %16, label %coro.free, label %after.coro.free

coro.free:                                        ; preds = %cleanup19
  %17 = call i64 @llvm.coro.size.i64()
  call void @_ZdlPvm(ptr noundef %15, i64 noundef %17) #2
  br label %after.coro.free

after.coro.free:                                  ; preds = %cleanup19, %coro.free
  switch i32 %cleanup.dest.slot.3, label %unreachable [
    i32 0, label %cleanup.cont22
    i32 2, label %coro.ret
  ]

cleanup.cont22:                                   ; preds = %after.coro.free
  br label %coro.ret

coro.ret:                                         ; preds = %cleanup.cont22, %after.coro.free, %final.suspend, %await.suspend, %init.suspend
  call void @llvm.coro.end(ptr null, i1 false, token none)
  ret void

terminate.lpad:                                   ; preds = %invoke.cont11, %catch, %invoke.cont1, %coro.init, %coro.alloc
  %18 = landingpad { ptr, i32 }
          catch ptr null
  %19 = extractvalue { ptr, i32 } %18, 0
  call void @__clang_call_terminate(ptr %19) #14
  unreachable

unreachable:                                      ; preds = %after.coro.free
  unreachable
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare token @llvm.coro.id(i32, ptr readnone, ptr readonly captures(none), ptr) #1

; Function Attrs: nounwind
declare i1 @llvm.coro.alloc(token) #2

; Function Attrs: nobuiltin allocsize(0)
declare dso_local noundef nonnull ptr @_Znwm(i64 noundef) #3

; Function Attrs: nounwind memory(none)
declare i64 @llvm.coro.size.i64() #4

declare dso_local i32 @__gxx_personality_v0(...)

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) #5 comdat {
  %2 = call ptr @__cxa_begin_catch(ptr %0) #2
  call void @_ZSt9terminatev() #14
  unreachable
}

declare dso_local ptr @__cxa_begin_catch(ptr)

declare dso_local void @_ZSt9terminatev()

; Function Attrs: nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #6

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4Task12promise_type17get_return_objectEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #7 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4Task12promise_type15initial_suspendEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #7 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZNKSt7__n486113suspend_never11await_readyEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #7 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret i1 true
}

; Function Attrs: nomerge nounwind
declare token @llvm.coro.save(ptr) #8

; Function Attrs: alwaysinline mustprogress
define internal void @_Z3foov.__await_suspend_wrapper__init(ptr noundef nonnull %0, ptr noundef %1) #9 {
entry:
  %.addr = alloca ptr, align 8
  %.addr1 = alloca ptr, align 8
  %agg.tmp = alloca %"struct.std::__n4861::coroutine_handle", align 8
  %ref.tmp = alloca %"struct.std::__n4861::coroutine_handle.0", align 8
  store ptr %0, ptr %.addr, align 8
  store ptr %1, ptr %.addr1, align 8
  %2 = load ptr, ptr %.addr, align 8
  %3 = load ptr, ptr %.addr1, align 8
  %call = call ptr @_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEE12from_addressEPv(ptr noundef %3) #2
  %coerce.dive = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle.0", ptr %ref.tmp, i32 0, i32 0
  store ptr %call, ptr %coerce.dive, align 8
  %call2 = call ptr @_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEEcvNS0_IvEEEv(ptr noundef nonnull align 8 dereferenceable(8) %ref.tmp) #2
  %coerce.dive3 = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %agg.tmp, i32 0, i32 0
  store ptr %call2, ptr %coerce.dive3, align 8
  %coerce.dive4 = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %agg.tmp, i32 0, i32 0
  %4 = load ptr, ptr %coerce.dive4, align 8
  call void @_ZNKSt7__n486113suspend_never13await_suspendENS_16coroutine_handleIvEE(ptr noundef nonnull align 1 dereferenceable(1) %2, ptr %4) #2
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNKSt7__n486113suspend_never13await_suspendENS_16coroutine_handleIvEE(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr %.coerce) #7 comdat align 2 {
entry:
  %0 = alloca %"struct.std::__n4861::coroutine_handle", align 8
  %this.addr = alloca ptr, align 8
  %coerce.dive = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %0, i32 0, i32 0
  store ptr %.coerce, ptr %coerce.dive, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local ptr @_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEE12from_addressEPv(ptr noundef %__a) #7 comdat align 2 {
entry:
  %retval = alloca %"struct.std::__n4861::coroutine_handle.0", align 8
  %__a.addr = alloca ptr, align 8
  store ptr %__a, ptr %__a.addr, align 8
  call void @_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEEC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %retval) #2
  %0 = load ptr, ptr %__a.addr, align 8
  %_M_fr_ptr = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle.0", ptr %retval, i32 0, i32 0
  store ptr %0, ptr %_M_fr_ptr, align 8
  %coerce.dive = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle.0", ptr %retval, i32 0, i32 0
  %1 = load ptr, ptr %coerce.dive, align 8
  ret ptr %1
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local ptr @_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEEcvNS0_IvEEEv(ptr noundef nonnull align 8 dereferenceable(8) %this) #7 comdat align 2 {
entry:
  %retval = alloca %"struct.std::__n4861::coroutine_handle", align 8
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  %call = call noundef ptr @_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEE7addressEv(ptr noundef nonnull align 8 dereferenceable(8) %this1) #2
  %call2 = call ptr @_ZNSt7__n486116coroutine_handleIvE12from_addressEPv(ptr noundef %call) #2
  %coerce.dive = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %retval, i32 0, i32 0
  store ptr %call2, ptr %coerce.dive, align 8
  %coerce.dive3 = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %retval, i32 0, i32 0
  %0 = load ptr, ptr %coerce.dive3, align 8
  ret ptr %0
}

declare void @llvm.coro.await.suspend.void(ptr, ptr, ptr)

; Function Attrs: nounwind
declare i8 @llvm.coro.suspend(token, i1) #2

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNKSt7__n486113suspend_never12await_resumeEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #7 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #6

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN8co_sleepC2Ei(ptr noundef nonnull align 4 dereferenceable(4) %this, i32 noundef %n) unnamed_addr #7 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  %n.addr = alloca i32, align 4
  store ptr %this, ptr %this.addr, align 8
  store i32 %n, ptr %n.addr, align 4
  %this1 = load ptr, ptr %this.addr, align 8
  %delay = getelementptr inbounds nuw %struct.co_sleep, ptr %this1, i32 0, i32 0
  %0 = load i32, ptr %n.addr, align 4
  store i32 %0, ptr %delay, align 4
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZNK8co_sleep11await_readyEv(ptr noundef nonnull align 4 dereferenceable(4) %this) #7 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret i1 false
}

; Function Attrs: alwaysinline mustprogress
define internal void @_Z3foov.__await_suspend_wrapper__await(ptr noundef nonnull %0, ptr noundef %1) #9 {
entry:
  %.addr = alloca ptr, align 8
  %.addr1 = alloca ptr, align 8
  %agg.tmp = alloca %"struct.std::__n4861::coroutine_handle", align 8
  %ref.tmp = alloca %"struct.std::__n4861::coroutine_handle.0", align 8
  store ptr %0, ptr %.addr, align 8
  store ptr %1, ptr %.addr1, align 8
  %2 = load ptr, ptr %.addr, align 8
  %3 = load ptr, ptr %.addr1, align 8
  %call = call ptr @_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEE12from_addressEPv(ptr noundef %3) #2
  %coerce.dive = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle.0", ptr %ref.tmp, i32 0, i32 0
  store ptr %call, ptr %coerce.dive, align 8
  %call2 = call ptr @_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEEcvNS0_IvEEEv(ptr noundef nonnull align 8 dereferenceable(8) %ref.tmp) #2
  %coerce.dive3 = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %agg.tmp, i32 0, i32 0
  store ptr %call2, ptr %coerce.dive3, align 8
  %coerce.dive4 = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %agg.tmp, i32 0, i32 0
  %4 = load ptr, ptr %coerce.dive4, align 8
  call void @_ZNK8co_sleep13await_suspendENSt7__n486116coroutine_handleIvEE(ptr noundef nonnull align 4 dereferenceable(4) %2, ptr %4) #2
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNK8co_sleep13await_suspendENSt7__n486116coroutine_handleIvEE(ptr noundef nonnull align 4 dereferenceable(4) %this, ptr %h.coerce) #7 comdat align 2 {
entry:
  %h = alloca %"struct.std::__n4861::coroutine_handle", align 8
  %this.addr = alloca ptr, align 8
  %coerce.dive = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %h, i32 0, i32 0
  store ptr %h.coerce, ptr %coerce.dive, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNK8co_sleep12await_resumeEv(ptr noundef nonnull align 4 dereferenceable(4) %this) #7 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4Task12promise_type19unhandled_exceptionEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #7 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret void
}

declare dso_local void @__cxa_end_catch()

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4Task12promise_type13final_suspendEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #7 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZNKSt7__n486114suspend_always11await_readyEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #7 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret i1 false
}

; Function Attrs: alwaysinline mustprogress
define internal void @_Z3foov.__await_suspend_wrapper__final(ptr noundef nonnull %0, ptr noundef %1) #9 {
entry:
  %.addr = alloca ptr, align 8
  %.addr1 = alloca ptr, align 8
  %agg.tmp = alloca %"struct.std::__n4861::coroutine_handle", align 8
  %ref.tmp = alloca %"struct.std::__n4861::coroutine_handle.0", align 8
  store ptr %0, ptr %.addr, align 8
  store ptr %1, ptr %.addr1, align 8
  %2 = load ptr, ptr %.addr, align 8
  %3 = load ptr, ptr %.addr1, align 8
  %call = call ptr @_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEE12from_addressEPv(ptr noundef %3) #2
  %coerce.dive = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle.0", ptr %ref.tmp, i32 0, i32 0
  store ptr %call, ptr %coerce.dive, align 8
  %call2 = call ptr @_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEEcvNS0_IvEEEv(ptr noundef nonnull align 8 dereferenceable(8) %ref.tmp) #2
  %coerce.dive3 = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %agg.tmp, i32 0, i32 0
  store ptr %call2, ptr %coerce.dive3, align 8
  %coerce.dive4 = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %agg.tmp, i32 0, i32 0
  %4 = load ptr, ptr %coerce.dive4, align 8
  call void @_ZNKSt7__n486114suspend_always13await_suspendENS_16coroutine_handleIvEE(ptr noundef nonnull align 1 dereferenceable(1) %2, ptr %4) #2
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNKSt7__n486114suspend_always13await_suspendENS_16coroutine_handleIvEE(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr %.coerce) #7 comdat align 2 {
entry:
  %0 = alloca %"struct.std::__n4861::coroutine_handle", align 8
  %this.addr = alloca ptr, align 8
  %coerce.dive = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %0, i32 0, i32 0
  store ptr %.coerce, ptr %coerce.dive, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNKSt7__n486114suspend_always12await_resumeEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #7 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret void
}

; Function Attrs: nobuiltin nounwind
declare dso_local void @_ZdlPvm(ptr noundef, i64 noundef) #10

; Function Attrs: nounwind memory(argmem: read)
declare ptr @llvm.coro.free(token, ptr readonly captures(none)) #11

; Function Attrs: nounwind
declare void @llvm.coro.end(ptr, i1, token) #2

; Function Attrs: mustprogress noinline norecurse nounwind optnone uwtable
define dso_local noundef i32 @main() #12 {
entry:
  %undef.agg.tmp = alloca %struct.Task, align 1
  call void @_Z3foov() #2
  ret i32 0
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEEC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #7 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  %_M_fr_ptr = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle.0", ptr %this1, i32 0, i32 0
  store ptr null, ptr %_M_fr_ptr, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local ptr @_ZNSt7__n486116coroutine_handleIvE12from_addressEPv(ptr noundef %__a) #7 comdat align 2 {
entry:
  %retval = alloca %"struct.std::__n4861::coroutine_handle", align 8
  %__a.addr = alloca ptr, align 8
  store ptr %__a, ptr %__a.addr, align 8
  call void @_ZNSt7__n486116coroutine_handleIvEC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %retval) #2
  %0 = load ptr, ptr %__a.addr, align 8
  %_M_fr_ptr = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %retval, i32 0, i32 0
  store ptr %0, ptr %_M_fr_ptr, align 8
  %coerce.dive = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %retval, i32 0, i32 0
  %1 = load ptr, ptr %coerce.dive, align 8
  ret ptr %1
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef ptr @_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEE7addressEv(ptr noundef nonnull align 8 dereferenceable(8) %this) #7 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  %_M_fr_ptr = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle.0", ptr %this1, i32 0, i32 0
  %0 = load ptr, ptr %_M_fr_ptr, align 8
  ret ptr %0
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNSt7__n486116coroutine_handleIvEC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #7 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  %_M_fr_ptr = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %this1, i32 0, i32 0
  store ptr null, ptr %_M_fr_ptr, align 8
  ret void
}

attributes #0 = { mustprogress noinline nounwind optnone presplitcoroutine uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #2 = { nounwind }
attributes #3 = { nobuiltin allocsize(0) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { nounwind memory(none) }
attributes #5 = { noinline noreturn nounwind uwtable "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #8 = { nomerge nounwind }
attributes #9 = { alwaysinline mustprogress "min-legal-vector-width"="0" }
attributes #10 = { nobuiltin nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #11 = { nounwind memory(argmem: read) }
attributes #12 = { mustprogress noinline norecurse nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #13 = { allocsize(0) }
attributes #14 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 2}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 22.0.0git (https://github.com/llvm/llvm-project.git 837d8c3f38d82a69e2cd9365ca9bcc82f31628df)"}
