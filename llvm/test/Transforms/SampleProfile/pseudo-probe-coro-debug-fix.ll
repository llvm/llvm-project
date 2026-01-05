; RUN: opt < %s -passes='pseudo-probe,cgscc(coro-split),coro-cleanup,always-inline,early-cse' -mtriple=x86_64 -pass-remarks=inline -S -o %t.ll
; RUN: llc -mtriple=x86_64 -stop-after=pseudo-probe-inserter < %t.ll --filetype=asm -o - | FileCheck -check-prefix=MIR %s
; MIR-NOT: PSEUDO_PROBE 4448042984153125393,
; MIR-NOT: PSEUDO_PROBE 3532999944647676065,
; MIR-NOT: PSEUDO_PROBE -7235034626494519075,

; RUN: llc -mtriple=x86_64 < %t.ll --filetype=obj -o %t.obj
; RUN: obj2yaml %t.obj | FileCheck -check-prefix=OBJ --match-full-lines %s
; OBJ:       - Name:            '.pseudo_probe (21)'{{$}}
; OBJ-NEXT:    Type:            SHT_PROGBITS{{$}}
; OBJ-NEXT:    Flags:           [ SHF_LINK_ORDER ]{{$}}
; OBJ-NEXT:    Link:            .text{{$}}
; OBJ-NEXT:    AddressAlign:    0x1{{$}}
; OBJ-NEXT:    Content:         7B340AC7FC888D7F{{[0-9A-F]+$}}


; Original source code:
;   clang++ -S -g -O0 SampleProfile/Inputs/pseudo-probe-coro-debug-fix.cpp -emit-llvm -Xclang -disable-llvm-passes -std=c++20 -o SampleProfile/pseudo-probe-coro-debug-fix.ll
;
; #include <coroutine>
; struct co_sleep {
;   co_sleep(int n) : delay{n} {}
;   constexpr bool await_ready() const noexcept { return false; }
;   void await_suspend(std::coroutine_handle<> h) const noexcept {}
;   void await_resume() const noexcept {}
;   int delay;
; };
; struct Task {
;   struct promise_type {
;     promise_type() = default;
;     Task get_return_object() { return {}; }
;     std::suspend_never initial_suspend() { return {}; }
;     std::suspend_always final_suspend() noexcept { return {}; }
;     void unhandled_exception() {}
;   };
; };
; Task foo() noexcept {
;   co_await co_sleep{10};
; }
; 
; int main() {
;   foo();
; }

; ModuleID = 'pseudo-probe-coro-debug-fix.cpp'
source_filename = "pseudo-probe-coro-debug-fix.cpp"
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
define dso_local void @_Z3foov() #0 personality ptr @__gxx_personality_v0 !dbg !156 {
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
  %0 = call token @llvm.coro.id(i32 16, ptr %__promise, ptr null, ptr null), !dbg !159
  %1 = call i1 @llvm.coro.alloc(token %0), !dbg !159
  br i1 %1, label %coro.alloc, label %coro.init, !dbg !159

coro.alloc:                                       ; preds = %entry
  %2 = call i64 @llvm.coro.size.i64(), !dbg !160
  %call = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %2) #13
          to label %invoke.cont unwind label %terminate.lpad, !dbg !160

invoke.cont:                                      ; preds = %coro.alloc
  br label %coro.init, !dbg !159

coro.init:                                        ; preds = %invoke.cont, %entry
  %3 = phi ptr [ null, %entry ], [ %call, %invoke.cont ], !dbg !159
  %4 = call ptr @llvm.coro.begin(token %0, ptr %3), !dbg !159
  call void @llvm.lifetime.start.p0(ptr %__promise) #2, !dbg !160
    #dbg_declare(ptr %__promise, !161, !DIExpression(), !167)
  invoke void @_ZN4Task12promise_type17get_return_objectEv(ptr noundef nonnull align 1 dereferenceable(1) %__promise)
          to label %invoke.cont1 unwind label %terminate.lpad, !dbg !160

invoke.cont1:                                     ; preds = %coro.init
  call void @llvm.lifetime.start.p0(ptr %ref.tmp) #2, !dbg !160
  invoke void @_ZN4Task12promise_type15initial_suspendEv(ptr noundef nonnull align 1 dereferenceable(1) %__promise)
          to label %invoke.cont2 unwind label %terminate.lpad, !dbg !160

invoke.cont2:                                     ; preds = %invoke.cont1
  %call4 = call noundef zeroext i1 @_ZNKSt7__n486113suspend_never11await_readyEv(ptr noundef nonnull align 1 dereferenceable(1) %ref.tmp) #2, !dbg !160
  br i1 %call4, label %init.ready, label %init.suspend, !dbg !160

init.suspend:                                     ; preds = %invoke.cont2
  %5 = call token @llvm.coro.save(ptr null), !dbg !160
  call void @llvm.coro.await.suspend.void(ptr %ref.tmp, ptr %4, ptr @_Z3foov.__await_suspend_wrapper__init) #2, !dbg !160
  %6 = call i8 @llvm.coro.suspend(token %5, i1 false), !dbg !160
  switch i8 %6, label %coro.ret [
    i8 0, label %init.ready
    i8 1, label %init.cleanup
  ], !dbg !160

init.cleanup:                                     ; preds = %init.suspend
  br label %cleanup, !dbg !160

init.ready:                                       ; preds = %init.suspend, %invoke.cont2
  call void @_ZNKSt7__n486113suspend_never12await_resumeEv(ptr noundef nonnull align 1 dereferenceable(1) %ref.tmp) #2, !dbg !160
  br label %cleanup, !dbg !160

cleanup:                                          ; preds = %init.ready, %init.cleanup
  %cleanup.dest.slot.0 = phi i32 [ 0, %init.ready ], [ 2, %init.cleanup ]
  call void @llvm.lifetime.end.p0(ptr %ref.tmp) #2, !dbg !160
  switch i32 %cleanup.dest.slot.0, label %cleanup19 [
    i32 0, label %cleanup.cont
  ]

cleanup.cont:                                     ; preds = %cleanup
  call void @llvm.lifetime.start.p0(ptr %ref.tmp5) #2, !dbg !168
  invoke void @_ZN8co_sleepC2Ei(ptr noundef nonnull align 4 dereferenceable(4) %ref.tmp5, i32 noundef 10)
          to label %invoke.cont6 unwind label %lpad, !dbg !168

invoke.cont6:                                     ; preds = %cleanup.cont
  %call7 = call noundef zeroext i1 @_ZNK8co_sleep11await_readyEv(ptr noundef nonnull align 4 dereferenceable(4) %ref.tmp5) #2, !dbg !168
  br i1 %call7, label %await.ready, label %await.suspend, !dbg !170

await.suspend:                                    ; preds = %invoke.cont6
  %7 = call token @llvm.coro.save(ptr null), !dbg !170
  call void @llvm.coro.await.suspend.void(ptr %ref.tmp5, ptr %4, ptr @_Z3foov.__await_suspend_wrapper__await) #2, !dbg !170
  %8 = call i8 @llvm.coro.suspend(token %7, i1 false), !dbg !170
  switch i8 %8, label %coro.ret [
    i8 0, label %await.ready
    i8 1, label %await.cleanup
  ], !dbg !170

await.cleanup:                                    ; preds = %await.suspend
  br label %cleanup8, !dbg !170

lpad:                                             ; preds = %cleanup.cont
  %9 = landingpad { ptr, i32 }
          catch ptr null, !dbg !171
  %10 = extractvalue { ptr, i32 } %9, 0, !dbg !171
  store ptr %10, ptr %exn.slot, align 8, !dbg !171
  %11 = extractvalue { ptr, i32 } %9, 1, !dbg !171
  store i32 %11, ptr %ehselector.slot, align 4, !dbg !171
  call void @llvm.lifetime.end.p0(ptr %ref.tmp5) #2, !dbg !170
  br label %catch, !dbg !170

catch:                                            ; preds = %lpad
  %exn = load ptr, ptr %exn.slot, align 8, !dbg !171
  %12 = call ptr @__cxa_begin_catch(ptr %exn) #2, !dbg !171
  invoke void @_ZN4Task12promise_type19unhandled_exceptionEv(ptr noundef nonnull align 1 dereferenceable(1) %__promise)
          to label %invoke.cont11 unwind label %terminate.lpad, !dbg !160

invoke.cont11:                                    ; preds = %catch
  invoke void @__cxa_end_catch()
          to label %invoke.cont12 unwind label %terminate.lpad, !dbg !160

invoke.cont12:                                    ; preds = %invoke.cont11
  br label %try.cont, !dbg !160

try.cont:                                         ; preds = %invoke.cont12, %cleanup.cont10
  br label %coro.final, !dbg !160

coro.final:                                       ; preds = %try.cont
  call void @llvm.lifetime.start.p0(ptr %ref.tmp13) #2, !dbg !160
  call void @_ZN4Task12promise_type13final_suspendEv(ptr noundef nonnull align 1 dereferenceable(1) %__promise) #2, !dbg !160
  %call15 = call noundef zeroext i1 @_ZNKSt7__n486114suspend_always11await_readyEv(ptr noundef nonnull align 1 dereferenceable(1) %ref.tmp13) #2, !dbg !160
  br i1 %call15, label %final.ready, label %final.suspend, !dbg !160

final.suspend:                                    ; preds = %coro.final
  %13 = call token @llvm.coro.save(ptr null), !dbg !160
  call void @llvm.coro.await.suspend.void(ptr %ref.tmp13, ptr %4, ptr @_Z3foov.__await_suspend_wrapper__final) #2, !dbg !160
  %14 = call i8 @llvm.coro.suspend(token %13, i1 true), !dbg !160
  switch i8 %14, label %coro.ret [
    i8 0, label %final.ready
    i8 1, label %final.cleanup
  ], !dbg !160

final.cleanup:                                    ; preds = %final.suspend
  br label %cleanup16, !dbg !160

await.ready:                                      ; preds = %await.suspend, %invoke.cont6
  call void @_ZNK8co_sleep12await_resumeEv(ptr noundef nonnull align 4 dereferenceable(4) %ref.tmp5) #2, !dbg !168
  br label %cleanup8, !dbg !170

cleanup8:                                         ; preds = %await.ready, %await.cleanup
  %cleanup.dest.slot.1 = phi i32 [ 0, %await.ready ], [ 2, %await.cleanup ]
  call void @llvm.lifetime.end.p0(ptr %ref.tmp5) #2, !dbg !170
  switch i32 %cleanup.dest.slot.1, label %cleanup19 [
    i32 0, label %cleanup.cont10
  ]

cleanup.cont10:                                   ; preds = %cleanup8
  br label %try.cont, !dbg !171

final.ready:                                      ; preds = %final.suspend, %coro.final
  call void @_ZNKSt7__n486114suspend_always12await_resumeEv(ptr noundef nonnull align 1 dereferenceable(1) %ref.tmp13) #2, !dbg !160
  br label %cleanup16, !dbg !160

cleanup16:                                        ; preds = %final.ready, %final.cleanup
  %cleanup.dest.slot.2 = phi i32 [ 0, %final.ready ], [ 2, %final.cleanup ]
  call void @llvm.lifetime.end.p0(ptr %ref.tmp13) #2, !dbg !160
  switch i32 %cleanup.dest.slot.2, label %cleanup19 [
    i32 0, label %cleanup.cont18
  ]

cleanup.cont18:                                   ; preds = %cleanup16
  br label %cleanup19, !dbg !160

cleanup19:                                        ; preds = %cleanup.cont18, %cleanup16, %cleanup8, %cleanup
  %cleanup.dest.slot.3 = phi i32 [ %cleanup.dest.slot.0, %cleanup ], [ %cleanup.dest.slot.1, %cleanup8 ], [ %cleanup.dest.slot.2, %cleanup16 ], [ 0, %cleanup.cont18 ], !dbg !167
  call void @llvm.lifetime.end.p0(ptr %__promise) #2, !dbg !160
  %15 = call ptr @llvm.coro.free(token %0, ptr %4), !dbg !160
  %16 = icmp ne ptr %15, null, !dbg !160
  br i1 %16, label %coro.free, label %after.coro.free, !dbg !160

coro.free:                                        ; preds = %cleanup19
  %17 = call i64 @llvm.coro.size.i64(), !dbg !160
  call void @_ZdlPvm(ptr noundef %15, i64 noundef %17) #2, !dbg !160
  br label %after.coro.free, !dbg !160

after.coro.free:                                  ; preds = %cleanup19, %coro.free
  switch i32 %cleanup.dest.slot.3, label %unreachable [
    i32 0, label %cleanup.cont22
    i32 2, label %coro.ret
  ]

cleanup.cont22:                                   ; preds = %after.coro.free
  br label %coro.ret, !dbg !160

coro.ret:                                         ; preds = %cleanup.cont22, %after.coro.free, %final.suspend, %await.suspend, %init.suspend
  call void @llvm.coro.end(ptr null, i1 false, token none), !dbg !160
  ret void, !dbg !160

terminate.lpad:                                   ; preds = %invoke.cont11, %catch, %invoke.cont1, %coro.init, %coro.alloc
  %18 = landingpad { ptr, i32 }
          catch ptr null, !dbg !160
  %19 = extractvalue { ptr, i32 } %18, 0, !dbg !160
  call void @__clang_call_terminate(ptr %19) #14, !dbg !160
  unreachable, !dbg !160

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
define linkonce_odr dso_local void @_ZN4Task12promise_type17get_return_objectEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #7 comdat align 2 !dbg !172 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !173, !DIExpression(), !175)
  %this1 = load ptr, ptr %this.addr, align 8
  ret void, !dbg !176
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4Task12promise_type15initial_suspendEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #7 comdat align 2 !dbg !177 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !178, !DIExpression(), !179)
  %this1 = load ptr, ptr %this.addr, align 8
  ret void, !dbg !180
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZNKSt7__n486113suspend_never11await_readyEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #7 comdat align 2 !dbg !181 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !182, !DIExpression(), !184)
  %this1 = load ptr, ptr %this.addr, align 8
  ret i1 true, !dbg !185
}

; Function Attrs: nomerge nounwind
declare token @llvm.coro.save(ptr) #8

; Function Attrs: alwaysinline mustprogress
define internal void @_Z3foov.__await_suspend_wrapper__init(ptr noundef nonnull %0, ptr noundef %1) #9 !dbg !186 {
entry:
  %.addr = alloca ptr, align 8
  %.addr1 = alloca ptr, align 8
  %agg.tmp = alloca %"struct.std::__n4861::coroutine_handle", align 8
  %ref.tmp = alloca %"struct.std::__n4861::coroutine_handle.0", align 8
  store ptr %0, ptr %.addr, align 8
    #dbg_declare(ptr %.addr, !188, !DIExpression(), !189)
  store ptr %1, ptr %.addr1, align 8
    #dbg_declare(ptr %.addr1, !190, !DIExpression(), !189)
  %2 = load ptr, ptr %.addr, align 8
  %3 = load ptr, ptr %.addr1, align 8
  %call = call ptr @_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEE12from_addressEPv(ptr noundef %3) #2, !dbg !191
  %coerce.dive = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle.0", ptr %ref.tmp, i32 0, i32 0, !dbg !191
  store ptr %call, ptr %coerce.dive, align 8, !dbg !191
  %call2 = call ptr @_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEEcvNS0_IvEEEv(ptr noundef nonnull align 8 dereferenceable(8) %ref.tmp) #2, !dbg !191
  %coerce.dive3 = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %agg.tmp, i32 0, i32 0, !dbg !191
  store ptr %call2, ptr %coerce.dive3, align 8, !dbg !191
  %coerce.dive4 = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %agg.tmp, i32 0, i32 0, !dbg !191
  %4 = load ptr, ptr %coerce.dive4, align 8, !dbg !191
  call void @_ZNKSt7__n486113suspend_never13await_suspendENS_16coroutine_handleIvEE(ptr noundef nonnull align 1 dereferenceable(1) %2, ptr %4) #2, !dbg !191
  ret void, !dbg !191
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNKSt7__n486113suspend_never13await_suspendENS_16coroutine_handleIvEE(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr %.coerce) #7 comdat align 2 !dbg !192 {
entry:
  %0 = alloca %"struct.std::__n4861::coroutine_handle", align 8
  %this.addr = alloca ptr, align 8
  %coerce.dive = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %0, i32 0, i32 0
  store ptr %.coerce, ptr %coerce.dive, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !193, !DIExpression(), !194)
    #dbg_declare(ptr %0, !195, !DIExpression(), !196)
  %this1 = load ptr, ptr %this.addr, align 8
  ret void, !dbg !197
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local ptr @_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEE12from_addressEPv(ptr noundef %__a) #7 comdat align 2 !dbg !198 {
entry:
  %retval = alloca %"struct.std::__n4861::coroutine_handle.0", align 8
  %__a.addr = alloca ptr, align 8
  store ptr %__a, ptr %__a.addr, align 8
    #dbg_declare(ptr %__a.addr, !199, !DIExpression(), !200)
    #dbg_declare(ptr %retval, !201, !DIExpression(), !202)
  call void @_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEEC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %retval) #2, !dbg !202
  %0 = load ptr, ptr %__a.addr, align 8, !dbg !203
  %_M_fr_ptr = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle.0", ptr %retval, i32 0, i32 0, !dbg !204
  store ptr %0, ptr %_M_fr_ptr, align 8, !dbg !205
  %coerce.dive = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle.0", ptr %retval, i32 0, i32 0, !dbg !206
  %1 = load ptr, ptr %coerce.dive, align 8, !dbg !206
  ret ptr %1, !dbg !206
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local ptr @_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEEcvNS0_IvEEEv(ptr noundef nonnull align 8 dereferenceable(8) %this) #7 comdat align 2 !dbg !207 {
entry:
  %retval = alloca %"struct.std::__n4861::coroutine_handle", align 8
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !208, !DIExpression(), !210)
  %this1 = load ptr, ptr %this.addr, align 8
  %call = call noundef ptr @_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEE7addressEv(ptr noundef nonnull align 8 dereferenceable(8) %this1) #2, !dbg !211
  %call2 = call ptr @_ZNSt7__n486116coroutine_handleIvE12from_addressEPv(ptr noundef %call) #2, !dbg !212
  %coerce.dive = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %retval, i32 0, i32 0, !dbg !212
  store ptr %call2, ptr %coerce.dive, align 8, !dbg !212
  %coerce.dive3 = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %retval, i32 0, i32 0, !dbg !213
  %0 = load ptr, ptr %coerce.dive3, align 8, !dbg !213
  ret ptr %0, !dbg !213
}

declare void @llvm.coro.await.suspend.void(ptr, ptr, ptr)

; Function Attrs: nounwind
declare i8 @llvm.coro.suspend(token, i1) #2

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNKSt7__n486113suspend_never12await_resumeEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #7 comdat align 2 !dbg !214 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !215, !DIExpression(), !216)
  %this1 = load ptr, ptr %this.addr, align 8
  ret void, !dbg !217
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #6

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN8co_sleepC2Ei(ptr noundef nonnull align 4 dereferenceable(4) %this, i32 noundef %n) unnamed_addr #7 comdat align 2 !dbg !218 {
entry:
  %this.addr = alloca ptr, align 8
  %n.addr = alloca i32, align 4
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !219, !DIExpression(), !221)
  store i32 %n, ptr %n.addr, align 4
    #dbg_declare(ptr %n.addr, !222, !DIExpression(), !223)
  %this1 = load ptr, ptr %this.addr, align 8
  %delay = getelementptr inbounds nuw %struct.co_sleep, ptr %this1, i32 0, i32 0, !dbg !224
  %0 = load i32, ptr %n.addr, align 4, !dbg !225
  store i32 %0, ptr %delay, align 4, !dbg !224
  ret void, !dbg !226
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZNK8co_sleep11await_readyEv(ptr noundef nonnull align 4 dereferenceable(4) %this) #7 comdat align 2 !dbg !227 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !228, !DIExpression(), !230)
  %this1 = load ptr, ptr %this.addr, align 8
  ret i1 false, !dbg !231
}

; Function Attrs: alwaysinline mustprogress
define internal void @_Z3foov.__await_suspend_wrapper__await(ptr noundef nonnull %0, ptr noundef %1) #9 !dbg !232 {
entry:
  %.addr = alloca ptr, align 8
  %.addr1 = alloca ptr, align 8
  %agg.tmp = alloca %"struct.std::__n4861::coroutine_handle", align 8
  %ref.tmp = alloca %"struct.std::__n4861::coroutine_handle.0", align 8
  store ptr %0, ptr %.addr, align 8
    #dbg_declare(ptr %.addr, !233, !DIExpression(), !234)
  store ptr %1, ptr %.addr1, align 8
    #dbg_declare(ptr %.addr1, !235, !DIExpression(), !234)
  %2 = load ptr, ptr %.addr, align 8
  %3 = load ptr, ptr %.addr1, align 8
  %call = call ptr @_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEE12from_addressEPv(ptr noundef %3) #2, !dbg !236
  %coerce.dive = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle.0", ptr %ref.tmp, i32 0, i32 0, !dbg !236
  store ptr %call, ptr %coerce.dive, align 8, !dbg !236
  %call2 = call ptr @_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEEcvNS0_IvEEEv(ptr noundef nonnull align 8 dereferenceable(8) %ref.tmp) #2, !dbg !236
  %coerce.dive3 = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %agg.tmp, i32 0, i32 0, !dbg !236
  store ptr %call2, ptr %coerce.dive3, align 8, !dbg !236
  %coerce.dive4 = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %agg.tmp, i32 0, i32 0, !dbg !236
  %4 = load ptr, ptr %coerce.dive4, align 8, !dbg !236
  call void @_ZNK8co_sleep13await_suspendENSt7__n486116coroutine_handleIvEE(ptr noundef nonnull align 4 dereferenceable(4) %2, ptr %4) #2, !dbg !236
  ret void, !dbg !236
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNK8co_sleep13await_suspendENSt7__n486116coroutine_handleIvEE(ptr noundef nonnull align 4 dereferenceable(4) %this, ptr %h.coerce) #7 comdat align 2 !dbg !237 {
entry:
  %h = alloca %"struct.std::__n4861::coroutine_handle", align 8
  %this.addr = alloca ptr, align 8
  %coerce.dive = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %h, i32 0, i32 0
  store ptr %h.coerce, ptr %coerce.dive, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !238, !DIExpression(), !239)
    #dbg_declare(ptr %h, !240, !DIExpression(), !241)
  %this1 = load ptr, ptr %this.addr, align 8
  ret void, !dbg !242
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNK8co_sleep12await_resumeEv(ptr noundef nonnull align 4 dereferenceable(4) %this) #7 comdat align 2 !dbg !243 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !244, !DIExpression(), !245)
  %this1 = load ptr, ptr %this.addr, align 8
  ret void, !dbg !246
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4Task12promise_type19unhandled_exceptionEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #7 comdat align 2 !dbg !247 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !248, !DIExpression(), !249)
  %this1 = load ptr, ptr %this.addr, align 8
  ret void, !dbg !250
}

declare dso_local void @__cxa_end_catch()

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4Task12promise_type13final_suspendEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #7 comdat align 2 !dbg !251 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !252, !DIExpression(), !253)
  %this1 = load ptr, ptr %this.addr, align 8
  ret void, !dbg !254
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZNKSt7__n486114suspend_always11await_readyEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #7 comdat align 2 !dbg !255 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !256, !DIExpression(), !258)
  %this1 = load ptr, ptr %this.addr, align 8
  ret i1 false, !dbg !259
}

; Function Attrs: alwaysinline mustprogress
define internal void @_Z3foov.__await_suspend_wrapper__final(ptr noundef nonnull %0, ptr noundef %1) #9 !dbg !260 {
entry:
  %.addr = alloca ptr, align 8
  %.addr1 = alloca ptr, align 8
  %agg.tmp = alloca %"struct.std::__n4861::coroutine_handle", align 8
  %ref.tmp = alloca %"struct.std::__n4861::coroutine_handle.0", align 8
  store ptr %0, ptr %.addr, align 8
    #dbg_declare(ptr %.addr, !261, !DIExpression(), !262)
  store ptr %1, ptr %.addr1, align 8
    #dbg_declare(ptr %.addr1, !263, !DIExpression(), !262)
  %2 = load ptr, ptr %.addr, align 8
  %3 = load ptr, ptr %.addr1, align 8
  %call = call ptr @_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEE12from_addressEPv(ptr noundef %3) #2, !dbg !264
  %coerce.dive = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle.0", ptr %ref.tmp, i32 0, i32 0, !dbg !264
  store ptr %call, ptr %coerce.dive, align 8, !dbg !264
  %call2 = call ptr @_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEEcvNS0_IvEEEv(ptr noundef nonnull align 8 dereferenceable(8) %ref.tmp) #2, !dbg !264
  %coerce.dive3 = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %agg.tmp, i32 0, i32 0, !dbg !264
  store ptr %call2, ptr %coerce.dive3, align 8, !dbg !264
  %coerce.dive4 = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %agg.tmp, i32 0, i32 0, !dbg !264
  %4 = load ptr, ptr %coerce.dive4, align 8, !dbg !264
  call void @_ZNKSt7__n486114suspend_always13await_suspendENS_16coroutine_handleIvEE(ptr noundef nonnull align 1 dereferenceable(1) %2, ptr %4) #2, !dbg !264
  ret void, !dbg !264
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNKSt7__n486114suspend_always13await_suspendENS_16coroutine_handleIvEE(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr %.coerce) #7 comdat align 2 !dbg !265 {
entry:
  %0 = alloca %"struct.std::__n4861::coroutine_handle", align 8
  %this.addr = alloca ptr, align 8
  %coerce.dive = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %0, i32 0, i32 0
  store ptr %.coerce, ptr %coerce.dive, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !266, !DIExpression(), !267)
    #dbg_declare(ptr %0, !268, !DIExpression(), !269)
  %this1 = load ptr, ptr %this.addr, align 8
  ret void, !dbg !270
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNKSt7__n486114suspend_always12await_resumeEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #7 comdat align 2 !dbg !271 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !272, !DIExpression(), !273)
  %this1 = load ptr, ptr %this.addr, align 8
  ret void, !dbg !274
}

; Function Attrs: nobuiltin nounwind
declare dso_local void @_ZdlPvm(ptr noundef, i64 noundef) #10

; Function Attrs: nounwind memory(argmem: read)
declare ptr @llvm.coro.free(token, ptr readonly captures(none)) #11

; Function Attrs: nounwind
declare void @llvm.coro.end(ptr, i1, token) #2

; Function Attrs: mustprogress noinline norecurse nounwind optnone uwtable
define dso_local noundef i32 @main() #12 !dbg !275 {
entry:
  %undef.agg.tmp = alloca %struct.Task, align 1
  call void @_Z3foov() #2, !dbg !278
  ret i32 0, !dbg !279
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEEC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #7 comdat align 2 !dbg !280 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !281, !DIExpression(), !283)
  %this1 = load ptr, ptr %this.addr, align 8
  %_M_fr_ptr = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle.0", ptr %this1, i32 0, i32 0, !dbg !284
  store ptr null, ptr %_M_fr_ptr, align 8, !dbg !284
  ret void, !dbg !285
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local ptr @_ZNSt7__n486116coroutine_handleIvE12from_addressEPv(ptr noundef %__a) #7 comdat align 2 !dbg !286 {
entry:
  %retval = alloca %"struct.std::__n4861::coroutine_handle", align 8
  %__a.addr = alloca ptr, align 8
  store ptr %__a, ptr %__a.addr, align 8
    #dbg_declare(ptr %__a.addr, !287, !DIExpression(), !288)
    #dbg_declare(ptr %retval, !289, !DIExpression(), !290)
  call void @_ZNSt7__n486116coroutine_handleIvEC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %retval) #2, !dbg !290
  %0 = load ptr, ptr %__a.addr, align 8, !dbg !291
  %_M_fr_ptr = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %retval, i32 0, i32 0, !dbg !292
  store ptr %0, ptr %_M_fr_ptr, align 8, !dbg !293
  %coerce.dive = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %retval, i32 0, i32 0, !dbg !294
  %1 = load ptr, ptr %coerce.dive, align 8, !dbg !294
  ret ptr %1, !dbg !294
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef ptr @_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEE7addressEv(ptr noundef nonnull align 8 dereferenceable(8) %this) #7 comdat align 2 !dbg !295 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !296, !DIExpression(), !297)
  %this1 = load ptr, ptr %this.addr, align 8
  %_M_fr_ptr = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle.0", ptr %this1, i32 0, i32 0, !dbg !298
  %0 = load ptr, ptr %_M_fr_ptr, align 8, !dbg !298
  ret ptr %0, !dbg !299
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNSt7__n486116coroutine_handleIvEC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #7 comdat align 2 !dbg !300 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !301, !DIExpression(), !303)
  %this1 = load ptr, ptr %this.addr, align 8
  %_M_fr_ptr = getelementptr inbounds nuw %"struct.std::__n4861::coroutine_handle", ptr %this1, i32 0, i32 0, !dbg !304
  store ptr null, ptr %_M_fr_ptr, align 8, !dbg !304
  ret void, !dbg !305
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

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!150, !151, !152, !153, !154}
!llvm.ident = !{!155}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 22.0.0git (https://github.com/llvm/llvm-project.git fe218aab737d55dfd67f4f84118744003f45958e)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "pseudo-probe-coro-debug-fix.cpp", directory: "/tmp", checksumkind: CSK_MD5, checksum: "601552fedf6b88dcf699ea9d066126eb")
!2 = !{!3, !31, !88, !131}
!3 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "promise_type", scope: !4, file: !1, line: 10, size: 8, flags: DIFlagTypePassByValue, elements: !6, identifier: "_ZTSN4Task12promise_typeE")
!4 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Task", file: !1, line: 9, size: 8, flags: DIFlagTypePassByValue, elements: !5, identifier: "_ZTS4Task")
!5 = !{}
!6 = !{!7, !11, !14, !71, !87}
!7 = !DISubprogram(name: "promise_type", linkageName: "_ZN4Task12promise_typeC4Ev", scope: !3, file: !1, line: 11, type: !8, scopeLine: 11, flags: DIFlagPrototyped, spFlags: 0)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!11 = !DISubprogram(name: "get_return_object", linkageName: "_ZN4Task12promise_type17get_return_objectEv", scope: !3, file: !1, line: 12, type: !12, scopeLine: 12, flags: DIFlagPrototyped, spFlags: 0)
!12 = !DISubroutineType(types: !13)
!13 = !{!4, !10}
!14 = !DISubprogram(name: "initial_suspend", linkageName: "_ZN4Task12promise_type15initial_suspendEv", scope: !3, file: !1, line: 13, type: !15, scopeLine: 13, flags: DIFlagPrototyped, spFlags: 0)
!15 = !DISubroutineType(types: !16)
!16 = !{!17, !10}
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "suspend_never", scope: !19, file: !18, line: 322, size: 8, flags: DIFlagTypePassByValue, elements: !21, identifier: "_ZTSNSt7__n486113suspend_neverE")
!18 = !DIFile(filename: "/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/coroutine", directory: "")
!19 = !DINamespace(name: "__n4861", scope: !20, exportSymbols: true)
!20 = !DINamespace(name: "std", scope: null)
!21 = !{!22, !28, !68}
!22 = !DISubprogram(name: "await_ready", linkageName: "_ZNKSt7__n486113suspend_never11await_readyEv", scope: !17, file: !18, line: 324, type: !23, scopeLine: 324, flags: DIFlagPrototyped, spFlags: 0)
!23 = !DISubroutineType(types: !24)
!24 = !{!25, !26}
!25 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !27, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!27 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !17)
!28 = !DISubprogram(name: "await_suspend", linkageName: "_ZNKSt7__n486113suspend_never13await_suspendENS_16coroutine_handleIvEE", scope: !17, file: !18, line: 326, type: !29, scopeLine: 326, flags: DIFlagPrototyped, spFlags: 0)
!29 = !DISubroutineType(types: !30)
!30 = !{null, !26, !31}
!31 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "coroutine_handle<void>", scope: !19, file: !18, line: 87, size: 64, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !32, templateParams: !66, identifier: "_ZTSNSt7__n486116coroutine_handleIvEE")
!32 = !{!33, !35, !39, !45, !49, !54, !57, !60, !61, !64, !65}
!33 = !DIDerivedType(tag: DW_TAG_member, name: "_M_fr_ptr", scope: !31, file: !18, line: 131, baseType: !34, size: 64, flags: DIFlagProtected)
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!35 = !DISubprogram(name: "coroutine_handle", linkageName: "_ZNSt7__n486116coroutine_handleIvEC4Ev", scope: !31, file: !18, line: 91, type: !36, scopeLine: 91, flags: DIFlagPrototyped, spFlags: 0)
!36 = !DISubroutineType(types: !37)
!37 = !{null, !38}
!38 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !31, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!39 = !DISubprogram(name: "coroutine_handle", linkageName: "_ZNSt7__n486116coroutine_handleIvEC4EDn", scope: !31, file: !18, line: 93, type: !40, scopeLine: 93, flags: DIFlagPrototyped, spFlags: 0)
!40 = !DISubroutineType(types: !41)
!41 = !{null, !38, !42}
!42 = !DIDerivedType(tag: DW_TAG_typedef, name: "nullptr_t", scope: !20, file: !43, line: 2445, baseType: !44)
!43 = !DIFile(filename: "/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/x86_64-redhat-linux/bits/c++config.h", directory: "", checksumkind: CSK_MD5, checksum: "9e5d800a0ad50a6623343c536b5593c0")
!44 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
!45 = !DISubprogram(name: "operator=", linkageName: "_ZNSt7__n486116coroutine_handleIvEaSEDn", scope: !31, file: !18, line: 97, type: !46, scopeLine: 97, flags: DIFlagPrototyped, spFlags: 0)
!46 = !DISubroutineType(types: !47)
!47 = !{!48, !38, !42}
!48 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !31, size: 64)
!49 = !DISubprogram(name: "address", linkageName: "_ZNKSt7__n486116coroutine_handleIvE7addressEv", scope: !31, file: !18, line: 105, type: !50, scopeLine: 105, flags: DIFlagPrototyped, spFlags: 0)
!50 = !DISubroutineType(types: !51)
!51 = !{!34, !52}
!52 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !53, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!53 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !31)
!54 = !DISubprogram(name: "from_address", linkageName: "_ZNSt7__n486116coroutine_handleIvE12from_addressEPv", scope: !31, file: !18, line: 107, type: !55, scopeLine: 107, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!55 = !DISubroutineType(types: !56)
!56 = !{!31, !34}
!57 = !DISubprogram(name: "operator bool", linkageName: "_ZNKSt7__n486116coroutine_handleIvEcvbEv", scope: !31, file: !18, line: 116, type: !58, scopeLine: 116, flags: DIFlagExplicit | DIFlagPrototyped, spFlags: 0)
!58 = !DISubroutineType(types: !59)
!59 = !{!25, !52}
!60 = !DISubprogram(name: "done", linkageName: "_ZNKSt7__n486116coroutine_handleIvE4doneEv", scope: !31, file: !18, line: 121, type: !58, scopeLine: 121, flags: DIFlagPrototyped, spFlags: 0)
!61 = !DISubprogram(name: "operator()", linkageName: "_ZNKSt7__n486116coroutine_handleIvEclEv", scope: !31, file: !18, line: 124, type: !62, scopeLine: 124, flags: DIFlagPrototyped, spFlags: 0)
!62 = !DISubroutineType(types: !63)
!63 = !{null, !52}
!64 = !DISubprogram(name: "resume", linkageName: "_ZNKSt7__n486116coroutine_handleIvE6resumeEv", scope: !31, file: !18, line: 126, type: !62, scopeLine: 126, flags: DIFlagPrototyped, spFlags: 0)
!65 = !DISubprogram(name: "destroy", linkageName: "_ZNKSt7__n486116coroutine_handleIvE7destroyEv", scope: !31, file: !18, line: 128, type: !62, scopeLine: 128, flags: DIFlagPrototyped, spFlags: 0)
!66 = !{!67}
!67 = !DITemplateTypeParameter(name: "_Promise", type: null, defaulted: true)
!68 = !DISubprogram(name: "await_resume", linkageName: "_ZNKSt7__n486113suspend_never12await_resumeEv", scope: !17, file: !18, line: 328, type: !69, scopeLine: 328, flags: DIFlagPrototyped, spFlags: 0)
!69 = !DISubroutineType(types: !70)
!70 = !{null, !26}
!71 = !DISubprogram(name: "final_suspend", linkageName: "_ZN4Task12promise_type13final_suspendEv", scope: !3, file: !1, line: 14, type: !72, scopeLine: 14, flags: DIFlagPrototyped, spFlags: 0)
!72 = !DISubroutineType(types: !73)
!73 = !{!74, !10}
!74 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "suspend_always", scope: !19, file: !18, line: 313, size: 8, flags: DIFlagTypePassByValue, elements: !75, identifier: "_ZTSNSt7__n486114suspend_alwaysE")
!75 = !{!76, !81, !84}
!76 = !DISubprogram(name: "await_ready", linkageName: "_ZNKSt7__n486114suspend_always11await_readyEv", scope: !74, file: !18, line: 315, type: !77, scopeLine: 315, flags: DIFlagPrototyped, spFlags: 0)
!77 = !DISubroutineType(types: !78)
!78 = !{!25, !79}
!79 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !80, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!80 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !74)
!81 = !DISubprogram(name: "await_suspend", linkageName: "_ZNKSt7__n486114suspend_always13await_suspendENS_16coroutine_handleIvEE", scope: !74, file: !18, line: 317, type: !82, scopeLine: 317, flags: DIFlagPrototyped, spFlags: 0)
!82 = !DISubroutineType(types: !83)
!83 = !{null, !79, !31}
!84 = !DISubprogram(name: "await_resume", linkageName: "_ZNKSt7__n486114suspend_always12await_resumeEv", scope: !74, file: !18, line: 319, type: !85, scopeLine: 319, flags: DIFlagPrototyped, spFlags: 0)
!85 = !DISubroutineType(types: !86)
!86 = !{null, !79}
!87 = !DISubprogram(name: "unhandled_exception", linkageName: "_ZN4Task12promise_type19unhandled_exceptionEv", scope: !3, file: !1, line: 15, type: !8, scopeLine: 15, flags: DIFlagPrototyped, spFlags: 0)
!88 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "coroutine_handle<Task::promise_type>", scope: !19, file: !18, line: 182, size: 64, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !89, templateParams: !129, identifier: "_ZTSNSt7__n486116coroutine_handleIN4Task12promise_typeEEE")
!89 = !{!90, !91, !95, !98, !102, !106, !111, !114, !117, !120, !121, !124, !125, !126}
!90 = !DIDerivedType(tag: DW_TAG_member, name: "_M_fr_ptr", scope: !88, file: !18, line: 244, baseType: !34, size: 64, flags: DIFlagPrivate)
!91 = !DISubprogram(name: "coroutine_handle", linkageName: "_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEEC4Ev", scope: !88, file: !18, line: 186, type: !92, scopeLine: 186, flags: DIFlagPrototyped, spFlags: 0)
!92 = !DISubroutineType(types: !93)
!93 = !{null, !94}
!94 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !88, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!95 = !DISubprogram(name: "coroutine_handle", linkageName: "_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEEC4EDn", scope: !88, file: !18, line: 188, type: !96, scopeLine: 188, flags: DIFlagPrototyped, spFlags: 0)
!96 = !DISubroutineType(types: !97)
!97 = !{null, !94, !42}
!98 = !DISubprogram(name: "from_promise", linkageName: "_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEE12from_promiseERS2_", scope: !88, file: !18, line: 191, type: !99, scopeLine: 191, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!99 = !DISubroutineType(types: !100)
!100 = !{!88, !101}
!101 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !3, size: 64)
!102 = !DISubprogram(name: "operator=", linkageName: "_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEEaSEDn", scope: !88, file: !18, line: 199, type: !103, scopeLine: 199, flags: DIFlagPrototyped, spFlags: 0)
!103 = !DISubroutineType(types: !104)
!104 = !{!105, !94, !42}
!105 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !88, size: 64)
!106 = !DISubprogram(name: "address", linkageName: "_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEE7addressEv", scope: !88, file: !18, line: 207, type: !107, scopeLine: 207, flags: DIFlagPrototyped, spFlags: 0)
!107 = !DISubroutineType(types: !108)
!108 = !{!34, !109}
!109 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !110, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!110 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !88)
!111 = !DISubprogram(name: "from_address", linkageName: "_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEE12from_addressEPv", scope: !88, file: !18, line: 209, type: !112, scopeLine: 209, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!112 = !DISubroutineType(types: !113)
!113 = !{!88, !34}
!114 = !DISubprogram(name: "operator coroutine_handle", linkageName: "_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEEcvNS0_IvEEEv", scope: !88, file: !18, line: 217, type: !115, scopeLine: 217, flags: DIFlagPrototyped, spFlags: 0)
!115 = !DISubroutineType(types: !116)
!116 = !{!31, !109}
!117 = !DISubprogram(name: "operator bool", linkageName: "_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEEcvbEv", scope: !88, file: !18, line: 221, type: !118, scopeLine: 221, flags: DIFlagExplicit | DIFlagPrototyped, spFlags: 0)
!118 = !DISubroutineType(types: !119)
!119 = !{!25, !109}
!120 = !DISubprogram(name: "done", linkageName: "_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEE4doneEv", scope: !88, file: !18, line: 226, type: !118, scopeLine: 226, flags: DIFlagPrototyped, spFlags: 0)
!121 = !DISubprogram(name: "operator()", linkageName: "_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEEclEv", scope: !88, file: !18, line: 229, type: !122, scopeLine: 229, flags: DIFlagPrototyped, spFlags: 0)
!122 = !DISubroutineType(types: !123)
!123 = !{null, !109}
!124 = !DISubprogram(name: "resume", linkageName: "_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEE6resumeEv", scope: !88, file: !18, line: 231, type: !122, scopeLine: 231, flags: DIFlagPrototyped, spFlags: 0)
!125 = !DISubprogram(name: "destroy", linkageName: "_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEE7destroyEv", scope: !88, file: !18, line: 233, type: !122, scopeLine: 233, flags: DIFlagPrototyped, spFlags: 0)
!126 = !DISubprogram(name: "promise", linkageName: "_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEE7promiseEv", scope: !88, file: !18, line: 236, type: !127, scopeLine: 236, flags: DIFlagPrototyped, spFlags: 0)
!127 = !DISubroutineType(types: !128)
!128 = !{!101, !109}
!129 = !{!130}
!130 = !DITemplateTypeParameter(name: "_Promise", type: !3)
!131 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "co_sleep", file: !1, line: 2, size: 32, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !132, identifier: "_ZTS8co_sleep")
!132 = !{!133, !135, !139, !144, !147}
!133 = !DIDerivedType(tag: DW_TAG_member, name: "delay", scope: !131, file: !1, line: 7, baseType: !134, size: 32)
!134 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!135 = !DISubprogram(name: "co_sleep", linkageName: "_ZN8co_sleepC4Ei", scope: !131, file: !1, line: 3, type: !136, scopeLine: 3, flags: DIFlagPrototyped, spFlags: 0)
!136 = !DISubroutineType(types: !137)
!137 = !{null, !138, !134}
!138 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !131, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!139 = !DISubprogram(name: "await_ready", linkageName: "_ZNK8co_sleep11await_readyEv", scope: !131, file: !1, line: 4, type: !140, scopeLine: 4, flags: DIFlagPrototyped, spFlags: 0)
!140 = !DISubroutineType(types: !141)
!141 = !{!25, !142}
!142 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !143, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!143 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !131)
!144 = !DISubprogram(name: "await_suspend", linkageName: "_ZNK8co_sleep13await_suspendENSt7__n486116coroutine_handleIvEE", scope: !131, file: !1, line: 5, type: !145, scopeLine: 5, flags: DIFlagPrototyped, spFlags: 0)
!145 = !DISubroutineType(types: !146)
!146 = !{null, !142, !31}
!147 = !DISubprogram(name: "await_resume", linkageName: "_ZNK8co_sleep12await_resumeEv", scope: !131, file: !1, line: 6, type: !148, scopeLine: 6, flags: DIFlagPrototyped, spFlags: 0)
!148 = !DISubroutineType(types: !149)
!149 = !{null, !142}
!150 = !{i32 7, !"Dwarf Version", i32 5}
!151 = !{i32 2, !"Debug Info Version", i32 3}
!152 = !{i32 1, !"wchar_size", i32 4}
!153 = !{i32 7, !"uwtable", i32 2}
!154 = !{i32 7, !"frame-pointer", i32 2}
!155 = !{!"clang version 22.0.0git (https://github.com/llvm/llvm-project.git fe218aab737d55dfd67f4f84118744003f45958e)"}
!156 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 18, type: !157, scopeLine: 18, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !5)
!157 = !DISubroutineType(types: !158)
!158 = !{!4}
!159 = !DILocation(line: 18, column: 21, scope: !156)
!160 = !DILocation(line: 18, column: 6, scope: !156)
!161 = !DILocalVariable(name: "__promise", scope: !156, type: !162, flags: DIFlagArtificial)
!162 = !DIDerivedType(tag: DW_TAG_typedef, name: "promise_type", scope: !163, file: !18, line: 75, baseType: !3)
!163 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__coroutine_traits_impl<Task, void>", scope: !19, file: !18, line: 72, size: 8, flags: DIFlagTypePassByValue, elements: !5, templateParams: !164, identifier: "_ZTSNSt7__n486123__coroutine_traits_implI4TaskvEE")
!164 = !{!165, !166}
!165 = !DITemplateTypeParameter(name: "_Result", type: !4)
!166 = !DITemplateTypeParameter(type: null, defaulted: true)
!167 = !DILocation(line: 0, scope: !156)
!168 = !DILocation(line: 19, column: 12, scope: !169)
!169 = distinct !DILexicalBlock(scope: !156, file: !1, line: 18, column: 21)
!170 = !DILocation(line: 19, column: 3, scope: !169)
!171 = !DILocation(line: 20, column: 1, scope: !169)
!172 = distinct !DISubprogram(name: "get_return_object", linkageName: "_ZN4Task12promise_type17get_return_objectEv", scope: !3, file: !1, line: 12, type: !12, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !11, retainedNodes: !5)
!173 = !DILocalVariable(name: "this", arg: 1, scope: !172, type: !174, flags: DIFlagArtificial | DIFlagObjectPointer)
!174 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64)
!175 = !DILocation(line: 0, scope: !172)
!176 = !DILocation(line: 12, column: 32, scope: !172)
!177 = distinct !DISubprogram(name: "initial_suspend", linkageName: "_ZN4Task12promise_type15initial_suspendEv", scope: !3, file: !1, line: 13, type: !15, scopeLine: 13, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !14, retainedNodes: !5)
!178 = !DILocalVariable(name: "this", arg: 1, scope: !177, type: !174, flags: DIFlagArtificial | DIFlagObjectPointer)
!179 = !DILocation(line: 0, scope: !177)
!180 = !DILocation(line: 13, column: 44, scope: !177)
!181 = distinct !DISubprogram(name: "await_ready", linkageName: "_ZNKSt7__n486113suspend_never11await_readyEv", scope: !17, file: !18, line: 324, type: !23, scopeLine: 324, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !22, retainedNodes: !5)
!182 = !DILocalVariable(name: "this", arg: 1, scope: !181, type: !183, flags: DIFlagArtificial | DIFlagObjectPointer)
!183 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !27, size: 64)
!184 = !DILocation(line: 0, scope: !181)
!185 = !DILocation(line: 324, column: 51, scope: !181)
!186 = distinct !DISubprogram(linkageName: "_Z3foov.__await_suspend_wrapper__init", scope: !1, file: !1, type: !187, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !5)
!187 = !DISubroutineType(types: !5)
!188 = !DILocalVariable(arg: 1, scope: !186, type: !34, flags: DIFlagArtificial)
!189 = !DILocation(line: 0, scope: !186)
!190 = !DILocalVariable(arg: 2, scope: !186, type: !34, flags: DIFlagArtificial)
!191 = !DILocation(line: 18, column: 6, scope: !186)
!192 = distinct !DISubprogram(name: "await_suspend", linkageName: "_ZNKSt7__n486113suspend_never13await_suspendENS_16coroutine_handleIvEE", scope: !17, file: !18, line: 326, type: !29, scopeLine: 326, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !28, retainedNodes: !5)
!193 = !DILocalVariable(name: "this", arg: 1, scope: !192, type: !183, flags: DIFlagArtificial | DIFlagObjectPointer)
!194 = !DILocation(line: 0, scope: !192)
!195 = !DILocalVariable(arg: 2, scope: !192, file: !18, line: 326, type: !31)
!196 = !DILocation(line: 326, column: 52, scope: !192)
!197 = !DILocation(line: 326, column: 70, scope: !192)
!198 = distinct !DISubprogram(name: "from_address", linkageName: "_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEE12from_addressEPv", scope: !88, file: !18, line: 209, type: !112, scopeLine: 210, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !111, retainedNodes: !5)
!199 = !DILocalVariable(name: "__a", arg: 1, scope: !198, file: !18, line: 209, type: !34)
!200 = !DILocation(line: 209, column: 60, scope: !198)
!201 = !DILocalVariable(name: "__self", scope: !198, file: !18, line: 211, type: !88)
!202 = !DILocation(line: 211, column: 19, scope: !198)
!203 = !DILocation(line: 212, column: 21, scope: !198)
!204 = !DILocation(line: 212, column: 9, scope: !198)
!205 = !DILocation(line: 212, column: 19, scope: !198)
!206 = !DILocation(line: 213, column: 2, scope: !198)
!207 = distinct !DISubprogram(name: "operator coroutine_handle", linkageName: "_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEEcvNS0_IvEEEv", scope: !88, file: !18, line: 217, type: !115, scopeLine: 218, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !114, retainedNodes: !5)
!208 = !DILocalVariable(name: "this", arg: 1, scope: !207, type: !209, flags: DIFlagArtificial | DIFlagObjectPointer)
!209 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !110, size: 64)
!210 = !DILocation(line: 0, scope: !207)
!211 = !DILocation(line: 218, column: 49, scope: !207)
!212 = !DILocation(line: 218, column: 16, scope: !207)
!213 = !DILocation(line: 218, column: 9, scope: !207)
!214 = distinct !DISubprogram(name: "await_resume", linkageName: "_ZNKSt7__n486113suspend_never12await_resumeEv", scope: !17, file: !18, line: 328, type: !69, scopeLine: 328, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !68, retainedNodes: !5)
!215 = !DILocalVariable(name: "this", arg: 1, scope: !214, type: !183, flags: DIFlagArtificial | DIFlagObjectPointer)
!216 = !DILocation(line: 0, scope: !214)
!217 = !DILocation(line: 328, column: 51, scope: !214)
!218 = distinct !DISubprogram(name: "co_sleep", linkageName: "_ZN8co_sleepC2Ei", scope: !131, file: !1, line: 3, type: !136, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !135, retainedNodes: !5)
!219 = !DILocalVariable(name: "this", arg: 1, scope: !218, type: !220, flags: DIFlagArtificial | DIFlagObjectPointer)
!220 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !131, size: 64)
!221 = !DILocation(line: 0, scope: !218)
!222 = !DILocalVariable(name: "n", arg: 2, scope: !218, file: !1, line: 3, type: !134)
!223 = !DILocation(line: 3, column: 16, scope: !218)
!224 = !DILocation(line: 3, column: 21, scope: !218)
!225 = !DILocation(line: 3, column: 27, scope: !218)
!226 = !DILocation(line: 3, column: 31, scope: !218)
!227 = distinct !DISubprogram(name: "await_ready", linkageName: "_ZNK8co_sleep11await_readyEv", scope: !131, file: !1, line: 4, type: !140, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !139, retainedNodes: !5)
!228 = !DILocalVariable(name: "this", arg: 1, scope: !227, type: !229, flags: DIFlagArtificial | DIFlagObjectPointer)
!229 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !143, size: 64)
!230 = !DILocation(line: 0, scope: !227)
!231 = !DILocation(line: 4, column: 49, scope: !227)
!232 = distinct !DISubprogram(linkageName: "_Z3foov.__await_suspend_wrapper__await", scope: !1, file: !1, type: !187, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !5)
!233 = !DILocalVariable(arg: 1, scope: !232, type: !34, flags: DIFlagArtificial)
!234 = !DILocation(line: 0, scope: !232)
!235 = !DILocalVariable(arg: 2, scope: !232, type: !34, flags: DIFlagArtificial)
!236 = !DILocation(line: 19, column: 12, scope: !232)
!237 = distinct !DISubprogram(name: "await_suspend", linkageName: "_ZNK8co_sleep13await_suspendENSt7__n486116coroutine_handleIvEE", scope: !131, file: !1, line: 5, type: !145, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !144, retainedNodes: !5)
!238 = !DILocalVariable(name: "this", arg: 1, scope: !237, type: !229, flags: DIFlagArtificial | DIFlagObjectPointer)
!239 = !DILocation(line: 0, scope: !237)
!240 = !DILocalVariable(name: "h", arg: 2, scope: !237, file: !1, line: 5, type: !31)
!241 = !DILocation(line: 5, column: 46, scope: !237)
!242 = !DILocation(line: 5, column: 65, scope: !237)
!243 = distinct !DISubprogram(name: "await_resume", linkageName: "_ZNK8co_sleep12await_resumeEv", scope: !131, file: !1, line: 6, type: !148, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !147, retainedNodes: !5)
!244 = !DILocalVariable(name: "this", arg: 1, scope: !243, type: !229, flags: DIFlagArtificial | DIFlagObjectPointer)
!245 = !DILocation(line: 0, scope: !243)
!246 = !DILocation(line: 6, column: 39, scope: !243)
!247 = distinct !DISubprogram(name: "unhandled_exception", linkageName: "_ZN4Task12promise_type19unhandled_exceptionEv", scope: !3, file: !1, line: 15, type: !8, scopeLine: 15, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !87, retainedNodes: !5)
!248 = !DILocalVariable(name: "this", arg: 1, scope: !247, type: !174, flags: DIFlagArtificial | DIFlagObjectPointer)
!249 = !DILocation(line: 0, scope: !247)
!250 = !DILocation(line: 15, column: 33, scope: !247)
!251 = distinct !DISubprogram(name: "final_suspend", linkageName: "_ZN4Task12promise_type13final_suspendEv", scope: !3, file: !1, line: 14, type: !72, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !71, retainedNodes: !5)
!252 = !DILocalVariable(name: "this", arg: 1, scope: !251, type: !174, flags: DIFlagArtificial | DIFlagObjectPointer)
!253 = !DILocation(line: 0, scope: !251)
!254 = !DILocation(line: 14, column: 52, scope: !251)
!255 = distinct !DISubprogram(name: "await_ready", linkageName: "_ZNKSt7__n486114suspend_always11await_readyEv", scope: !74, file: !18, line: 315, type: !77, scopeLine: 315, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !76, retainedNodes: !5)
!256 = !DILocalVariable(name: "this", arg: 1, scope: !255, type: !257, flags: DIFlagArtificial | DIFlagObjectPointer)
!257 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !80, size: 64)
!258 = !DILocation(line: 0, scope: !255)
!259 = !DILocation(line: 315, column: 51, scope: !255)
!260 = distinct !DISubprogram(linkageName: "_Z3foov.__await_suspend_wrapper__final", scope: !1, file: !1, type: !187, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !5)
!261 = !DILocalVariable(arg: 1, scope: !260, type: !34, flags: DIFlagArtificial)
!262 = !DILocation(line: 0, scope: !260)
!263 = !DILocalVariable(arg: 2, scope: !260, type: !34, flags: DIFlagArtificial)
!264 = !DILocation(line: 18, column: 6, scope: !260)
!265 = distinct !DISubprogram(name: "await_suspend", linkageName: "_ZNKSt7__n486114suspend_always13await_suspendENS_16coroutine_handleIvEE", scope: !74, file: !18, line: 317, type: !82, scopeLine: 317, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !81, retainedNodes: !5)
!266 = !DILocalVariable(name: "this", arg: 1, scope: !265, type: !257, flags: DIFlagArtificial | DIFlagObjectPointer)
!267 = !DILocation(line: 0, scope: !265)
!268 = !DILocalVariable(arg: 2, scope: !265, file: !18, line: 317, type: !31)
!269 = !DILocation(line: 317, column: 52, scope: !265)
!270 = !DILocation(line: 317, column: 70, scope: !265)
!271 = distinct !DISubprogram(name: "await_resume", linkageName: "_ZNKSt7__n486114suspend_always12await_resumeEv", scope: !74, file: !18, line: 319, type: !85, scopeLine: 319, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !84, retainedNodes: !5)
!272 = !DILocalVariable(name: "this", arg: 1, scope: !271, type: !257, flags: DIFlagArtificial | DIFlagObjectPointer)
!273 = !DILocation(line: 0, scope: !271)
!274 = !DILocation(line: 319, column: 51, scope: !271)
!275 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 22, type: !276, scopeLine: 22, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!276 = !DISubroutineType(types: !277)
!277 = !{!134}
!278 = !DILocation(line: 23, column: 3, scope: !275)
!279 = !DILocation(line: 24, column: 1, scope: !275)
!280 = distinct !DISubprogram(name: "coroutine_handle", linkageName: "_ZNSt7__n486116coroutine_handleIN4Task12promise_typeEEC2Ev", scope: !88, file: !18, line: 186, type: !92, scopeLine: 186, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !91, retainedNodes: !5)
!281 = !DILocalVariable(name: "this", arg: 1, scope: !280, type: !282, flags: DIFlagArtificial | DIFlagObjectPointer)
!282 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !88, size: 64)
!283 = !DILocation(line: 0, scope: !280)
!284 = !DILocation(line: 244, column: 13, scope: !280)
!285 = !DILocation(line: 186, column: 47, scope: !280)
!286 = distinct !DISubprogram(name: "from_address", linkageName: "_ZNSt7__n486116coroutine_handleIvE12from_addressEPv", scope: !31, file: !18, line: 107, type: !55, scopeLine: 108, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !54, retainedNodes: !5)
!287 = !DILocalVariable(name: "__a", arg: 1, scope: !286, file: !18, line: 107, type: !34)
!288 = !DILocation(line: 107, column: 60, scope: !286)
!289 = !DILocalVariable(name: "__self", scope: !286, file: !18, line: 109, type: !31)
!290 = !DILocation(line: 109, column: 19, scope: !286)
!291 = !DILocation(line: 110, column: 21, scope: !286)
!292 = !DILocation(line: 110, column: 9, scope: !286)
!293 = !DILocation(line: 110, column: 19, scope: !286)
!294 = !DILocation(line: 111, column: 2, scope: !286)
!295 = distinct !DISubprogram(name: "address", linkageName: "_ZNKSt7__n486116coroutine_handleIN4Task12promise_typeEE7addressEv", scope: !88, file: !18, line: 207, type: !107, scopeLine: 207, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !106, retainedNodes: !5)
!296 = !DILocalVariable(name: "this", arg: 1, scope: !295, type: !209, flags: DIFlagArtificial | DIFlagObjectPointer)
!297 = !DILocation(line: 0, scope: !295)
!298 = !DILocation(line: 207, column: 57, scope: !295)
!299 = !DILocation(line: 207, column: 50, scope: !295)
!300 = distinct !DISubprogram(name: "coroutine_handle", linkageName: "_ZNSt7__n486116coroutine_handleIvEC2Ev", scope: !31, file: !18, line: 91, type: !36, scopeLine: 91, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !35, retainedNodes: !5)
!301 = !DILocalVariable(name: "this", arg: 1, scope: !300, type: !302, flags: DIFlagArtificial | DIFlagObjectPointer)
!302 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !31, size: 64)
!303 = !DILocation(line: 0, scope: !300)
!304 = !DILocation(line: 91, column: 47, scope: !300)
!305 = !DILocation(line: 91, column: 61, scope: !300)
