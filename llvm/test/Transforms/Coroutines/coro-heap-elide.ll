; Tests that the dynamic allocation and deallocation of the coroutine frame is
; elided and any tail calls referencing the coroutine frame has the tail
; call attribute removed.
; RUN: opt < %s -S \
; RUN: -passes='cgscc(inline,function(coro-elide,instsimplify,simplifycfg))' \
; RUN:   -aa-pipeline='basic-aa' | FileCheck %s

declare void @print(i32) nounwind

%f.frame = type {i32}

declare void @bar(ptr)

declare fastcc void @f.resume(ptr align 4 dereferenceable(4))
declare fastcc void @f.destroy(ptr)
declare fastcc void @f.cleanup(ptr)

declare void @may_throw()
declare ptr @CustomAlloc(i32)
declare void @CustomFree(ptr)

@f.resumers = internal constant [3 x ptr]
  [ptr @f.resume, ptr @f.destroy, ptr @f.cleanup]

; a coroutine start function
define ptr @f() personality ptr null {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null,
                      ptr @f,
                      ptr @f.resumers)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %alloc = call ptr @CustomAlloc(i32 4)
  br label %coro.begin
coro.begin:
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %phi)
  invoke void @may_throw()
    to label %ret unwind label %ehcleanup
ret:
  ret ptr %hdl

ehcleanup:
  %tok = cleanuppad within none []
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  %need.dyn.free = icmp ne ptr %mem, null
  br i1 %need.dyn.free, label %dyn.free, label %if.end
dyn.free:
  call void @CustomFree(ptr %mem)
  br label %if.end
if.end:
  cleanupret from %tok unwind to caller
}

; CHECK-LABEL: @callResume(
define void @callResume() {
entry:
; CHECK: alloca [4 x i8], align 4
; CHECK-NOT: coro.begin
; CHECK-NOT: CustomAlloc
; CHECK: call void @may_throw()
  %hdl = call ptr @f()

; Need to remove 'tail' from the first call to @bar
; CHECK-NOT: tail call void @bar(
; CHECK: call void @bar(
  tail call void @bar(ptr %hdl)
; CHECK: tail call void @bar(
  tail call void @bar(ptr null)

; CHECK-NEXT: call fastcc void @f.resume(ptr %0)
  %0 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  call fastcc void %0(ptr %hdl)

; CHECK-NEXT: call fastcc void @f.cleanup(ptr %0)
  %1 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %1(ptr %hdl)

; CHECK-NEXT: ret void
  ret void
}

; CHECK-LABEL: @callResume_with_coro_suspend_1(
define void @callResume_with_coro_suspend_1() {
entry:
; CHECK: alloca [4 x i8], align 4
; CHECK-NOT: coro.begin
; CHECK-NOT: CustomAlloc
; CHECK: call void @may_throw()
  %hdl = call ptr @f()

; CHECK-NEXT: call fastcc void @f.resume(ptr %0)
  %0 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  call fastcc void %0(ptr %hdl)
  %1 = call token @llvm.coro.save(ptr %hdl)
  %2 = call i8 @llvm.coro.suspend(token %1, i1 false)
  switch i8 %2, label  %coro.ret [
    i8 0, label %final.suspend
    i8 1, label %cleanups
  ]

; CHECK-LABEL: final.suspend:
final.suspend:
; CHECK-NEXT: call fastcc void @f.cleanup(ptr %0)
  %3 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %3(ptr %hdl)
  %4 = call token @llvm.coro.save(ptr %hdl)
  %5 = call i8 @llvm.coro.suspend(token %4, i1 true)
  switch i8 %5, label  %coro.ret [
    i8 0, label %coro.ret
    i8 1, label %cleanups
  ]

; CHECK-LABEL: cleanups:
cleanups:
; CHECK-NEXT: call fastcc void @f.cleanup(ptr %0)
  %6 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %6(ptr %hdl)
  br label %coro.ret

; CHECK-LABEL: coro.ret:
coro.ret:
; CHECK-NEXT: ret void
  ret void
}

; CHECK-LABEL: @callResume_with_coro_suspend_2(
define void @callResume_with_coro_suspend_2() personality ptr null {
entry:
; CHECK: alloca [4 x i8], align 4
; CHECK-NOT: coro.begin
; CHECK-NOT: CustomAlloc
; CHECK: call void @may_throw()
  %hdl = call ptr @f()

  %0 = call token @llvm.coro.save(ptr %hdl)
; CHECK: invoke fastcc void @f.resume(ptr %0)
  %1 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  invoke fastcc void %1(ptr %hdl)
    to label %invoke.cont1 unwind label %lpad

; CHECK-LABEL: invoke.cont1:
invoke.cont1:
  %2 = call i8 @llvm.coro.suspend(token %0, i1 false)
  switch i8 %2, label  %coro.ret [
    i8 0, label %final.ready
    i8 1, label %cleanups
  ]

; CHECK-LABEL: lpad:
lpad:
  %3 = landingpad { ptr, i32 }
          catch ptr null
; CHECK: call fastcc void @f.cleanup(ptr %0)
  %4 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %4(ptr %hdl)
  br label %final.suspend

; CHECK-LABEL: final.ready:
final.ready:
; CHECK-NEXT: call fastcc void @f.cleanup(ptr %0)
  %5 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %5(ptr %hdl)
  br label %final.suspend

; CHECK-LABEL: final.suspend:
final.suspend:
  %6 = call token @llvm.coro.save(ptr %hdl)
  %7 = call i8 @llvm.coro.suspend(token %6, i1 true)
  switch i8 %7, label  %coro.ret [
    i8 0, label %coro.ret
    i8 1, label %cleanups
  ]

; CHECK-LABEL: cleanups:
cleanups:
; CHECK-NEXT: call fastcc void @f.cleanup(ptr %0)
  %8 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %8(ptr %hdl)
  br label %coro.ret

; CHECK-LABEL: coro.ret:
coro.ret:
; CHECK-NEXT: ret void
  ret void
}

; CHECK-LABEL: @callResume_with_coro_suspend_3(
define void @callResume_with_coro_suspend_3(i8 %cond) {
entry:
; CHECK: alloca [4 x i8], align 4
  switch i8 %cond, label  %coro.ret [
    i8 0, label %init.suspend
    i8 1, label %coro.ret
  ]

init.suspend:
; CHECK-NOT: llvm.coro.begin
; CHECK-NOT: CustomAlloc
; CHECK: call void @may_throw()
  %hdl = call ptr @f()
; CHECK-NEXT: call fastcc void @f.resume(ptr %0)
  %0 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  call fastcc void %0(ptr %hdl)
  %1 = call token @llvm.coro.save(ptr %hdl)
  %2 = call i8 @llvm.coro.suspend(token %1, i1 false)
  switch i8 %2, label  %coro.ret [
    i8 0, label %final.suspend
    i8 1, label %cleanups
  ]

; CHECK-LABEL: final.suspend:
final.suspend:
; CHECK-NEXT: call fastcc void @f.cleanup(ptr %0)
  %3 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %3(ptr %hdl)
  %4 = call token @llvm.coro.save(ptr %hdl)
  %5 = call i8 @llvm.coro.suspend(token %4, i1 true)
  switch i8 %5, label  %coro.ret [
    i8 0, label %coro.ret
    i8 1, label %cleanups
  ]

; CHECK-LABEL: cleanups:
cleanups:
; CHECK-NEXT: call fastcc void @f.cleanup(ptr %0)
  %6 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %6(ptr %hdl)
  br label %coro.ret

; CHECK-LABEL: coro.ret:
coro.ret:
; CHECK-NEXT: ret void
  ret void
}



; CHECK-LABEL: @callResume_PR34897_no_elision(
define void @callResume_PR34897_no_elision(i1 %cond) {
; CHECK-LABEL: entry:
entry:
; CHECK: call ptr @CustomAlloc(
  %hdl = call ptr @f()
; CHECK: tail call void @bar(
  tail call void @bar(ptr %hdl)
; CHECK: tail call void @bar(
  tail call void @bar(ptr null)
  br i1 %cond, label %if.then, label %if.else

; CHECK-LABEL: if.then:
if.then:
; CHECK: call fastcc void @f.resume(ptr
  %0 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  call fastcc void %0(ptr %hdl)
; CHECK-NEXT: call fastcc void @f.destroy(ptr
  %1 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %1(ptr %hdl)
  br label %return

if.else:
  br label %return

; CHECK-LABEL: return:
return:
; CHECK: ret void
  ret void
}

; CHECK-LABEL: @callResume_PR34897_elision(
define void @callResume_PR34897_elision(i1 %cond) {
; CHECK-LABEL: entry:
entry:
; CHECK: alloca [4 x i8], align 4
; CHECK: tail call void @bar(
  tail call void @bar(ptr null)
  br i1 %cond, label %if.then, label %if.else

if.then:
; CHECK-NOT: CustomAlloc
; CHECK: call void @may_throw()
  %hdl = call ptr @f()
; CHECK: call void @bar(
  tail call void @bar(ptr %hdl)
; CHECK: call fastcc void @f.resume(ptr
  %0 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  call fastcc void %0(ptr %hdl)
; CHECK-NEXT: call fastcc void @f.cleanup(ptr
  %1 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %1(ptr %hdl)
  br label %return

if.else:
  br label %return

; CHECK-LABEL: return:
return:
; CHECK: ret void
  ret void
}


; a coroutine start function (cannot elide heap alloc, due to second argument to
; coro.begin not pointint to coro.alloc)
define ptr @f_no_elision() personality ptr null {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null,
                      ptr @f_no_elision,
                      ptr @f.resumers)
  %alloc = call ptr @CustomAlloc(i32 4)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  ret ptr %hdl
}

; CHECK-LABEL: @callResume_no_elision(
define void @callResume_no_elision() {
entry:
; CHECK: call ptr @CustomAlloc(
  %hdl = call ptr @f_no_elision()

; Tail call should remain tail calls
; CHECK: tail call void @bar(
  tail call void @bar(ptr %hdl)
; CHECK: tail call void @bar(
  tail call void @bar(ptr null)

; CHECK-NEXT: call fastcc void @f.resume(ptr
  %0 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  call fastcc void %0(ptr %hdl)

; CHECK-NEXT: call fastcc void @f.destroy(ptr
  %1 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %1(ptr %hdl)

; CHECK-NEXT: ret void
  ret void
}

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.free(token, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare ptr @llvm.coro.frame(token)
declare ptr @llvm.coro.subfn.addr(ptr, i8)
declare i8 @llvm.coro.suspend(token, i1)
declare token @llvm.coro.save(ptr)
