; Tests that the dynamic allocation and deallocation of the coroutine frame is
; elided and any tail calls referencing the coroutine frame has the tail 
; call attribute removed.
; RUN: opt < %s -S \
; RUN:   -passes='cgscc(inline,function(coro-elide,instsimplify,simplifycfg))' \
; RUN:   -aa-pipeline='basic-aa' | FileCheck %s

declare void @print(i32) nounwind

%f.frame = type {i32}

declare void @bar(i8*)

declare fastcc void @f.resume(%f.frame*)
declare fastcc void @f.destroy(%f.frame*)
declare fastcc void @f.cleanup(%f.frame*)

declare void @may_throw()
declare i8* @CustomAlloc(i32)
declare void @CustomFree(i8*)

@f.resumers = internal constant [3 x void (%f.frame*)*] 
  [void (%f.frame*)* @f.resume, void (%f.frame*)* @f.destroy, void (%f.frame*)* @f.cleanup]

; a coroutine start function
define i8* @f() personality i8* null {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null,
                      i8* bitcast (i8*()* @f to i8*),
                      i8* bitcast ([3 x void (%f.frame*)*]* @f.resumers to i8*))
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %alloc = call i8* @CustomAlloc(i32 4)
  br label %coro.begin
coro.begin:
  %phi = phi i8* [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %phi)
  invoke void @may_throw() 
    to label %ret unwind label %ehcleanup
ret:          
  ret i8* %hdl

ehcleanup:
  %tok = cleanuppad within none []
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  %need.dyn.free = icmp ne i8* %mem, null
  br i1 %need.dyn.free, label %dyn.free, label %if.end
dyn.free:
  call void @CustomFree(i8* %mem)
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
  %hdl = call i8* @f()

; Need to remove 'tail' from the first call to @bar
; CHECK-NOT: tail call void @bar(
; CHECK: call void @bar(
  tail call void @bar(i8* %hdl)
; CHECK: tail call void @bar(
  tail call void @bar(i8* null)

; CHECK-NEXT: call fastcc void bitcast (void (%f.frame*)* @f.resume to void (i8*)*)(i8* %vFrame)
  %0 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 0)
  %1 = bitcast i8* %0 to void (i8*)*
  call fastcc void %1(i8* %hdl)

; CHECK-NEXT: call fastcc void bitcast (void (%f.frame*)* @f.cleanup to void (i8*)*)(i8* %vFrame)
  %2 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %3 = bitcast i8* %2 to void (i8*)*
  call fastcc void %3(i8* %hdl)

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
  %hdl = call i8* @f()

; CHECK-NEXT: call fastcc void bitcast (void (%f.frame*)* @f.resume to void (i8*)*)(i8* %vFrame)
  %0 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 0)
  %1 = bitcast i8* %0 to void (i8*)*
  call fastcc void %1(i8* %hdl)
  %2 = call token @llvm.coro.save(i8* %hdl)
  %3 = call i8 @llvm.coro.suspend(token %2, i1 false)
  switch i8 %3, label  %coro.ret [
    i8 0, label %final.suspend
    i8 1, label %cleanups
  ]

; CHECK-LABEL: final.suspend:
final.suspend:
; CHECK-NEXT: call fastcc void bitcast (void (%f.frame*)* @f.cleanup to void (i8*)*)(i8* %vFrame)
  %4 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %5 = bitcast i8* %4 to void (i8*)*
  call fastcc void %5(i8* %hdl)
  %6 = call token @llvm.coro.save(i8* %hdl)
  %7 = call i8 @llvm.coro.suspend(token %6, i1 true)
  switch i8 %7, label  %coro.ret [
    i8 0, label %coro.ret
    i8 1, label %cleanups
  ]

; CHECK-LABEL: cleanups:
cleanups:
; CHECK-NEXT: call fastcc void bitcast (void (%f.frame*)* @f.cleanup to void (i8*)*)(i8* %vFrame)
  %8 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %9 = bitcast i8* %8 to void (i8*)*
  call fastcc void %9(i8* %hdl)
  br label %coro.ret

; CHECK-LABEL: coro.ret:
coro.ret:
; CHECK-NEXT: ret void
  ret void
}

; CHECK-LABEL: @callResume_with_coro_suspend_2(
define void @callResume_with_coro_suspend_2() personality i8* null {
entry:
; CHECK: alloca [4 x i8], align 4
; CHECK-NOT: coro.begin
; CHECK-NOT: CustomAlloc
; CHECK: call void @may_throw()
  %hdl = call i8* @f()

  %0 = call token @llvm.coro.save(i8* %hdl)
; CHECK: invoke fastcc void bitcast (void (%f.frame*)* @f.resume to void (i8*)*)(i8* %vFrame)
  %1 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 0)
  %2 = bitcast i8* %1 to void (i8*)*
  invoke fastcc void %2(i8* %hdl)
    to label %invoke.cont1 unwind label %lpad

; CHECK-LABEL: invoke.cont1:
invoke.cont1:
  %3 = call i8 @llvm.coro.suspend(token %0, i1 false)
  switch i8 %3, label  %coro.ret [
    i8 0, label %final.ready
    i8 1, label %cleanups
  ]

; CHECK-LABEL: lpad:
lpad:
  %4 = landingpad { i8*, i32 }
          catch i8* null
; CHECK: call fastcc void bitcast (void (%f.frame*)* @f.cleanup to void (i8*)*)(i8* %vFrame)
  %5 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %6 = bitcast i8* %5 to void (i8*)*
  call fastcc void %6(i8* %hdl)
  br label %final.suspend

; CHECK-LABEL: final.ready:
final.ready:
; CHECK-NEXT: call fastcc void bitcast (void (%f.frame*)* @f.cleanup to void (i8*)*)(i8* %vFrame)
  %7 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %8 = bitcast i8* %7 to void (i8*)*
  call fastcc void %8(i8* %hdl)
  br label %final.suspend

; CHECK-LABEL: final.suspend:
final.suspend:
  %9 = call token @llvm.coro.save(i8* %hdl)
  %10 = call i8 @llvm.coro.suspend(token %9, i1 true)
  switch i8 %10, label  %coro.ret [
    i8 0, label %coro.ret
    i8 1, label %cleanups
  ]

; CHECK-LABEL: cleanups:
cleanups:
; CHECK-NEXT: call fastcc void bitcast (void (%f.frame*)* @f.cleanup to void (i8*)*)(i8* %vFrame)
  %11 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %12 = bitcast i8* %11 to void (i8*)*
  call fastcc void %12(i8* %hdl)
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
  %hdl = call i8* @f()
; CHECK-NEXT: call fastcc void bitcast (void (%f.frame*)* @f.resume to void (i8*)*)(i8* %vFrame)
  %0 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 0)
  %1 = bitcast i8* %0 to void (i8*)*
  call fastcc void %1(i8* %hdl)
  %2 = call token @llvm.coro.save(i8* %hdl)
  %3 = call i8 @llvm.coro.suspend(token %2, i1 false)
  switch i8 %3, label  %coro.ret [
    i8 0, label %final.suspend
    i8 1, label %cleanups
  ]

; CHECK-LABEL: final.suspend:
final.suspend:
; CHECK-NEXT: call fastcc void bitcast (void (%f.frame*)* @f.cleanup to void (i8*)*)(i8* %vFrame)
  %4 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %5 = bitcast i8* %4 to void (i8*)*
  call fastcc void %5(i8* %hdl)
  %6 = call token @llvm.coro.save(i8* %hdl)
  %7 = call i8 @llvm.coro.suspend(token %6, i1 true)
  switch i8 %7, label  %coro.ret [
    i8 0, label %coro.ret
    i8 1, label %cleanups
  ]

; CHECK-LABEL: cleanups:
cleanups:
; CHECK-NEXT: call fastcc void bitcast (void (%f.frame*)* @f.cleanup to void (i8*)*)(i8* %vFrame)
  %8 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %9 = bitcast i8* %8 to void (i8*)*
  call fastcc void %9(i8* %hdl)
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
; CHECK: call i8* @CustomAlloc(
  %hdl = call i8* @f()
; CHECK: tail call void @bar(
  tail call void @bar(i8* %hdl)
; CHECK: tail call void @bar(
  tail call void @bar(i8* null)
  br i1 %cond, label %if.then, label %if.else

; CHECK-LABEL: if.then:
if.then:
; CHECK: call fastcc void bitcast (void (%f.frame*)* @f.resume to void (i8*)*)(i8*
  %0 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 0)
  %1 = bitcast i8* %0 to void (i8*)*
  call fastcc void %1(i8* %hdl)
; CHECK-NEXT: call fastcc void bitcast (void (%f.frame*)* @f.destroy to void (i8*)*)(i8*
  %2 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %3 = bitcast i8* %2 to void (i8*)*
  call fastcc void %3(i8* %hdl)
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
  tail call void @bar(i8* null)
  br i1 %cond, label %if.then, label %if.else

if.then:
; CHECK-NOT: CustomAlloc
; CHECK: call void @may_throw()
  %hdl = call i8* @f()
; CHECK: call void @bar(
  tail call void @bar(i8* %hdl)
; CHECK: call fastcc void bitcast (void (%f.frame*)* @f.resume to void (i8*)*)(i8*
  %0 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 0)
  %1 = bitcast i8* %0 to void (i8*)*
  call fastcc void %1(i8* %hdl)
; CHECK-NEXT: call fastcc void bitcast (void (%f.frame*)* @f.cleanup to void (i8*)*)(i8*
  %2 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %3 = bitcast i8* %2 to void (i8*)*
  call fastcc void %3(i8* %hdl)
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
define i8* @f_no_elision() personality i8* null {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null,
                      i8* bitcast (i8*()* @f_no_elision to i8*),
                      i8* bitcast ([3 x void (%f.frame*)*]* @f.resumers to i8*))
  %alloc = call i8* @CustomAlloc(i32 4)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)
  ret i8* %hdl
}

; CHECK-LABEL: @callResume_no_elision(
define void @callResume_no_elision() {
entry:
; CHECK: call i8* @CustomAlloc(
  %hdl = call i8* @f_no_elision()

; Tail call should remain tail calls
; CHECK: tail call void @bar(
  tail call void @bar(i8* %hdl)
; CHECK: tail call void @bar(  
  tail call void @bar(i8* null)

; CHECK-NEXT: call fastcc void bitcast (void (%f.frame*)* @f.resume to void (i8*)*)(i8*
  %0 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 0)
  %1 = bitcast i8* %0 to void (i8*)*
  call fastcc void %1(i8* %hdl)

; CHECK-NEXT: call fastcc void bitcast (void (%f.frame*)* @f.destroy to void (i8*)*)(i8*
  %2 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %3 = bitcast i8* %2 to void (i8*)*
  call fastcc void %3(i8* %hdl)

; CHECK-NEXT: ret void
  ret void
}

declare token @llvm.coro.id(i32, i8*, i8*, i8*)
declare i1 @llvm.coro.alloc(token)
declare i8* @llvm.coro.free(token, i8*)
declare i8* @llvm.coro.begin(token, i8*)
declare i8* @llvm.coro.frame(token)
declare i8* @llvm.coro.subfn.addr(i8*, i8)
declare i8 @llvm.coro.suspend(token, i1)
declare token @llvm.coro.save(i8*)
