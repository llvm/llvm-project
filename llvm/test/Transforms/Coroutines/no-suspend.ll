; Test no suspend coroutines
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse,simplifycfg' -S | FileCheck %s

; Coroutine with no-suspends will turn into:
;
; CHECK-LABEL: define void @no_suspends(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    alloca
; CHECK-NEXT:    bitcast
; CHECK-NEXT:    call void @print(i32 %n)
; CHECK-NEXT:    ret void
;
define void @no_suspends(i32 %n) "coroutine.presplit"="1" {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi i8* [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %phi)
  br label %body
body:
  call void @print(i32 %n)
  br label %cleanup
cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  %need.dyn.free = icmp ne i8* %mem, null
  br i1 %need.dyn.free, label %dyn.free, label %suspend
dyn.free:
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 false)
  ret void
}

; SimplifySuspendPoint will detect that coro.resume resumes itself and will
; replace suspend with a jump to %resume label turning it into no-suspend
; coroutine.
;
; CHECK-LABEL: define void @simplify_resume(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    alloca
; CHECK-NEXT:    bitcast
; CHECK-NEXT:    call void @llvm.memcpy
; CHECK-NEXT:    call void @print(i32 0)
; CHECK-NEXT:    ret void
;
define void @simplify_resume(i8* %src, i8* %dst) "coroutine.presplit"="1" {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi i8* [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %phi)
  br label %body
body:
  %save = call token @llvm.coro.save(i8* %hdl)
  ; memcpy intrinsics should not prevent simplification.
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 1, i1 false)
  %subfn = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 0)
  %bres = bitcast i8* %subfn to void (i8*)*
  call fastcc void %bres(i8* %hdl)
  %0 = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %pre.cleanup]
resume:
  call void @print(i32 0)
  br label %cleanup

pre.cleanup:
  call void @print(i32 1)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 false)
  ret void
}

; SimplifySuspendPoint will detect that coroutine destroys itself and will
; replace suspend with a jump to %cleanup label turning it into no-suspend
; coroutine.
;
; CHECK-LABEL: define void @simplify_destroy(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    alloca
; CHECK-NEXT:    bitcast
; CHECK-NEXT:    call void @print(i32 1)
; CHECK-NEXT:    ret void
;
define void @simplify_destroy() "coroutine.presplit"="1" personality i32 0 {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi i8* [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %phi)
  br label %body
body:
  %save = call token @llvm.coro.save(i8* %hdl)
  %subfn = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %bcast = bitcast i8* %subfn to void (i8*)*
  invoke fastcc void %bcast(i8* %hdl) to label %real_susp unwind label %lpad

real_susp:
  %0 = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %pre.cleanup]
resume:
  call void @print(i32 0)
  br label %cleanup

pre.cleanup:
  call void @print(i32 1)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 false)
  ret void
lpad:
  %lpval = landingpad { i8*, i32 }
     cleanup

  call void @print(i32 2)
  resume { i8*, i32 } %lpval
}

; SimplifySuspendPoint will detect that coro.resume resumes itself and will
; replace suspend with a jump to %resume label turning it into no-suspend
; coroutine.
;
; CHECK-LABEL: define void @simplify_resume_with_inlined_if(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    alloca
; CHECK-NEXT:    bitcast
; CHECK-NEXT:    br i1
; CHECK:         call void @print(i32 0)
; CHECK-NEXT:    ret void
;
define void @simplify_resume_with_inlined_if(i8* %src, i8* %dst, i1 %cond) "coroutine.presplit"="1" {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi i8* [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %phi)
  br label %body
body:
  %save = call token @llvm.coro.save(i8* %hdl)
  br i1 %cond, label %if.then, label %if.else
if.then:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 1, i1 false)
  br label %if.end
if.else:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %src, i8* %dst, i64 1, i1 false)
  br label %if.end
if.end:
  %subfn = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 0)
  %bres = bitcast i8* %subfn to void (i8*)*
  call fastcc void %bres(i8* %hdl)
  %0 = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %pre.cleanup]
resume:
  call void @print(i32 0)
  br label %cleanup

pre.cleanup:
  call void @print(i32 1)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 false)
  ret void
}



; SimplifySuspendPoint won't be able to simplify if it detects that there are
; other calls between coro.save and coro.suspend. They potentially can call
; resume or destroy, so we should not simplify this suspend point.
;
; CHECK-LABEL: define void @cannot_simplify_other_calls(
; CHECK-NEXT:  entry:
; CHECK-NEXT:     llvm.coro.id

define void @cannot_simplify_other_calls() "coroutine.presplit"="1" {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi i8* [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %phi)
  br label %body
body:
  %save = call token @llvm.coro.save(i8* %hdl)
  br label %body1

body1:
  call void @foo()
  br label %body2

body2:
  %subfn = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %bcast = bitcast i8* %subfn to void (i8*)*
  call fastcc void %bcast(i8* %hdl)
  %0 = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %pre.cleanup]
resume:
  call void @print(i32 0)
  br label %cleanup

pre.cleanup:
  call void @print(i32 1)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 false)
  ret void
}

; SimplifySuspendPoint won't be able to simplify if it detects that there are
; other calls between coro.save and coro.suspend. They potentially can call
; resume or destroy, so we should not simplify this suspend point.
;
; CHECK-LABEL: define void @cannot_simplify_calls_in_terminator(
; CHECK-NEXT:  entry:
; CHECK-NEXT:     llvm.coro.id

define void @cannot_simplify_calls_in_terminator() "coroutine.presplit"="1" personality i32 0 {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi i8* [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %phi)
  br label %body
body:
  %save = call token @llvm.coro.save(i8* %hdl)
  invoke void @foo() to label %resume_cont unwind label %lpad
resume_cont:
  %subfn = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %bcast = bitcast i8* %subfn to void (i8*)*
  call fastcc void %bcast(i8* %hdl)
  %0 = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %pre.cleanup]
resume:
  call void @print(i32 0)
  br label %cleanup

pre.cleanup:
  call void @print(i32 1)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 false)
  ret void
lpad:
  %lpval = landingpad { i8*, i32 }
     cleanup

  call void @print(i32 2)
  resume { i8*, i32 } %lpval
}

; SimplifySuspendPoint won't be able to simplify if it detects that resume or
; destroy does not immediately preceed coro.suspend.
;
; CHECK-LABEL: define void @cannot_simplify_not_last_instr(
; CHECK-NEXT:  entry:
; CHECK-NEXT:     llvm.coro.id

define void @cannot_simplify_not_last_instr(i8* %dst, i8* %src) "coroutine.presplit"="1" {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi i8* [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %phi)
  br label %body
body:
  %save = call token @llvm.coro.save(i8* %hdl)
  %subfn = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %bcast = bitcast i8* %subfn to void (i8*)*
  call fastcc void %bcast(i8* %hdl)
  ; memcpy separates destory from suspend, therefore cannot simplify.
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 1, i1 false)
  %0 = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %pre.cleanup]
resume:
  call void @print(i32 0)
  br label %cleanup

pre.cleanup:
  call void @print(i32 1)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 false)
  ret void
}

; SimplifySuspendPoint should not simplify final suspend point
;
; CHECK-LABEL: define void @cannot_simplify_final_suspend(
; CHECK-NEXT:  entry:
; CHECK-NEXT:     llvm.coro.id
;
define void @cannot_simplify_final_suspend() "coroutine.presplit"="1" personality i32 0 {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi i8* [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %phi)
  br label %body
body:
  %save = call token @llvm.coro.save(i8* %hdl)
  %subfn = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %bcast = bitcast i8* %subfn to void (i8*)*
  invoke fastcc void %bcast(i8* %hdl) to label %real_susp unwind label %lpad

real_susp:
  %0 = call i8 @llvm.coro.suspend(token %save, i1 1)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %pre.cleanup]
resume:
  call void @print(i32 0)
  br label %cleanup

pre.cleanup:
  call void @print(i32 1)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 false)
  ret void
lpad:
  %lpval = landingpad { i8*, i32 }
     cleanup

  call void @print(i32 2)
  resume { i8*, i32 } %lpval
}

declare i8* @malloc(i32)
declare void @free(i8*) willreturn
declare void @print(i32)
declare void @foo()

declare token @llvm.coro.id(i32, i8*, i8*, i8*)
declare i1 @llvm.coro.alloc(token)
declare i32 @llvm.coro.size.i32()
declare i8* @llvm.coro.begin(token, i8*)
declare token @llvm.coro.save(i8* %hdl)
declare i8 @llvm.coro.suspend(token, i1)
declare i8* @llvm.coro.free(token, i8*)
declare i1 @llvm.coro.end(i8*, i1)

declare i8* @llvm.coro.subfn.addr(i8*, i8)

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)
