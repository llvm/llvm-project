; Test no suspend coroutines
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse,simplifycfg' -S | FileCheck %s

; Coroutine with no-suspends will turn into:
;
; CHECK-LABEL: define void @no_suspends(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @print(i32 %n)
; CHECK-NEXT:    ret void
;
define void @no_suspends(i32 %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias ptr @llvm.coro.begin(token %id, ptr %phi)
  br label %body
body:
  call void @print(i32 %n)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  %need.dyn.free = icmp ne ptr %mem, null
  br i1 %need.dyn.free, label %dyn.free, label %suspend
dyn.free:
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret void
}

; SimplifySuspendPoint will detect that coro.resume resumes itself and will
; replace suspend with a jump to %resume label turning it into no-suspend
; coroutine.
;
; CHECK-LABEL: define void @simplify_resume(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.memcpy
; CHECK-NEXT:    call void @print(i32 0)
; CHECK-NEXT:    ret void
;
define void @simplify_resume(ptr %src, ptr %dst) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias ptr @llvm.coro.begin(token %id, ptr %phi)
  br label %body
body:
  %save = call token @llvm.coro.save(ptr %hdl)
  ; memcpy intrinsics should not prevent simplification.
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 1, i1 false)
  %subfn = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  call fastcc void %subfn(ptr %hdl)
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
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret void
}

; SimplifySuspendPoint will detect that coroutine destroys itself and will
; replace suspend with a jump to %cleanup label turning it into no-suspend
; coroutine.
;
; CHECK-LABEL: define void @simplify_destroy(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @print(i32 1)
; CHECK-NEXT:    ret void
;
define void @simplify_destroy() presplitcoroutine personality i32 0 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias ptr @llvm.coro.begin(token %id, ptr %phi)
  br label %body
body:
  %save = call token @llvm.coro.save(ptr %hdl)
  %subfn = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  invoke fastcc void %subfn(ptr %hdl) to label %real_susp unwind label %lpad

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
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret void
lpad:
  %lpval = landingpad { ptr, i32 }
     cleanup

  call void @print(i32 2)
  resume { ptr, i32 } %lpval
}

; SimplifySuspendPoint will detect that coro.resume resumes itself and will
; replace suspend with a jump to %resume label turning it into no-suspend
; coroutine.
;
; CHECK-LABEL: define void @simplify_resume_with_inlined_if(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1
; CHECK:         call void @print(i32 0)
; CHECK-NEXT:    ret void
;
define void @simplify_resume_with_inlined_if(ptr %src, ptr %dst, i1 %cond) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias ptr @llvm.coro.begin(token %id, ptr %phi)
  br label %body
body:
  %save = call token @llvm.coro.save(ptr %hdl)
  br i1 %cond, label %if.then, label %if.else
if.then:
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 1, i1 false)
  br label %if.end
if.else:
  call void @llvm.memcpy.p0.p0.i64(ptr %src, ptr %dst, i64 1, i1 false)
  br label %if.end
if.end:
  %subfn = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  call fastcc void %subfn(ptr %hdl)
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
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret void
}



; SimplifySuspendPoint won't be able to simplify if it detects that there are
; other calls between coro.save and coro.suspend. They potentially can call
; resume or destroy, so we should not simplify this suspend point.
;
; CHECK-LABEL: define void @cannot_simplify_other_calls(
; CHECK-NEXT:  entry:
; CHECK-NEXT:     llvm.coro.id

define void @cannot_simplify_other_calls() presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias ptr @llvm.coro.begin(token %id, ptr %phi)
  br label %body
body:
  %save = call token @llvm.coro.save(ptr %hdl)
  br label %body1

body1:
  call void @foo()
  br label %body2

body2:
  %subfn = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %subfn(ptr %hdl)
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
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret void
}

; SimplifySuspendPoint won't be able to simplify if it detects that there are
; other calls between coro.save and coro.suspend. They potentially can call
; resume or destroy, so we should not simplify this suspend point.
;
; CHECK-LABEL: define void @cannot_simplify_calls_in_terminator(
; CHECK-NEXT:  entry:
; CHECK-NEXT:     llvm.coro.id

define void @cannot_simplify_calls_in_terminator() presplitcoroutine personality i32 0 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias ptr @llvm.coro.begin(token %id, ptr %phi)
  br label %body
body:
  %save = call token @llvm.coro.save(ptr %hdl)
  invoke void @foo() to label %resume_cont unwind label %lpad
resume_cont:
  %subfn = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %subfn(ptr %hdl)
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
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret void
lpad:
  %lpval = landingpad { ptr, i32 }
     cleanup

  call void @print(i32 2)
  resume { ptr, i32 } %lpval
}

; SimplifySuspendPoint won't be able to simplify if it detects that resume or
; destroy does not immediately preceed coro.suspend.
;
; CHECK-LABEL: define void @cannot_simplify_not_last_instr(
; CHECK-NEXT:  entry:
; CHECK-NEXT:     llvm.coro.id

define void @cannot_simplify_not_last_instr(ptr %dst, ptr %src) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias ptr @llvm.coro.begin(token %id, ptr %phi)
  br label %body
body:
  %save = call token @llvm.coro.save(ptr %hdl)
  %subfn = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %subfn(ptr %hdl)
  ; memcpy separates destroy from suspend, therefore cannot simplify.
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 1, i1 false)
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
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret void
}

; SimplifySuspendPoint should not simplify final suspend point
;
; CHECK-LABEL: define void @cannot_simplify_final_suspend(
; CHECK-NEXT:  entry:
; CHECK-NEXT:     llvm.coro.id
;
define void @cannot_simplify_final_suspend() presplitcoroutine personality i32 0 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call noalias ptr @llvm.coro.begin(token %id, ptr %phi)
  br label %body
body:
  %save = call token @llvm.coro.save(ptr %hdl)
  %subfn = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  invoke fastcc void %subfn(ptr %hdl) to label %real_susp unwind label %lpad

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
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret void
lpad:
  %lpval = landingpad { ptr, i32 }
     cleanup

  call void @print(i32 2)
  resume { ptr, i32 } %lpval
}

declare ptr @malloc(i32) allockind("alloc,uninitialized") allocsize(0)
declare void @free(ptr) willreturn allockind("free")
declare void @print(i32)
declare void @foo()

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare i32 @llvm.coro.size.i32()
declare ptr @llvm.coro.begin(token, ptr)
declare token @llvm.coro.save(ptr %hdl)
declare i8 @llvm.coro.suspend(token, i1)
declare ptr @llvm.coro.free(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)

declare ptr @llvm.coro.subfn.addr(ptr, i8)

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1)
