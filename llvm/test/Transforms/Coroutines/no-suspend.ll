; Test no suspend coroutines
; RUN: opt < %s -O2 -enable-coroutines -S | FileCheck %s

; Coroutine with no-suspends will turn into:
;
; CHECK-LABEL: define void @no_suspends(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @print(i32 %n)
; CHECK-NEXT:    ret void
;
define void @no_suspends(i32 %n) {
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
; CHECK-NEXT:    call void @print(i32 0)
; CHECK-NEXT:    ret void
;
define void @simplify_resume() {
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
  call void @llvm.coro.resume(i8* %hdl)
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
; CHECK-NEXT:    call void @print(i32 1)
; CHECK-NEXT:    ret void
;
define void @simplify_destroy() {
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
  call void @llvm.coro.destroy(i8* %hdl)
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
; CHECK-LABEL: define void @cannot_simplify(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call i8* @malloc

define void @cannot_simplify() {
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
  call void @foo()
  call void @llvm.coro.destroy(i8* %hdl)
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

declare i8* @malloc(i32)
declare void @free(i8*)
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

declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)
