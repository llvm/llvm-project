; RUN: opt -mtriple=i686-pc-windows-msvc -S -x86-winehstate  < %s | FileCheck %s
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

@jb = external global i8

define i32 @test1() personality i32 (...)* @__CxxFrameHandler3 {
entry:
; CHECK-LABEL: define i32 @test1(
; CHECK: %[[eh_reg:.*]] = alloca
; CHECK: %[[gep:.*]] = getelementptr inbounds {{.*}}, {{.*}} %[[eh_reg]], i32 0, i32 2
; CHECK: store i32 -1, i32* %[[gep]]
; CHECK: %[[gep:.*]] = getelementptr inbounds {{.*}}, {{.*}} %[[eh_reg]], i32 0, i32 2
; CHECK: store i32 0, i32* %[[gep]]
; CHECK: %[[lsda:.*]] = call i8* @llvm.x86.seh.lsda(i8* bitcast (i32 ()* @test1 to i8*))
; CHECK: invoke i32 (i8*, i32, ...) @_setjmp3(i8* @jb, i32 3, void (i8*)* @__CxxLongjmpUnwind, i32 0, i8* %[[lsda]])
  %inv = invoke i32 (i8*, i32, ...) @_setjmp3(i8* @jb, i32 0) #2
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:
; CHECK: %[[gep:.*]] = getelementptr inbounds {{.*}}, {{.*}} %[[eh_reg]], i32 0, i32 2
; CHECK: store i32 -1, i32* %[[gep]]
; CHECK: %[[lsda:.*]] = call i8* @llvm.x86.seh.lsda(i8* bitcast (i32 ()* @test1 to i8*))
; CHECK: call i32 (i8*, i32, ...) @_setjmp3(i8* @jb, i32 3, void (i8*)* @__CxxLongjmpUnwind, i32 -1, i8* %[[lsda]])
  call i32 (i8*, i32, ...) @_setjmp3(i8* @jb, i32 0)
  call void @cleanup()
  ret i32 %inv

ehcleanup:
  %cp = cleanuppad within none []
; CHECK: %[[gep:.*]] = getelementptr inbounds {{.*}}, {{.*}} %[[eh_reg]], i32 0, i32 2
; CHECK: %[[load:.*]] = load i32, i32* %[[gep]]
; CHECK: %[[lsda:.*]] = call i8* @llvm.x86.seh.lsda(i8* bitcast (i32 ()* @test1 to i8*))
; CHECK: call i32 (i8*, i32, ...) @_setjmp3(i8* @jb, i32 3, void (i8*)* @__CxxLongjmpUnwind, i32 %[[load]], i8* %[[lsda]]) [ "funclet"(token %cp) ]
  %cal = call i32 (i8*, i32, ...) @_setjmp3(i8* @jb, i32 0) [ "funclet"(token %cp) ]
  call void @cleanup() [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller
}

define i32 @test2() personality i32 (...)* @_except_handler3 {
entry:
; CHECK-LABEL: define i32 @test2(
; CHECK: %[[eh_reg:.*]] = alloca
; CHECK: %[[gep:.*]] = getelementptr inbounds {{.*}}, {{.*}} %[[eh_reg]], i32 0, i32 4
; CHECK: store i32 -1, i32* %[[gep]]
; CHECK: %[[gep:.*]] = getelementptr inbounds {{.*}}, {{.*}} %[[eh_reg]], i32 0, i32 4
; CHECK: store i32 0, i32* %[[gep]]
; CHECK: invoke i32 (i8*, i32, ...) @_setjmp3(i8* @jb, i32 2, void (i8*)* @_seh_longjmp_unwind, i32 0)
  %inv = invoke i32 (i8*, i32, ...) @_setjmp3(i8* @jb, i32 0) #2
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:
; CHECK: %[[gep:.*]] = getelementptr inbounds {{.*}}, {{.*}} %[[eh_reg]], i32 0, i32 4
; CHECK: store i32 -1, i32* %[[gep]]
; CHECK: call i32 (i8*, i32, ...) @_setjmp3(i8* @jb, i32 2, void (i8*)* @_seh_longjmp_unwind, i32 -1)
  call i32 (i8*, i32, ...) @_setjmp3(i8* @jb, i32 0)
  call void @cleanup()
  ret i32 %inv

ehcleanup:
  %cp = cleanuppad within none []
  call void @cleanup() [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller
}

; Function Attrs: returns_twice
declare i32 @_setjmp3(i8*, i32, ...) #2

declare i32 @__CxxFrameHandler3(...)

declare i32 @_except_handler3(...)

declare void @cleanup()

attributes #2 = { returns_twice }
