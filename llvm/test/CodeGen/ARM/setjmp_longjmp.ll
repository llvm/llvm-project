; RUN: llc -simplifycfg-require-and-preserve-domtree=1 %s -o - | FileCheck %s
; RUN: llc -mtriple=armv7-linux -exception-model sjlj -simplifycfg-require-and-preserve-domtree=1 %s -o - | FileCheck %s -check-prefix CHECK-LINUX
; RUN: llc -mtriple=thumbv7-win32 -exception-model sjlj -simplifycfg-require-and-preserve-domtree=1 %s -o - | FileCheck %s -check-prefix CHECK-WIN32
target triple = "armv7-apple-ios"

declare i32 @llvm.eh.sjlj.setjmp(i8*)
declare void @llvm.eh.sjlj.longjmp(i8*)
@g = external global i32

declare void @may_throw()
declare i32 @__gxx_personality_sj0(...)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
declare i32 @llvm.eh.typeid.for(i8*)
declare i8* @llvm.frameaddress(i32)
declare i8* @llvm.stacksave()
@_ZTIPKc = external constant i8*

; CHECK-LABEL: foobar
;
; setjmp sequence:
; CHECK: add [[PCREG:r[0-9]+]], pc, #8
; CHECK-NEXT: str [[PCREG]], [[[BUFREG:r[0-9]+]], #4]
; CHECK-NEXT: mov r0, #0
; CHECK-NEXT: add pc, pc, #0
; CHECK-NEXT: mov r0, #1
;
; longjmp sequence:
; CHECK: ldr sp, [{{\s*}}[[BUFREG:r[0-9]+]], #8]
; CHECK-NEXT: ldr [[DESTREG:r[0-9]+]], [[[BUFREG]], #4]
; CHECK-NEXT: ldr r7, [[[BUFREG]]]
; CHECK-NEXT: bx [[DESTREG]]

; CHECK-LINUX: ldr sp, [{{\s*}}[[BUFREG:r[0-9]+]], #8]
; CHECK-LINUX-NEXT: ldr [[DESTREG:r[0-9]+]], [[[BUFREG]], #4]
; CHECK-LINUX-NEXT: ldr r7, [[[BUFREG]]]
; CHECK-LINUX-NEXT: ldr r11, [[[BUFREG]]]
; CHECK-LINUX-NEXT: bx [[DESTREG]]

; CHECK-WIN32: ldr.w r11, [{{\s*}}[[BUFREG:r[0-9]+]]]
; CHECK-WIN32-NEXT: ldr.w sp, [[[BUFREG]], #8]
; CHECK-WIN32-NEXT: ldr.w pc, [[[BUFREG]], #4]
define void @foobar() {
entry:
  %buf = alloca [5 x i8*], align 4
  %arraydecay = getelementptr inbounds [5 x i8*], [5 x i8*]* %buf, i32 0, i32 0
  %bufptr = bitcast i8** %arraydecay to i8*
  ; Note: This is simplified, in reality you have to store the framepointer +
  ; stackpointer in the buffer as well for this to be legal!
  %setjmpres = call i32 @llvm.eh.sjlj.setjmp(i8* %bufptr)
  %tobool = icmp ne i32 %setjmpres, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:
  store volatile i32 1, i32* @g, align 4
  br label %if.end

if.else:
  store volatile i32 0, i32* @g, align 4
  call void @llvm.eh.sjlj.longjmp(i8* %bufptr)
  unreachable

if.end:
  ret void
}

; CHECK-LABEL: combine_sjlj_eh_and_setjmp_longjmp
; Check that we can mix sjlj exception handling with __builtin_setjmp
; and __builtin_longjmp.
;
; setjmp sequence:
; CHECK: add [[PCREG:r[0-9]+]], pc, #8
; CHECK-NEXT: str [[PCREG]], [[[BUFREG:r[0-9]+]], #4]
; CHECK-NEXT: mov r0, #0
; CHECK-NEXT: add pc, pc, #0
; CHECK-NEXT: mov r0, #1
;
; longjmp sequence:
; CHECK: ldr sp, [{{\s*}}[[BUFREG:r[0-9]+]], #8]
; CHECK-NEXT: ldr [[DESTREG:r[0-9]+]], [[[BUFREG]], #4]
; CHECK-NEXT: ldr r7, [[[BUFREG]]]
; CHECK-NEXT: bx [[DESTREG]]
define void @combine_sjlj_eh_and_setjmp_longjmp() personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
entry:
  %buf = alloca [5 x i8*], align 4
  invoke void @may_throw() to label %try.cont unwind label %lpad

lpad:
  %0 = landingpad { i8*, i32 } catch i8* bitcast (i8** @_ZTIPKc to i8*)
  %1 = extractvalue { i8*, i32 } %0, 1
  %2 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIPKc to i8*)) #3
  %matches = icmp eq i32 %1, %2
  br i1 %matches, label %catch, label %eh.resume

catch:
  %3 = extractvalue { i8*, i32 } %0, 0
  %4 = tail call i8* @__cxa_begin_catch(i8* %3) #3
  store volatile i32 0, i32* @g, align 4
  %5 = bitcast [5 x i8*]* %buf to i8*
  %arraydecay = getelementptr inbounds [5 x i8*], [5 x i8*]* %buf, i64 0, i64 0
  %6 = tail call i8* @llvm.frameaddress(i32 0)
  store i8* %6, i8** %arraydecay, align 16
  %7 = tail call i8* @llvm.stacksave()
  %8 = getelementptr [5 x i8*], [5 x i8*]* %buf, i64 0, i64 2
  store i8* %7, i8** %8, align 16
  %9 = call i32 @llvm.eh.sjlj.setjmp(i8* %5)
  %tobool = icmp eq i32 %9, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:
  store volatile i32 2, i32* @g, align 4
  call void @__cxa_end_catch() #3
  br label %try.cont

if.else:
  store volatile i32 1, i32* @g, align 4
  call void @llvm.eh.sjlj.longjmp(i8* %5)
  unreachable

eh.resume:
  resume { i8*, i32 } %0

try.cont:
  ret void
}
