; RUN: llc -simplifycfg-require-and-preserve-domtree=1 %s -o - | FileCheck %s
; RUN: llc -mtriple=armv7-linux -exception-model sjlj -simplifycfg-require-and-preserve-domtree=1 %s -o - | FileCheck %s -check-prefix CHECK-LINUX
; RUN: llc -mtriple=thumbv7-win32 -exception-model sjlj -simplifycfg-require-and-preserve-domtree=1 %s -o - | FileCheck %s -check-prefix CHECK-WIN32
target triple = "armv7-apple-ios"

declare i32 @llvm.eh.sjlj.setjmp(ptr)
declare void @llvm.eh.sjlj.longjmp(ptr)
@g = external global i32

declare void @may_throw()
declare i32 @__gxx_personality_sj0(...)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()
declare i32 @llvm.eh.typeid.for(ptr)
declare ptr @llvm.frameaddress(i32)
declare ptr @llvm.stacksave()
@_ZTIPKc = external constant ptr

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
  %buf = alloca [5 x ptr], align 4
  ; Note: This is simplified, in reality you have to store the framepointer +
  ; stackpointer in the buffer as well for this to be legal!
  %setjmpres = call i32 @llvm.eh.sjlj.setjmp(ptr %buf)
  %tobool = icmp ne i32 %setjmpres, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:
  store volatile i32 1, ptr @g, align 4
  br label %if.end

if.else:
  store volatile i32 0, ptr @g, align 4
  call void @llvm.eh.sjlj.longjmp(ptr %buf)
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
define void @combine_sjlj_eh_and_setjmp_longjmp() personality ptr @__gxx_personality_sj0 {
entry:
  %buf = alloca [5 x ptr], align 4
  invoke void @may_throw() to label %try.cont unwind label %lpad

lpad:
  %0 = landingpad { ptr, i32 } catch ptr @_ZTIPKc
  %1 = extractvalue { ptr, i32 } %0, 1
  %2 = tail call i32 @llvm.eh.typeid.for(ptr @_ZTIPKc) #3
  %matches = icmp eq i32 %1, %2
  br i1 %matches, label %catch, label %eh.resume

catch:
  %3 = extractvalue { ptr, i32 } %0, 0
  %4 = tail call ptr @__cxa_begin_catch(ptr %3) #3
  store volatile i32 0, ptr @g, align 4
  %5 = tail call ptr @llvm.frameaddress(i32 0)
  store ptr %5, ptr %buf, align 16
  %6 = tail call ptr @llvm.stacksave()
  %7 = getelementptr [5 x ptr], ptr %buf, i64 0, i64 2
  store ptr %6, ptr %7, align 16
  %8 = call i32 @llvm.eh.sjlj.setjmp(ptr %buf)
  %tobool = icmp eq i32 %8, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:
  store volatile i32 2, ptr @g, align 4
  call void @__cxa_end_catch() #3
  br label %try.cont

if.else:
  store volatile i32 1, ptr @g, align 4
  call void @llvm.eh.sjlj.longjmp(ptr %buf)
  unreachable

eh.resume:
  resume { ptr, i32 } %0

try.cont:
  ret void
}
