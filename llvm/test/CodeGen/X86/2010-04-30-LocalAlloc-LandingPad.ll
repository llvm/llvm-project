; RUN: llc < %s -O0 -regalloc=fast -relocation-model=pic -frame-pointer=all | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin10.0.0"

%struct.S = type { [2 x ptr] }

@_ZTIi = external constant ptr                    ; <ptr> [#uses=1]
@.str = internal constant [4 x i8] c"%p\0A\00"    ; <ptr> [#uses=1]
@llvm.used = appending global [1 x ptr] [ptr @_Z4test1SiS_], section "llvm.metadata" ; <ptr> [#uses=0]

; Verify that %s1 gets spilled before the call.
; CHECK: Z4test1SiS
; CHECK: leal 8(%ebp), %[[reg:[^ ]*]]
; CHECK: movl %[[reg]],{{.*}}(%ebp) ## 4-byte Spill
; CHECK: calll __Z6throwsv

define ptr @_Z4test1SiS_(ptr byval(%struct.S) %s1, i32 %n, ptr byval(%struct.S) %s2) ssp personality ptr @__gxx_personality_v0 {
entry:
  %retval = alloca ptr, align 4                   ; <ptr> [#uses=2]
  %n.addr = alloca i32, align 4                   ; <ptr> [#uses=1]
  %_rethrow = alloca ptr                          ; <ptr> [#uses=4]
  %0 = alloca i32, align 4                        ; <ptr> [#uses=1]
  %cleanup.dst = alloca i32                       ; <ptr> [#uses=3]
  %cleanup.dst7 = alloca i32                      ; <ptr> [#uses=6]
  store i32 %n, ptr %n.addr
  invoke void @_Z6throwsv()
          to label %invoke.cont unwind label %try.handler

invoke.cont:                                      ; preds = %entry
  store i32 1, ptr %cleanup.dst7
  br label %finally

terminate.handler:                                ; preds = %match.end
  %1 = landingpad { ptr, i32 }
           cleanup
  call void @_ZSt9terminatev() noreturn nounwind
  unreachable

try.handler:                                      ; preds = %entry
  %exc1.ptr = landingpad { ptr, i32 }
           catch ptr null
  %exc1 = extractvalue { ptr, i32 } %exc1.ptr, 0
  %selector = extractvalue { ptr, i32 } %exc1.ptr, 1
  %2 = call i32 @llvm.eh.typeid.for(ptr @_ZTIi) ; <i32> [#uses=1]
  %3 = icmp eq i32 %selector, %2                  ; <i1> [#uses=1]
  br i1 %3, label %match, label %catch.next

match:                                            ; preds = %try.handler
  %4 = call ptr @__cxa_begin_catch(ptr %exc1)     ; <ptr> [#uses=1]
  %5 = load i32, ptr %4                               ; <i32> [#uses=1]
  store i32 %5, ptr %0
  %call = invoke i32 (ptr, ...) @printf(ptr @.str, ptr %s2)
          to label %invoke.cont2 unwind label %match.handler ; <i32> [#uses=0]

invoke.cont2:                                     ; preds = %match
  store i32 1, ptr %cleanup.dst
  br label %match.end

match.handler:                                    ; preds = %match
  %exc3 = landingpad { ptr, i32 }
           cleanup
  %6 = extractvalue { ptr, i32 } %exc3, 0
  store ptr %6, ptr %_rethrow
  store i32 2, ptr %cleanup.dst
  br label %match.end

cleanup.pad:                                      ; preds = %cleanup.switch
  store i32 1, ptr %cleanup.dst7
  br label %finally

cleanup.pad4:                                     ; preds = %cleanup.switch
  store i32 2, ptr %cleanup.dst7
  br label %finally

match.end:                                        ; preds = %match.handler, %invoke.cont2
  invoke void @__cxa_end_catch()
          to label %invoke.cont5 unwind label %terminate.handler

invoke.cont5:                                     ; preds = %match.end
  br label %cleanup.switch

cleanup.switch:                                   ; preds = %invoke.cont5
  %tmp = load i32, ptr %cleanup.dst                   ; <i32> [#uses=1]
  switch i32 %tmp, label %cleanup.end [
    i32 1, label %cleanup.pad
    i32 2, label %cleanup.pad4
  ]

cleanup.end:                                      ; preds = %cleanup.switch
  store i32 2, ptr %cleanup.dst7
  br label %finally

catch.next:                                       ; preds = %try.handler
  store ptr %exc1, ptr %_rethrow
  store i32 2, ptr %cleanup.dst7
  br label %finally

finally:                                          ; preds = %catch.next, %cleanup.end, %cleanup.pad4, %cleanup.pad, %invoke.cont
  br label %cleanup.switch9

cleanup.switch9:                                  ; preds = %finally
  %tmp8 = load i32, ptr %cleanup.dst7                 ; <i32> [#uses=1]
  switch i32 %tmp8, label %cleanup.end10 [
    i32 1, label %finally.end
    i32 2, label %finally.throw
  ]

cleanup.end10:                                    ; preds = %cleanup.switch9
  br label %finally.end

finally.throw:                                    ; preds = %cleanup.switch9
  %7 = load ptr, ptr %_rethrow                        ; <ptr> [#uses=1]
  call void @_Unwind_Resume_or_Rethrow(ptr %7)
  unreachable

finally.end:                                      ; preds = %cleanup.end10, %cleanup.switch9
  %tmp11 = getelementptr inbounds %struct.S, ptr %s1, i32 0, i32 0 ; <ptr> [#uses=1]
  %arraydecay = getelementptr inbounds [2 x ptr], ptr %tmp11, i32 0, i32 0 ; <ptr> [#uses=1]
  %arrayidx = getelementptr inbounds ptr, ptr %arraydecay, i32 1 ; <ptr> [#uses=1]
  %tmp12 = load ptr, ptr %arrayidx                    ; <ptr> [#uses=1]
  store ptr %tmp12, ptr %retval
  %8 = load ptr, ptr %retval                          ; <ptr> [#uses=1]
  ret ptr %8
}

declare void @_Z6throwsv() ssp

declare i32 @__gxx_personality_v0(...)

declare void @_ZSt9terminatev()

declare void @_Unwind_Resume_or_Rethrow(ptr)

declare i32 @llvm.eh.typeid.for(ptr) nounwind

declare ptr @__cxa_begin_catch(ptr)

declare i32 @printf(ptr, ...)

declare void @__cxa_end_catch()
