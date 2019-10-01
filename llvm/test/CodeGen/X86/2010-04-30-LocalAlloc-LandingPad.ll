; RUN: llc < %s -O0 -regalloc=fast -relocation-model=pic -frame-pointer=all | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin10.0.0"

%struct.S = type { [2 x i8*] }

@_ZTIi = external constant i8*                    ; <i8**> [#uses=1]
@.str = internal constant [4 x i8] c"%p\0A\00"    ; <[4 x i8]*> [#uses=1]
@llvm.used = appending global [1 x i8*] [i8* bitcast (i8* (%struct.S*, i32, %struct.S*)* @_Z4test1SiS_ to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

; Verify that %s1 gets spilled before the call.
; CHECK: Z4test1SiS
; CHECK: leal 8(%ebp), %[[reg:[^ ]*]]
; CHECK: movl %[[reg]],{{.*}}(%ebp) ## 4-byte Spill
; CHECK: calll __Z6throwsv

define i8* @_Z4test1SiS_(%struct.S* byval %s1, i32 %n, %struct.S* byval %s2) ssp personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %retval = alloca i8*, align 4                   ; <i8**> [#uses=2]
  %n.addr = alloca i32, align 4                   ; <i32*> [#uses=1]
  %_rethrow = alloca i8*                          ; <i8**> [#uses=4]
  %0 = alloca i32, align 4                        ; <i32*> [#uses=1]
  %cleanup.dst = alloca i32                       ; <i32*> [#uses=3]
  %cleanup.dst7 = alloca i32                      ; <i32*> [#uses=6]
  store i32 %n, i32* %n.addr
  invoke void @_Z6throwsv()
          to label %invoke.cont unwind label %try.handler

invoke.cont:                                      ; preds = %entry
  store i32 1, i32* %cleanup.dst7
  br label %finally

terminate.handler:                                ; preds = %match.end
  %1 = landingpad { i8*, i32 }
           cleanup
  call void @_ZSt9terminatev() noreturn nounwind
  unreachable

try.handler:                                      ; preds = %entry
  %exc1.ptr = landingpad { i8*, i32 }
           catch i8* null
  %exc1 = extractvalue { i8*, i32 } %exc1.ptr, 0
  %selector = extractvalue { i8*, i32 } %exc1.ptr, 1
  %2 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) ; <i32> [#uses=1]
  %3 = icmp eq i32 %selector, %2                  ; <i1> [#uses=1]
  br i1 %3, label %match, label %catch.next

match:                                            ; preds = %try.handler
  %4 = call i8* @__cxa_begin_catch(i8* %exc1)     ; <i8*> [#uses=1]
  %5 = bitcast i8* %4 to i32*                     ; <i32*> [#uses=1]
  %6 = load i32, i32* %5                               ; <i32> [#uses=1]
  store i32 %6, i32* %0
  %call = invoke i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), %struct.S* %s2)
          to label %invoke.cont2 unwind label %match.handler ; <i32> [#uses=0]

invoke.cont2:                                     ; preds = %match
  store i32 1, i32* %cleanup.dst
  br label %match.end

match.handler:                                    ; preds = %match
  %exc3 = landingpad { i8*, i32 }
           cleanup
  %7 = extractvalue { i8*, i32 } %exc3, 0
  store i8* %7, i8** %_rethrow
  store i32 2, i32* %cleanup.dst
  br label %match.end

cleanup.pad:                                      ; preds = %cleanup.switch
  store i32 1, i32* %cleanup.dst7
  br label %finally

cleanup.pad4:                                     ; preds = %cleanup.switch
  store i32 2, i32* %cleanup.dst7
  br label %finally

match.end:                                        ; preds = %match.handler, %invoke.cont2
  invoke void @__cxa_end_catch()
          to label %invoke.cont5 unwind label %terminate.handler

invoke.cont5:                                     ; preds = %match.end
  br label %cleanup.switch

cleanup.switch:                                   ; preds = %invoke.cont5
  %tmp = load i32, i32* %cleanup.dst                   ; <i32> [#uses=1]
  switch i32 %tmp, label %cleanup.end [
    i32 1, label %cleanup.pad
    i32 2, label %cleanup.pad4
  ]

cleanup.end:                                      ; preds = %cleanup.switch
  store i32 2, i32* %cleanup.dst7
  br label %finally

catch.next:                                       ; preds = %try.handler
  store i8* %exc1, i8** %_rethrow
  store i32 2, i32* %cleanup.dst7
  br label %finally

finally:                                          ; preds = %catch.next, %cleanup.end, %cleanup.pad4, %cleanup.pad, %invoke.cont
  br label %cleanup.switch9

cleanup.switch9:                                  ; preds = %finally
  %tmp8 = load i32, i32* %cleanup.dst7                 ; <i32> [#uses=1]
  switch i32 %tmp8, label %cleanup.end10 [
    i32 1, label %finally.end
    i32 2, label %finally.throw
  ]

cleanup.end10:                                    ; preds = %cleanup.switch9
  br label %finally.end

finally.throw:                                    ; preds = %cleanup.switch9
  %8 = load i8*, i8** %_rethrow                        ; <i8*> [#uses=1]
  call void @_Unwind_Resume_or_Rethrow(i8* %8)
  unreachable

finally.end:                                      ; preds = %cleanup.end10, %cleanup.switch9
  %tmp11 = getelementptr inbounds %struct.S, %struct.S* %s1, i32 0, i32 0 ; <[2 x i8*]*> [#uses=1]
  %arraydecay = getelementptr inbounds [2 x i8*], [2 x i8*]* %tmp11, i32 0, i32 0 ; <i8**> [#uses=1]
  %arrayidx = getelementptr inbounds i8*, i8** %arraydecay, i32 1 ; <i8**> [#uses=1]
  %tmp12 = load i8*, i8** %arrayidx                    ; <i8*> [#uses=1]
  store i8* %tmp12, i8** %retval
  %9 = load i8*, i8** %retval                          ; <i8*> [#uses=1]
  ret i8* %9
}

declare void @_Z6throwsv() ssp

declare i32 @__gxx_personality_v0(...)

declare void @_ZSt9terminatev()

declare void @_Unwind_Resume_or_Rethrow(i8*)

declare i32 @llvm.eh.typeid.for(i8*) nounwind

declare i8* @__cxa_begin_catch(i8*)

declare i32 @printf(i8*, ...)

declare void @__cxa_end_catch()
