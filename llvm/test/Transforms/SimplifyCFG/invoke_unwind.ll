; RUN: opt < %s -simplifycfg -S | FileCheck %s

declare void @bar()

; This testcase checks to see if the simplifycfg pass is converting invoke
; instructions to call instructions if the handler just rethrows the exception.
define i32 @test1() {
; CHECK: @test1
; CHECK-NEXT: call void @bar()
; CHECK-NEXT: ret i32 0
        invoke void @bar( )
                        to label %1 unwind label %Rethrow
        ret i32 0
Rethrow:
        %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
                 catch i8* null
        resume { i8*, i32 } %exn
}

declare i32 @__gxx_personality_v0(...)
