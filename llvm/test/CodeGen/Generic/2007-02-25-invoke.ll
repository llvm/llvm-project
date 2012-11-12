; RUN: llc < %s

; PR1224

declare i32 @test()
define i32 @test2() {
        %A = invoke i32 @test() to label %invcont unwind label %blat
invcont:
        ret i32 %A
blat:
  %lpad = landingpad { i8*, i32 } personality i32 (...)* @__gxx_personality_v0
            cleanup
  ret i32 0
}

declare i32 @__gxx_personality_v0(...)
