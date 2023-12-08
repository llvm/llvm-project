; RUN: llc < %s

; PR1224

declare i32 @test()
define i32 @test2() personality ptr @__gxx_personality_v0 {
        %A = invoke i32 @test() to label %invcont unwind label %blat
invcont:
        ret i32 %A
blat:
  %lpad = landingpad { ptr, i32 }
            cleanup
  ret i32 0
}

declare i32 @__gxx_personality_v0(...) addrspace(0)
