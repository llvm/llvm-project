; RUN: opt < %s -passes=mem2reg -S
; PR5023

declare i32 @test1f()

define i32 @test1() personality i32 (...)* @__gxx_personality_v0 {
entry:
  %whichFlag = alloca i32
  %A = invoke i32 @test1f()
          to label %invcont2 unwind label %lpad86

invcont2:
  store i32 %A, i32* %whichFlag
  br label %bb15

bb15:
  %B = load i32, i32* %whichFlag
  ret i32 %B

lpad86:
  %exn = landingpad {i8*, i32}
           cleanup
  br label %bb15
  
}

declare i32 @__gxx_personality_v0(...)


define i32 @test2() {
entry:
  %whichFlag = alloca i32
  br label %bb15

bb15:
  %B = load i32, i32* %whichFlag
  ret i32 %B

invcont2:
  %C = load i32, i32* %whichFlag
  store i32 %C, i32* %whichFlag
  br label %bb15
}

