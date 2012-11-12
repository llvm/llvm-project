; rdar://7879828
; RUN: opt -inline -argpromotion %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

define void @foo() {
  invoke void @foo2()
          to label %if.end432 unwind label %for.end520 

if.end432:  
  unreachable

for.end520: 
  unreachable
}

define internal  void @foo2() ssp {
  %call7 = call fastcc i8* @foo3(i1 (i8*)* @foo4)
  %call58 = call fastcc i8* @foo3(i1 (i8*)* @foo5)
  unreachable
}

define internal fastcc i8* @foo3(i1 (i8*)* %Pred) {
entry:
  unreachable
}

define internal i1 @foo4(i8* %O) nounwind {
entry:
  %call = call zeroext i1 @foo5(i8* %O) ; <i1> [#uses=0]
  unreachable
}

define internal i1 @foo5(i8* %O) nounwind {
entry:
  ret i1 undef
}


; PR8932 - infinite promotion.
%0 = type { %0* }

define i32 @test2(i32 %a) {
init:
  %0 = alloca %0
  %1 = alloca %0
  %2 = call i32 @"clay_assign(Chain, Chain)"(%0* %0, %0* %1)
  ret i32 0
}

define internal i32 @"clay_assign(Chain, Chain)"(%0* %c, %0* %d) {
init:
  %0 = getelementptr %0* %d, i32 0, i32 0
  %1 = load %0** %0
  %2 = getelementptr %0* %c, i32 0, i32 0
  %3 = load %0** %2
  %4 = call i32 @"clay_assign(Chain, Chain)"(%0* %3, %0* %1)
  ret i32 0
}
