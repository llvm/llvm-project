; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | grep select
; rudimentary test of select on vectors returning vector of bool

define <4 x i32> @foo(<4 x i32> %a, <4 x i32> %b,
    <4 x i1> %cond) nounwind  {
entry:
  %cmp = select <4 x i1>  %cond, <4 x i32> %a, <4 x i32> %b 
                             ; <4 x i32> [#uses=1]
  ret <4 x i32> %cmp
}

