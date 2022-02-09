; RUN: opt < %s -passes='print<cost-model>' 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z13
;
; Test that cost functions can handle immediates of more than 64 bits without crashing.

; Cost of a load which is checked for folding into a compare w/ memory.
define i32 @fun0(i72* %Src) {
  %L = load i72, i72* %Src
  %B = icmp ult i72 %L, 166153499473114484112
  %Res = zext i1 %B to i32
  ret i32 %Res
}

; Cost of a compare which is checked for elimination by Load and Test.
define i32 @fun1(i72* %Src, i72* %Dst) {
  %L = load i72, i72* %Src
  store i72 %L, i72* %Dst
  %B = icmp ult i72 %L, 166153499473114484112
  %Res = zext i1 %B to i32
  ret i32 %Res
}
