; RUN: llc < %s -mattr=-sse2,-sse41 -verify-machineinstrs
target triple = "x86_64-unknown-linux-gnu"

; PR10503
; This test case produces INSERT_SUBREG 0, <undef> instructions that
; ProcessImplicitDefs doesn't eliminate.
define void @autogen_136178_500() {
BB:
  %Shuff6 = shufflevector <32 x i32> undef, <32 x i32> undef, <32 x i32> <i32 27, i32 29, i32 31, i32 undef, i32 undef, i32 37, i32 39, i32 41, i32 undef, i32 45, i32 47, i32 49, i32 51, i32 53, i32 55, i32 57, i32 undef, i32 61, i32 63, i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 undef, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25>
  %S17 = select i1 true, <8 x float>* null, <8 x float>* null
  br label %CF

CF:                                               ; preds = %CF, %BB
  %L19 = load <8 x float>* %S17
  %BC = bitcast <32 x i32> %Shuff6 to <32 x float>
  %S28 = fcmp ord double 0x3ED1A1F787BB2185, 0x3EE59DE55A8DF890
  br i1 %S28, label %CF, label %CF39

CF39:                                             ; preds = %CF39, %CF
  store <8 x float> %L19, <8 x float>* %S17
  %I35 = insertelement <32 x float> %BC, float 0x3EC2489F60000000, i32 9
  %S38 = fcmp ule double 0x3EE59DE55A8DF890, 0x3EC4AB0CBB986A1A
  br i1 %S38, label %CF39, label %CF40

CF40:                                             ; preds = %CF39
  ret void
}
