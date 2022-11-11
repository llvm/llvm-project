; RUN: llc < %s -mtriple=x86_64-pc-linux -mcpu=corei7-avx

; No check in a crash test

define void @autogen_262380_1000() {
BB:
  br label %CF79

CF79:                                             ; preds = %CF79, %BB
  br i1 undef, label %CF79, label %CF84.critedge.critedge

CF84.critedge.critedge:                           ; preds = %CF79
  %L35 = load <8 x i32>, ptr undef
  br label %CF85

CF85:                                             ; preds = %CF85, %CF84.critedge.critedge
  br i1 undef, label %CF85, label %CF86

CF86:                                             ; preds = %CF86, %CF85
  %B61 = sub <8 x i32> %L35, zeroinitializer
  %S64 = icmp ne <8 x i32> %B61, zeroinitializer
  %E73 = extractelement <8 x i1> %S64, i32 6
  br i1 %E73, label %CF86, label %CF87

CF87:                                             ; preds = %CF87, %CF86
  br i1 undef, label %CF87, label %CF88

CF88:                                             ; preds = %CF87
  ret void
}
