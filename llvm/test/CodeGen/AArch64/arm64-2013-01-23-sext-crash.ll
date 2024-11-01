; RUN: llc < %s -mtriple=arm64-eabi

; Make sure we are not crashing on this test.

define void @autogen_SD12881() {
BB:
  %B17 = ashr <4 x i32> zeroinitializer, zeroinitializer
  br label %CF

CF:                                               ; preds = %CF83, %CF, %BB
  br i1 undef, label %CF, label %CF83

CF83:                                             ; preds = %CF
  %FC70 = sitofp <4 x i32> %B17 to <4 x double>
  br label %CF
}


define void @autogen_SD12881_2() {
BB:
  %B17 = ashr <4 x i32> zeroinitializer, zeroinitializer
  br label %CF

CF:                                               ; preds = %CF83, %CF, %BB
  br i1 undef, label %CF, label %CF83

CF83:                                             ; preds = %CF
  %FC70 = uitofp <4 x i32> %B17 to <4 x double>
  br label %CF
}

define void @_Z12my_example2bv() nounwind noinline ssp {
entry:
  %0 = fptosi <2 x double> undef to <2 x i32>
  store <2 x i32> %0, ptr undef, align 8
  ret void
}
