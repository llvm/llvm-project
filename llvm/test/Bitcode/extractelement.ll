; RUN: opt < %s -passes=instsimplify | llvm-dis -disable-output
; RUN: verify-uselistorder < %s
; PR3465

define double @test() {
  %tmp24 = extractelement <2 x double> bitcast (<1 x i128> < i128 85070591730234615870450834276742070272 > to <2 x double>), i32 0
  ret double %tmp24
}

