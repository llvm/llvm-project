; RUN: llc %s -o - | FileCheck %s

target triple = "mipsel-unknown-linux-gnu"

define dso_local void @test() #0 {
  ret void
}

attributes #0 = { "target-cpu"="mips32r2" "target-features"="+fp64,+mips32r2,+nan2008" }


; CHECK: .nan    2008
; CHECK: .module fp=64
