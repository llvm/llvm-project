; RUN: llc -mtriple=riscv32  -verify-machineinstrs < %s -filetype=obj -o %t
; RUN: llvm-readelf -A %t | FileCheck %s

; CHECK: Value: rv32i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0
define dso_local i32 @func0() #0 {
entry:
  ret i32 0
}

define dso_local i32 @func1() #1 {
entry:
  ret i32 0
}

define dso_local i32 @func2() #2 {
entry:
  ret i32 0
}

define dso_local i32 @func3() #3 {
entry:
  ret i32 0
}

attributes #0 = { nounwind "target-features"="+32bit,+d,+zicsr" }
attributes #1 = { nounwind "target-features"="+32bit,+d,+f,+m" }
attributes #2 = { nounwind "target-features"="+32bit,+f,+c" }
attributes #3 = { nounwind "target-features"="+32bit,+a" }
