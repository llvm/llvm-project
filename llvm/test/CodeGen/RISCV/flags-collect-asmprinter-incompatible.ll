; RUN: llc -mtriple=riscv32  -verify-machineinstrs < %s -filetype=obj -o %t
; RUN: llvm-readelf -A %t | FileCheck %s

; CHECK: Value:  rv32i2p1_a2p1_zicsr2p0_zca1p0
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

define dso_local i32 @func4() #4 {
entry:
  ret i32 0
}

attributes #0 = { nounwind "target-features"="+32bit,+d" }
attributes #1 = { nounwind "target-features"="+32bit,+c" }
attributes #2 = { nounwind "target-features"="+32bit,+f,+zcmp" }
attributes #3 = { nounwind "target-features"="+32bit,+a,+zfinx" }
attributes #4 = { nounwind "target-features"="+32bit,+a,+zcmt" }
