; RUN: llc -mtriple=mips-linux-gnu -mattr=-nan2008 < %s | FileCheck %s
; RUN: llc -mtriple=mips-linux-gnu -mattr=+nan2008 < %s | FileCheck %s

; Make sure that lowering does not corrupt the value of NaN values,
; regardless of what the NaN mode is.

define float @test1() {
; CHECK: .4byte 0x7fc00000
  ret float bitcast (i32 u0x7fc00000 to float)
}

define float @test2() {
; CHECK: .4byte 0x7fc00001
  ret float bitcast (i32 u0x7fc00001 to float)
}

define float @test3() {
; CHECK: .4byte 0x7f800000
  ret float bitcast (i32 u0x7f800000 to float)
}

define float @test4() {
; CHECK: .4byte 0x7f800001
  ret float bitcast (i32 u0x7f800001 to float)
}
