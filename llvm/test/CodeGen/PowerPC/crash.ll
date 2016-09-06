; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7

define void @test1(i1 %x, i8 %x2, i8* %x3, i64 %x4) {
entry:
  %tmp3 = and i64 %x4, 16
  %bf.shl = trunc i64 %tmp3 to i8
  %bf.clear = and i8 %x2, -17
  %bf.set = or i8 %bf.shl, %bf.clear
  br i1 %x, label %if.then, label %if.end

if.then:
  ret void

if.end:
  store i8 %bf.set, i8* %x3, align 4
  ret void
}
