; RUN: llc -mattr=avr6,sram < %s -march=avr | FileCheck %s

; CHECK: ld {{r[0-9]+}}, [[PTR:[XYZ]]]
; CHECK: ldd {{r[0-9]+}}, [[PTR]]+1
; CHECK: st [[PTR2:[XYZ]]], {{r[0-9]+}}
; CHECK: std [[PTR2]]+1, {{r[0-9]+}}
define void @load_store_16(i16* nocapture %ptr) local_unnamed_addr #1 {
entry:
  %0 = load i16, i16* %ptr, align 2
  %add = add i16 %0, 5
  store i16 %add, i16* %ptr, align 2
  ret void
}
