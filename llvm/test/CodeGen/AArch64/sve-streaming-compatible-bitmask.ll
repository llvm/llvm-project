; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve,+sme -O2 -verify-machineinstrs < %s | FileCheck %s

; The bitmask includes all 16 lanes. A byte reduction drops bits 8 through 15.
define i16 @pack_mask(ptr %address) #0 {
; CHECK-LABEL: pack_mask:
; CHECK: uaddv {{d[0-9]+}}, {{p[0-9]+}}, {{z[0-9]+}}.h
  %v = load <16 x i8>, ptr %address, align 1
  %mask = icmp slt <16 x i8> %v, zeroinitializer
  %packed = bitcast <16 x i1> %mask to i16
  ret i16 %packed
}

attributes #0 = { "aarch64_pstate_sm_compatible" }
