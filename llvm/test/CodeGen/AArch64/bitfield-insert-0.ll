; RUN: llc -mtriple=aarch64-linux-gnu -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s

; The encoding of lsb -> immr in the CGed bitfield instructions was wrong at one
; point, in the edge case where lsb = 0. Just make sure.

define void @test_bfi0(i32* %existing, i32* %new) {
; CHECK: bfxil {{w[0-9]+}}, {{w[0-9]+}}, #0, #18

  %oldval = load volatile i32, i32* %existing
  %oldval_keep = and i32 %oldval, 4294705152 ; 0xfffc_0000

  %newval = load volatile i32, i32* %new
  %newval_masked = and i32 %newval, 262143 ; = 0x0003_ffff

  %combined = or i32 %newval_masked, %oldval_keep
  store volatile i32 %combined, i32* %existing

  ret void
}
