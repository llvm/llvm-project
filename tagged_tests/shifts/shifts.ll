; =========================
; rv32: SL/SR rewrite check
; =========================
target triple = "riscv32-unknown-elf"

define i32 @test_shl32(i32 %x, i32 %amt) nounwind {
entry:
  %y = shl i32 %x, %amt
  ret i32 %y
}

define i32 @test_lshr32(i32 %x, i32 %amt) nounwind {
entry:
  %y = lshr i32 %x, %amt
  ret i32 %y
}

define i32 @test_ashr32(i32 %x, i32 %amt) nounwind {
entry:
  %y = ashr i32 %x, %amt
  ret i32 %y
}