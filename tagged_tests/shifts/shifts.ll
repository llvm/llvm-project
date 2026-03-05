; =========================
; Tagged RISC-V Shift Test
; Expect: SLL/SRL/SRA -> SL/SR (after your pre-RA rewrite)
; =========================

target triple = "riscv64-unknown-elf"

@sink = global i64 0, align 8

; --- Left shift: should become SLL (then rewritten to SL) ---
define i64 @test_shl(i64 %x, i64 %amt) nounwind {
entry:
  %y = shl i64 %x, %amt
  ret i64 %y
}

; --- Logical right shift: should become SRL (then rewritten to SR) ---
define i64 @test_lshr(i64 %x, i64 %amt) nounwind {
entry:
  %y = lshr i64 %x, %amt
  ret i64 %y
}

; --- Arithmetic right shift: should become SRA (then rewritten to SR) ---
define i64 @test_ashr(i64 %x, i64 %amt) nounwind {
entry:
  %y = ashr i64 %x, %amt
  ret i64 %y
}

; A small driver that prevents everything from being DCE’d.
; The volatile store keeps results observable.
define void @run(i64 %a, i64 %b) nounwind {
entry:
  %s1 = call i64 @test_shl(i64 %a, i64 %b)
  %s2 = call i64 @test_lshr(i64 %a, i64 %b)
  %s3 = call i64 @test_ashr(i64 %a, i64 %b)

  %t0 = xor i64 %s1, %s2
  %t1 = xor i64 %t0, %s3

  store volatile i64 %t1, ptr @sink, align 8
  ret void
}