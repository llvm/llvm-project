; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 -O3 -o /dev/null
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -O3 -o /dev/null
;
; Test that regalloc does not run out of registers

; This test will include a GR128 virtual reg.
define void @test0(i64 %dividend, i64 %divisor) {
  %rem = urem i64 %dividend, %divisor
  call void asm sideeffect "", "{r0},{r1},{r2},{r3},{r4},{r5},{r6},{r7},{r8},{r9},{r10},{r11},{r12},{r13},{r14}"(i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 %rem)
  ret void
}

; This test will include an ADDR128 virtual reg.
define i64 @test1(i64 %dividend, i64 %divisor) {
%rem = urem i64 %dividend, %divisor
call void asm sideeffect "", "{r2},{r3},{r4},{r5},{r6},{r7},{r8},{r9},{r10},{r11},{r12},{r13},{r14}"(i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 %rem)
%ret = add i64 %rem, 1
ret i64 %ret
}
