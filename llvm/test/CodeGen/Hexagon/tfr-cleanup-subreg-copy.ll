; RUN: llc -stop-after=tfr-cleanup -verify-machineinstrs -mtriple=hexagon %s -o - | FileCheck %s

; Create a copy from a 64-bit argument (double regs) to 32-bit intregs via subreg.
; The tfr-cleanup pass should not assert on size mismatch and should leave the
; copy intact when sizes differ.

; CHECK: name:            test
; CHECK: liveins: $d0, $r2
; CHECK: renamable $r0 = A2_add killed renamable $r0, renamable $r1
; CHECK: S2_storeri_io

define dso_local void @test(i64 %x, ptr nocapture %out) local_unnamed_addr {
entry:
  %lo = trunc i64 %x to i32
  %hi.shift = lshr i64 %x, 32
  %hi = trunc i64 %hi.shift to i32
  %sum = add i32 %lo, %hi
  store i32 %sum, ptr %out, align 4
  ret void
}
