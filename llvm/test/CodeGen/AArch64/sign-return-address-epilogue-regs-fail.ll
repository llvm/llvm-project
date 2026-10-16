; RUN: not llc -mtriple=aarch64 -asm-verbose=0              < %s 2>&1 | FileCheck --check-prefixes=NOT-ENOUGH-REGS %s
; RUN: not llc -mtriple=aarch64 -asm-verbose=0 -mattr=v8.3a < %s 2>&1 | FileCheck --check-prefixes=NOT-ENOUGH-REGS %s

; Compared to @test_multiple_scratch_regs, this test case keeps one more register
; alive (X20) at the insertion point of PAUTH_EPILOGUE.

; NOT-ENOUGH-REGS: LLVM ERROR: Cannot insert PAUTH_EPILOGUE: ran out of registers

%large_struct = type { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 }

%regs_t = type { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 }

define swifttailcc i64 @test_multiple_scratch_regs_fail(i64 %arg, %large_struct %s) "branch-protection-pauth-lr" "sign-return-address"="non-leaf" {
  %local = alloca [64 x i8], align 8
  %regs = tail call { i64, i64 } asm sideeffect "mov $0, 42\0A\09mov $1, 123", "={x16},={x17}"()
  %live_x16 = extractvalue { i64, i64 } %regs, 0
  %live_x17 = extractvalue { i64, i64 } %regs, 1

  %other.regs.def = tail call %regs_t asm sideeffect "", "={x0},={x1},={x2},={x3},={x4},={x5},={x6},={x7},={x8},={x9},={x10},={x11},={x12},={x13},={x14},={x18},={x20}"()
  %live_x0 = extractvalue %regs_t %other.regs.def, 0
  %live_x1 = extractvalue %regs_t %other.regs.def, 1
  %live_x2 = extractvalue %regs_t %other.regs.def, 2
  %live_x3 = extractvalue %regs_t %other.regs.def, 3
  %live_x4 = extractvalue %regs_t %other.regs.def, 4
  %live_x5 = extractvalue %regs_t %other.regs.def, 5
  %live_x6 = extractvalue %regs_t %other.regs.def, 6
  %live_x7 = extractvalue %regs_t %other.regs.def, 7
  %live_x8 = extractvalue %regs_t %other.regs.def, 8
  %live_x9 = extractvalue %regs_t %other.regs.def, 9
  %live_x10 = extractvalue %regs_t %other.regs.def, 10
  %live_x11 = extractvalue %regs_t %other.regs.def, 11
  %live_x12 = extractvalue %regs_t %other.regs.def, 12
  %live_x13 = extractvalue %regs_t %other.regs.def, 13
  %live_x14 = extractvalue %regs_t %other.regs.def, 14

  %live_x18 = extractvalue %regs_t %other.regs.def, 15
  %live_x20 = extractvalue %regs_t %other.regs.def, 16

  %cond = icmp eq i64 %arg, 0
  br i1 %cond, label %if.end, label %if.then

if.then:
  call void @llvm.lifetime.start.p0(i64 64, ptr %local)
  tail call void asm sideeffect "mov x30, 12345", "~{lr}"()
  call void @llvm.lifetime.end.p0(i64 64, ptr %local)
  br label %if.end

if.end:
  tail call void asm sideeffect "", "{x0},{x1},{x2},{x3},{x4},{x5},{x6},{x7},{x8},{x9},{x10},{x11},{x12},{x13},{x14},{x18},{x20}" (i64 %live_x0, i64 %live_x1, i64 %live_x2, i64 %live_x3, i64 %live_x4, i64 %live_x5, i64 %live_x6, i64 %live_x7, i64 %live_x8, i64 %live_x9, i64 %live_x10, i64 %live_x11, i64 %live_x12, i64 %live_x13, i64 %live_x14, i64 %live_x18, i64 %live_x20)
  %result = tail call i64 asm sideeffect "add $0, $1, $2", "=r,{x16},{x17}"(i64 %live_x16, i64 %live_x17)
  ret i64 %result
}
