; RUN: llc -mtriple=arm-eabi -mcpu=swift -verify-machineinstrs %s -o /dev/null

%union.opcode.0.2.5.8.15.28 = type { i32 }

@opcode = external global %union.opcode.0.2.5.8.15.28, align 4
@operands = external hidden global [50 x i8], align 4
@.str86 = external hidden unnamed_addr constant [13 x i8], align 1

; Function Attrs: nounwind ssp
define void @xfr() {
entry:
  %bf.load4 = load i32, ptr @opcode, align 4
  %bf.clear10 = and i32 %bf.load4, 65535
  %and11 = and i32 %bf.load4, 32768
  %tobool12 = icmp ne i32 %and11, 0
  %cond13 = select i1 %tobool12, i32 1073676288, i32 0
  %or = or i32 %cond13, %bf.clear10
  %shl = shl nuw i32 %or, 2
  %add = add i32 0, %shl
  tail call void (ptr, i32, i32, ptr, ...) @__sprintf_chk(ptr @operands, i32 0, i32 50, ptr @.str86, i32 undef, i32 undef, i32 %add)
  ret void
}

declare void @__sprintf_chk(ptr, i32, i32, ptr, ...)
