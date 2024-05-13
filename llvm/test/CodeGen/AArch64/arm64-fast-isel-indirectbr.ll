; RUN: llc -O0 -fast-isel -fast-isel-abort=1 -verify-machineinstrs -mtriple=arm64-apple-darwin < %s | FileCheck %s

@fn.table = internal global [2 x ptr] [ptr blockaddress(@fn, %ZERO), ptr blockaddress(@fn, %ONE)], align 8

define i32 @fn(i32 %target) nounwind {
entry:
; CHECK-LABEL: fn
  %retval = alloca i32, align 4
  %target.addr = alloca i32, align 4
  store i32 %target, ptr %target.addr, align 4
  %0 = load i32, ptr %target.addr, align 4
  %idxprom = zext i32 %0 to i64
  %arrayidx = getelementptr inbounds [2 x ptr], ptr @fn.table, i32 0, i64 %idxprom
  %1 = load ptr, ptr %arrayidx, align 8
  br label %indirectgoto

ZERO:                                             ; preds = %indirectgoto
; CHECK: LBB0_1
  store i32 0, ptr %retval
  br label %return

ONE:                                              ; preds = %indirectgoto
; CHECK: LBB0_2
  store i32 1, ptr %retval
  br label %return

return:                                           ; preds = %ONE, %ZERO
  %2 = load i32, ptr %retval
  ret i32 %2

indirectgoto:                                     ; preds = %entry
; CHECK:      ldr [[REG:x[0-9]+]], [sp]
; CHECK-NEXT: br [[REG]]
  %indirect.goto.dest = phi ptr [ %1, %entry ]
  indirectbr ptr %indirect.goto.dest, [label %ZERO, label %ONE]
}
