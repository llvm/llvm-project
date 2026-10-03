; RUN: llc < %s -mtriple=thumbv5e-none-linux-gnueabi --float-abi=soft -verify-machineinstrs -o /dev/null
; This used to crash in Thumb1InstrInfo::copyPhysReg when computing liveness
; for a pre-v6 low GPR copy. The backward liveness walk from MBB.end() to the
; COPY instruction was calling LiveRegUnits::stepBackward on DBG_VALUE
; instructions, which asserts that debug instructions must not affect liveness
; calculation.

define void @foo(ptr %res) !dbg !3 {
entry:
  #dbg_declare(ptr %res, !4, !DIExpression(), !5)
  call void @llvm.memcpy.p0.p0.i32(ptr %res, ptr null, i32 1, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i32, i1 immarg)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, type: !DISubroutineType(types: !{}), unit: !0)
!4 = !DILocalVariable(scope: !3)
!5 = !DILocation(line: 1, scope: !3)
