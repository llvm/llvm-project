; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+amx-int8,+avx512f -verify-machineinstrs | FileCheck %s

define void @insert_at_block_start_debug(i16 %r, i16 %c, i1 %cond, ptr %p,
; CHECK-LABEL: insert_at_block_start_debug:
; CHECK:       .LBB{{[0-9]+}}_2: # %b
; CHECK-NEXT:    ldtilecfg -{{[0-9]+}}(%rsp)
; CHECK-NEXT:    tilezero %tmm1
                                         ptr %q) !dbg !5 {
entry:
  br i1 %cond, label %a, label %b

a:
  %r2 = load i16, ptr %p
  %t = call x86_amx @llvm.x86.tilezero.internal(i16 %r2, i16 %c)
  call void @llvm.x86.tilestored64.internal(i16 %r2, i16 %c, ptr %q, i64 64,
                                            x86_amx %t)
  br label %exit

b:
    #dbg_value(i16 %c, !7, !DIExpression(), !8)
  %t2 = call x86_amx @llvm.x86.tilezero.internal(i16 %c, i16 %c)
  call void @llvm.x86.tilestored64.internal(i16 %c, i16 %c, ptr %q, i64 64,
                                            x86_amx %t2)
  br label %exit

exit:
  ret void
}

define void @insert_at_block_start(i16 %r, i16 %c, i1 %cond, ptr %p,
                                   ptr %q) {
; CHECK-LABEL: insert_at_block_start:
; CHECK:       .LBB{{[0-9]+}}_2: # %b
; CHECK-NEXT:    ldtilecfg -{{[0-9]+}}(%rsp)
; CHECK-NEXT:    tilezero %tmm1
entry:
  br i1 %cond, label %a, label %b

a:
  %r2 = load i16, ptr %p
  %t = call x86_amx @llvm.x86.tilezero.internal(i16 %r2, i16 %c)
  call void @llvm.x86.tilestored64.internal(i16 %r2, i16 %c, ptr %q, i64 64,
                                            x86_amx %t)
  br label %exit

b:
  %t2 = call x86_amx @llvm.x86.tilezero.internal(i16 %c, i16 %c)
  call void @llvm.x86.tilestored64.internal(i16 %c, i16 %c, ptr %q, i64 64,
                                            x86_amx %t2)
  br label %exit

exit:
  ret void
}

declare x86_amx @llvm.x86.tilezero.internal(i16, i16)
declare void @llvm.x86.tilestored64.internal(i16, i16, ptr, i64, x86_amx)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "insert_at_block_start_debug", scope: !1, file: !1, line: 1, type: !6, unit: !0, spFlags: DISPFlagDefinition, retainedNodes: !{})
!6 = !DISubroutineType(types: !{})
!7 = !DILocalVariable(name: "c", scope: !5, file: !1, line: 1, type: !9)
!8 = !DILocation(line: 1, scope: !5)
!9 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
