; RUN: split-file %s %t
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -dwarf-version=5 -filetype=obj -o - %t/skip.ll | llvm-dwarfdump -v - | FileCheck %s --check-prefix=SKIP-NATIVE
; RUN: not --crash llc -mtriple=x86_64-unknown-linux-gnu -dwarf-version=4 -filetype=obj -o /dev/null %t/skip.ll 2>&1 | FileCheck %s --check-prefix=SKIP-LEGACY
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -dwarf-version=5 -filetype=obj -o - %t/bra.ll | llvm-dwarfdump -v - | FileCheck %s --check-prefix=BRA-NATIVE
; RUN: not --crash llc -mtriple=x86_64-unknown-linux-gnu -dwarf-version=4 -filetype=obj -o /dev/null %t/bra.ll 2>&1 | FileCheck %s --check-prefix=BRA-LEGACY
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -dwarf-version=5 -filetype=obj -o - %t/label.ll | llvm-dwarfdump -v - | FileCheck %s --check-prefix=LABEL-NATIVE
; RUN: not --crash llc -mtriple=x86_64-unknown-linux-gnu -dwarf-version=4 -filetype=obj -o /dev/null %t/label.ll 2>&1 | FileCheck %s --check-prefix=LABEL-LEGACY

; DWARF 5 emits DW_OP_convert directly, so labels and branches can appear
; between conversions. With DWARF 4 we may defer one conversion until the next;
; if a label, branch, or skip splits the pair, CodeGen reports an error.

; SKIP-NATIVE: DW_AT_location [DW_FORM_exprloc] (DW_OP_breg5 RDI+0, DW_OP_convert {{.*}} "DW_ATE_signed_32", DW_OP_skip +5, DW_OP_convert {{.*}} "DW_ATE_signed_64")
; SKIP-LEGACY: LLVM ERROR: cannot lower DW_OP_LLVM_convert across DW_OP_LLVM_skip without DW_OP_convert support

; BRA-NATIVE: DW_AT_location [DW_FORM_exprloc] (DW_OP_breg5 RDI+0, DW_OP_convert {{.*}} "DW_ATE_signed_32", DW_OP_dup, DW_OP_bra +5, DW_OP_convert {{.*}} "DW_ATE_signed_32")
; BRA-LEGACY: LLVM ERROR: cannot lower DW_OP_LLVM_convert across DW_OP_LLVM_bra without DW_OP_convert support

; LABEL-NATIVE: DW_AT_location [DW_FORM_exprloc] (DW_OP_breg5 RDI+0, DW_OP_convert {{.*}} "DW_ATE_signed_32")
; LABEL-LEGACY: LLVM ERROR: cannot lower DW_OP_LLVM_convert across DW_OP_LLVM_label without DW_OP_convert support

;--- skip.ll
define void @skip(i64 %x) !dbg !5 {
entry:
  #dbg_value(i64 %x, !9,
             !DIExpression(DW_OP_LLVM_convert, 32, DW_ATE_signed,
                           DW_OP_LLVM_skip, 1,
                           DW_OP_LLVM_convert, 64, DW_ATE_signed,
                           DW_OP_LLVM_label, 1), !10)
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1,
                             emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/")
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "skip", scope: !1, type: !6,
                            spFlags: DISPFlagDefinition, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!9 = !DILocalVariable(name: "skip", arg: 1, scope: !5, type: !11)
!10 = !DILocation(line: 1, column: 1, scope: !5)
!11 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)

;--- label.ll
define void @label(i64 %x) !dbg !5 {
entry:
  #dbg_value(i64 %x, !9,
             !DIExpression(DW_OP_LLVM_convert, 32, DW_ATE_signed,
                           DW_OP_LLVM_label, 3), !10)
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1,
                             emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/")
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "label", scope: !1, type: !6,
                            spFlags: DISPFlagDefinition, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!9 = !DILocalVariable(name: "label", arg: 1, scope: !5, type: !11)
!10 = !DILocation(line: 1, column: 1, scope: !5)
!11 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)

;--- bra.ll
define void @bra(i64 %x) !dbg !5 {
entry:
  #dbg_value(i64 %x, !9,
             !DIExpression(DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_dup,
                           DW_OP_LLVM_bra, 2,
                           DW_OP_LLVM_convert, 32, DW_ATE_signed,
                           DW_OP_LLVM_label, 2), !10)
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1,
                             emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/")
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "bra", scope: !1, type: !6,
                            spFlags: DISPFlagDefinition, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!9 = !DILocalVariable(name: "bra", arg: 1, scope: !5, type: !11)
!10 = !DILocation(line: 1, column: 1, scope: !5)
!11 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
