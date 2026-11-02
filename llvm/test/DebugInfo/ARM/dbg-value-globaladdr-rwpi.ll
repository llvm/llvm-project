;; Under RWPI a writable global is addressed relative to the static base
;; register, so naming its symbol alone would describe a link-time offset where
;; the source variable holds a runtime address. Check that a variable holding
;; that address keeps the location it is materialized into, while a read-only
;; global, which does keep an absolute address, is still named directly.

; RUN: llc -O2 -mtriple=armv7-none-eabi -relocation-model=rwpi \
; RUN:   -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR \
; RUN:       --implicit-check-not='DBG_VALUE @g'
; RUN: llc -O2 -mtriple=armv7-none-eabi -relocation-model=rwpi -filetype=obj \
; RUN:   < %s | llvm-dwarfdump - | FileCheck %s --check-prefix=DWARF

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

@g = global i32 0, align 4, !dbg !0
@ro = constant i32 7, align 4, !dbg !14

;; The static base register the debugger would have to add is not part of the
;; symbol, and DwarfCompileUnit's construction for it needs an R_ARM_SBREL32
;; relocation, which a location list cannot carry.
; MIR-LABEL: name: writable
; MIR: DBG_VALUE %{{[0-9]+}}, $noreg, ![[#]], !DIExpression()
;
; DWARF-LABEL: DW_AT_name ("writable")
; DWARF: DW_TAG_variable
; DWARF-NEXT: DW_AT_location (indexed (0x0) loclist = 0x{{[0-9a-f]+}}:
; DWARF-NEXT: DW_OP_reg{{[0-9]+}} R{{[0-9]+}})
; DWARF-NEXT: DW_AT_name ("p")
define void @writable() !dbg !9 {
entry:
    #dbg_value(ptr @g, !12, !DIExpression(), !13)
  %box = tail call ptr @alloc(), !dbg !13
  store ptr @g, ptr %box, align 4, !dbg !13
  tail call void @sink(ptr %box), !dbg !13
  ret void, !dbg !13
}

; MIR-LABEL: name: readonly
; MIR: DBG_VALUE @ro, $noreg, ![[#]], !DIExpression()
;
; DWARF-LABEL: DW_AT_name ("readonly")
; DWARF: DW_TAG_variable
; DWARF-NEXT: DW_AT_location (DW_OP_addrx 0x{{[0-9a-f]+}}, DW_OP_stack_value)
; DWARF-NEXT: DW_AT_name ("q")
define void @readonly() !dbg !16 {
entry:
    #dbg_value(ptr @ro, !17, !DIExpression(), !18)
  %box = tail call ptr @alloc(), !dbg !18
  store ptr @ro, ptr %box, align 4, !dbg !18
  tail call void @sink(ptr %box), !dbg !18
  ret void, !dbg !18
}

declare void @sink(ptr)
declare ptr @alloc()

!llvm.module.flags = !{!5, !6}
!llvm.dbg.cu = !{!2}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang", isOptimized: true, emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "t.c", directory: "/")
!4 = !{!0, !14}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !DISubroutineType(types: !{null})
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = distinct !DISubprogram(name: "writable", scope: !3, file: !3, line: 5, type: !7, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 32)
!11 = distinct !DILexicalBlock(scope: !9, file: !3, line: 6, column: 1)
!12 = !DILocalVariable(name: "p", scope: !11, file: !3, line: 6, type: !10)
!13 = !DILocation(line: 6, column: 1, scope: !11)
!14 = !DIGlobalVariableExpression(var: !15, expr: !DIExpression())
!15 = distinct !DIGlobalVariable(name: "ro", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true)
!16 = distinct !DISubprogram(name: "readonly", scope: !3, file: !3, line: 12, type: !7, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!17 = !DILocalVariable(name: "q", scope: !19, file: !3, line: 13, type: !10)
!18 = !DILocation(line: 13, column: 1, scope: !19)
!19 = distinct !DILexicalBlock(scope: !16, file: !3, line: 13, column: 1)
