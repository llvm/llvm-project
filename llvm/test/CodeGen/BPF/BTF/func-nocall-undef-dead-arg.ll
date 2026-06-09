; RUN: llc -mtriple=bpfel -mcpu=v3 -stop-after=prologepilog -o - %s | FileCheck -check-prefixes=MIR %s
; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=BTF %s

; Reproducer for nocall BTF signature filtering with a dead source argument
; described by dbg.value(poison, ...). During lowering that dead parameter
; becomes DBG_VALUE $noreg, which should not count as an alive argument when
; building the filtered BTF FUNC_PROTO.

; MIR: ![[DEAD:[0-9]+]] = !DILocalVariable(name: "dead", arg: 1, scope:
; MIR: DBG_VALUE $noreg, $noreg, ![[DEAD]], !DIExpression()

; BTF:      [1] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; BTF-NEXT: [2] FUNC_PROTO '(anon)' ret_type_id=1 vlen=1
; BTF-NEXT: 	'a1' type_id=1
; BTF-NEXT: [3] FUNC 'sub' type_id=2 linkage=static

define internal i32 @sub(i32 %0) #0 !dbg !7 {
  call void @llvm.dbg.value(metadata i32 poison, metadata !12,
                            metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.value(metadata i32 %0, metadata !13,
                            metadata !DIExpression()), !dbg !14
  ret i32 %0, !dbg !15
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

attributes #0 = { noinline }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/DNE")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "sub", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(cc: DW_CC_nocall, types: !9)
!9 = !{!10, !10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !13}
!12 = !DILocalVariable(name: "dead", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!13 = !DILocalVariable(name: "a1", arg: 2, scope: !7, file: !1, line: 1, type: !10)
!14 = !DILocation(line: 1, column: 1, scope: !7)
!15 = !DILocation(line: 2, column: 3, scope: !7)
