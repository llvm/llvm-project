; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK %s

; Simulates DeadArgElimination on a function with btf_decl_tag annotations:
;   static __noinline int sub(int dead __attribute__((btf_decl_tag("dead_tag"))),
;                             int a1 __attribute__((btf_decl_tag("a1_tag"))),
;                             int a2 __attribute__((btf_decl_tag("a2_tag")))) {
;     return a1 + a2;
;   }
; DAE removes 'dead' (arg 1). BTF should emit FUNC_PROTO with (a1, a2)
; and remap decl_tag component indices: a1 becomes param 0, a2 becomes param 1.
; The dead arg's annotation should be dropped.

; CHECK:      [1] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-NEXT: [2] FUNC_PROTO '(anon)' ret_type_id=1 vlen=2
; CHECK-NEXT: 	'a1' type_id=1
; CHECK-NEXT: 	'a2' type_id=1
; CHECK-NEXT: [3] FUNC 'sub' type_id=2 linkage=static
; CHECK-NEXT: [4] DECL_TAG 'a1_tag' type_id=3 component_idx=0
; CHECK-NEXT: [5] DECL_TAG 'a2_tag' type_id=3 component_idx=1

define internal i32 @sub(i32 %0, i32 %1) #0 !dbg !7 {
  call void @llvm.dbg.value(metadata i32 %0, metadata !15, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 %1, metadata !16, metadata !DIExpression()), !dbg !17
  %3 = add i32 %0, %1, !dbg !18
  ret i32 %3, !dbg !19
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
!9 = !{!10, !10, !10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !15, !16}
!12 = !DILocalVariable(name: "dead", arg: 1, scope: !7, file: !1, line: 1, type: !10, annotations: !13)
!13 = !{!14}
!14 = !{!"btf_decl_tag", !"dead_tag"}
!15 = !DILocalVariable(name: "a1", arg: 2, scope: !7, file: !1, line: 1, type: !10, annotations: !20)
!16 = !DILocalVariable(name: "a2", arg: 3, file: !1, line: 1, type: !10, scope: !7, annotations: !22)
!17 = !DILocation(line: 1, column: 1, scope: !7)
!18 = !DILocation(line: 2, column: 10, scope: !7)
!19 = !DILocation(line: 2, column: 3, scope: !7)
!20 = !{!21}
!21 = !{!"btf_decl_tag", !"a1_tag"}
!22 = !{!23}
!23 = !{!"btf_decl_tag", !"a2_tag"}
