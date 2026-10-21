; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK %s

; DeadArgElimination on a function with a bool parameter:
;   static __noinline int sub(int dead, _Bool flag, int a) {
;     return flag ? a : 0;
;   }

; CHECK:      [1] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-NEXT: [2] TYPEDEF 'bool' type_id=3
; CHECK-NEXT: [3] INT '_Bool' size=1 bits_offset=0 nr_bits=8 encoding=BOOL
; CHECK-NEXT: [4] FUNC_PROTO '(anon)' ret_type_id=1 vlen=2
; CHECK-NEXT: 	'flag' type_id=2
; CHECK-NEXT: 	'a' type_id=1
; CHECK-NEXT: [5] FUNC 'sub' type_id=4 linkage=static

define internal i32 @sub(i1 zeroext %0, i32 %1) #0 !dbg !7 {
  call void @llvm.dbg.value(metadata i1 %0, metadata !15,
                            metadata !DIExpression(DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 8, DW_ATE_unsigned, DW_OP_stack_value)), !dbg !17
  call void @llvm.dbg.value(metadata i32 %1, metadata !16, metadata !DIExpression()), !dbg !17
  %3 = select i1 %0, i32 %1, i32 0, !dbg !18
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
!9 = !{!10, !10, !13, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!14, !15, !16}
!12 = !DIBasicType(name: "_Bool", size: 8, encoding: DW_ATE_boolean)
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "bool", file: !1, line: 1, baseType: !12)
!14 = !DILocalVariable(name: "dead", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!15 = !DILocalVariable(name: "flag", arg: 2, scope: !7, file: !1, line: 1, type: !13)
!16 = !DILocalVariable(name: "a", arg: 3, scope: !7, file: !1, line: 1, type: !10)
!17 = !DILocation(line: 1, column: 1, scope: !7)
!18 = !DILocation(line: 2, column: 10, scope: !7)
!19 = !DILocation(line: 2, column: 3, scope: !7)
