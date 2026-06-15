; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK %s

; Simulates DeadArgElimination on a function with mixed parameter types:
;   enum color { RED };
;   static __noinline int sub(int dead, int *p, enum color c) {
;     return *p + c;
;   }

; CHECK:      [1] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-NEXT: [2] PTR '(anon)' type_id=1
; CHECK-NEXT: [3] ENUM 'color' encoding=UNSIGNED size=4 vlen=1
; CHECK-NEXT: 	'RED' val=0
; CHECK-NEXT: [4] FUNC_PROTO '(anon)' ret_type_id=1 vlen=2
; CHECK-NEXT: 	'p' type_id=2
; CHECK-NEXT: 	'c' type_id=3
; CHECK-NEXT: [5] FUNC 'sub' type_id=4 linkage=static

define internal i32 @sub(ptr %0, i32 %1) #0 !dbg !7 {
  call void @llvm.dbg.value(metadata ptr %0, metadata !18, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 %1, metadata !19, metadata !DIExpression()), !dbg !20
  %3 = load i32, ptr %0, !dbg !21
  %4 = add i32 %3, %1, !dbg !22
  ret i32 %4, !dbg !23
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
!9 = !{!10, !10, !13, !14}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !18, !19}
!12 = !DILocalVariable(name: "dead", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
; enum color { RED = 0 }
!14 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "color", file: !1, line: 1, baseType: !15, size: 32, elements: !16)
!15 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!16 = !{!17}
!17 = !DIEnumerator(name: "RED", value: 0)
!18 = !DILocalVariable(name: "p", arg: 2, scope: !7, file: !1, line: 1, type: !13)
!19 = !DILocalVariable(name: "c", arg: 3, scope: !7, file: !1, line: 1, type: !14)
!20 = !DILocation(line: 1, column: 1, scope: !7)
!21 = !DILocation(line: 2, column: 10, scope: !7)
!22 = !DILocation(line: 2, column: 15, scope: !7)
!23 = !DILocation(line: 2, column: 3, scope: !7)
