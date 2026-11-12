; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK %s

; DW_CC_nocall function where no arguments were removed (ArgumentPromotion
; changed struct big to i64 but all source args survived). Since no args
; were eliminated, there is nothing to optimize — fall back to the original
; source prototype.

; CHECK:      [1] STRUCT 'big' size=16 vlen=2
; CHECK-NEXT: 	'x' type_id=2 bits_offset=0
; CHECK-NEXT: 	'y' type_id=2 bits_offset=64
; CHECK-NEXT: [2] INT 'long' size=8 bits_offset=0 nr_bits=64 encoding=SIGNED
; CHECK-NEXT: [3] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-NEXT: [4] FUNC_PROTO '(anon)' ret_type_id=3 vlen=2
; CHECK-NEXT: 	's' type_id=1
; CHECK-NEXT: 	'a' type_id=3
; CHECK-NEXT: [5] FUNC 'foo' type_id=4 linkage=static

define internal i32 @foo(i64 %0, i32 %1) #0 !dbg !7 {
  call void @llvm.dbg.value(metadata i64 %0, metadata !17,
                            metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg !19
  call void @llvm.dbg.value(metadata i64 poison, metadata !17,
                            metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64)), !dbg !19
  call void @llvm.dbg.value(metadata i32 %1, metadata !18, metadata !DIExpression()), !dbg !19
  %3 = trunc i64 %0 to i32, !dbg !20
  %4 = add i32 %3, %1, !dbg !21
  ret i32 %4, !dbg !22
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
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(cc: DW_CC_nocall, types: !9)
!9 = !{!10, !12, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!17, !18}
!12 = !DICompositeType(tag: DW_TAG_structure_type, name: "big", file: !1, line: 1, size: 128, elements: !13)
!13 = !{!14, !15}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !12, file: !1, line: 1, baseType: !16, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !12, file: !1, line: 1, baseType: !16, size: 64, offset: 64)
!16 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!17 = !DILocalVariable(name: "s", arg: 1, scope: !7, file: !1, line: 3, type: !12)
!18 = !DILocalVariable(name: "a", arg: 2, scope: !7, file: !1, line: 3, type: !10)
!19 = !DILocation(line: 3, column: 1, scope: !7)
!20 = !DILocation(line: 4, column: 10, scope: !7)
!21 = !DILocation(line: 4, column: 15, scope: !7)
!22 = !DILocation(line: 4, column: 3, scope: !7)
