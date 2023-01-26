; RUN: opt < %s -passes=reassociate -S | FileCheck %s

; Check that reassociate pass now salvages debug info when dropping instructions.

define hidden i32 @main(i32 %argc, ptr %argv) {
entry:
  ; CHECK: call void @llvm.dbg.value(metadata i32 %argc, metadata [[VAR_B:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value))
  %add = add nsw i32 %argc, 1, !dbg !26
  call void @llvm.dbg.value(metadata i32 %add, metadata !22, metadata !DIExpression()), !dbg !25
  %add1 = add nsw i32 %argc, %add, !dbg !27
  ret i32 %add1, !dbg !28
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 10.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "test2.cpp", directory: "C:\")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 10.0.0"}
!8 = distinct !DISubprogram(name: "main", scope: !9, file: !9, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !18)
!9 = !DIFile(filename: "./test2.cpp", directory: "C:\")
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !13, !14}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !12)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !17)
!17 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!18 = !{!19, !20, !21, !22, !23, !24}
!19 = !DILocalVariable(name: "argc", arg: 1, scope: !8, file: !9, line: 1, type: !13)
!20 = !DILocalVariable(name: "argv", arg: 2, scope: !8, file: !9, line: 1, type: !14)
!21 = !DILocalVariable(name: "a", scope: !8, file: !9, line: 2, type: !12)
; CHECK: [[VAR_B]] = !DILocalVariable(name: "b"
!22 = !DILocalVariable(name: "b", scope: !8, file: !9, line: 3, type: !12)
!23 = !DILocalVariable(name: "to_return", scope: !8, file: !9, line: 4, type: !12)
!24 = !DILocalVariable(name: "result", scope: !8, file: !9, line: 5, type: !12)
!25 = !DILocation(line: 0, scope: !8)
!26 = !DILocation(line: 3, scope: !8)
!27 = !DILocation(line: 4, scope: !8)
!28 = !DILocation(line: 6, scope: !8)
