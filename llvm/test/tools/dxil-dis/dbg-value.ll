; RUN: llc --filetype=obj %s -o  - | dxil-dis -o - | FileCheck %s

target triple = "dxil-pc-shadermodel6.3-library"

; CHECK: define i32 @main(i32 %argc, i8* %argv)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.dbg.value(metadata i32 %argc, i64 0, metadata ![[LV_ARGC:[0-9]+]], metadata ![[EXPR:[0-9]+]]), !dbg ![[LOC_ARGC:[0-9]+]]
; CHECK-NEXT:   call void @llvm.dbg.value(metadata i8* %argv, i64 0, metadata ![[LV_ARGV:[0-9]+]], metadata ![[EXPR:[0-9]+]]), !dbg ![[LOC_ARGV:[0-9]+]]
; CHECK-NEXT:   ret i32 0
; CHECK-NEXT: }

define i32 @main(i32 %argc, ptr %argv) !dbg !25 {
entry:
  call void @llvm.dbg.value(metadata i32 %argc, metadata !31, metadata !33), !dbg !34
  call void @llvm.dbg.value(metadata ptr %argv, metadata !32, metadata !33), !dbg !35
  ret i32 0
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

; CHECK-DAG: ![[EXPR]] = !DIExpression()
; CHECK-DAG: ![[LV_ARGC]] = !DILocalVariable(tag: DW_TAG_arg_variable, name: "argc", arg: 1, scope: !{{.*}}, file: !{{.*}}, line: 4, type: !{{.*}})
; CHECK-DAG: ![[LV_ARGV]] = !DILocalVariable(tag: DW_TAG_arg_variable, name: "argv", arg: 2, scope: !{{.*}}, file: !{{.*}}, line: 4, type: !{{.*}})
; CHECK-DAG: ![[LOC_ARGC]] = !DILocation(line: 4, column: 14, scope: !{{.*}})
; CHECK-DAG: ![[LOC_ARGV]] = !DILocation(line: 4, column: 27, scope: !{{.*}})

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15, !16}

!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "test.c", directory: "/")
!4 = !{}
!5 = !{}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!25 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 4, type: !26, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !30)
!26 = !DISubroutineType(types: !27)
!27 = !{!10, !10, !28}
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !29, size: 64)
!29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!30 = !{!31, !32}
!31 = !DILocalVariable(name: "argc", arg: 1, scope: !25, file: !3, line: 4, type: !10)
!32 = !DILocalVariable(name: "argv", arg: 2, scope: !25, file: !3, line: 4, type: !28)
!33 = !DIExpression()
!34 = !DILocation(line: 4, column: 14, scope: !25)
!35 = !DILocation(line: 4, column: 27, scope: !25)
