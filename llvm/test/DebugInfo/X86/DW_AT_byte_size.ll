; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=all %t | FileCheck %s

; Checks that we don't emit a size for a pointer type.
; CHECK: DW_TAG_pointer_type
; CHECK-NEXT: DW_AT_type
; CHECK-NOT: DW_AT_byte_size
; CHECK: DW_TAG
; CHECK: .debug_info contents

%struct.A = type { i32 }

define i32 @_Z3fooP1A(%struct.A* %a) nounwind uwtable ssp !dbg !5 {
entry:
  %a.addr = alloca %struct.A*, align 8
  store %struct.A* %a, %struct.A** %a.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.A** %a.addr, metadata !16, metadata !DIExpression()), !dbg !17
  %0 = load %struct.A*, %struct.A** %a.addr, align 8, !dbg !18
  %b = getelementptr inbounds %struct.A, %struct.A* %0, i32 0, i32 0, !dbg !18
  %1 = load i32, i32* %b, align 4, !dbg !18
  ret i32 %1, !dbg !18
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.1 (trunk 150996)", isOptimized: false, emissionKind: 0, file: !20, enums: !1, retainedTypes: !1, subprograms: !3, globals: !1, imports:  !1)
!1 = !{}
!3 = !{!5}
!5 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooP1A", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !20, scope: !6, type: !7)
!6 = !DIFile(filename: "foo.cpp", directory: "/Users/echristo")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !11)
!11 = !DICompositeType(tag: DW_TAG_class_type, name: "A", line: 1, size: 32, align: 32, file: !20, elements: !12)
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 1, size: 32, align: 32, file: !20, scope: !11, baseType: !9)
!16 = !DILocalVariable(name: "a", line: 3, arg: 1, scope: !5, file: !6, type: !10)
!17 = !DILocation(line: 3, column: 13, scope: !5)
!18 = !DILocation(line: 4, column: 3, scope: !19)
!19 = distinct !DILexicalBlock(line: 3, column: 16, file: !20, scope: !5)
!20 = !DIFile(filename: "foo.cpp", directory: "/Users/echristo")
!21 = !{i32 1, !"Debug Info Version", i32 3}
