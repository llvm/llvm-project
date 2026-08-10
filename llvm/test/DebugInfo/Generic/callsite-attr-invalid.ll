; RUN: opt -passes=verify < %s 2>&1 | FileCheck %s

; CHECK: DIFlagAllCallsDescribed must be attached to a definition
; CHECK: warning: ignoring invalid debug info

; Source:
;   struct A { ~A(); };
;   void foo() { A x; }

%struct.A = type { i8 }

define void @_Z3foov() !dbg !8 {
entry:
  %x = alloca %struct.A, align 1
  call void @llvm.dbg.declare(metadata ptr %x, metadata !12, metadata !DIExpression()), !dbg !19
  call void @_ZN1AD1Ev(ptr %x) #3, !dbg !20
  ret void, !dbg !20
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare void @_ZN1AD1Ev(ptr)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "-", directory: "/Users/vsk/src/builds/llvm-project-tailcall-RA")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 8.0.0 "}
!8 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !9, file: !9, line: 1, type: !10, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !DIFile(filename: "<stdin>", directory: "/Users/vsk/src/builds/llvm-project-tailcall-RA")
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !DILocalVariable(name: "x", scope: !8, file: !9, line: 1, type: !13)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !9, line: 1, size: 8, flags: DIFlagTypePassByReference, elements: !14, identifier: "_ZTS1A")
!14 = !{!15}
!15 = !DISubprogram(name: "~A", scope: !13, file: !9, line: 1, type: !16, isLocal: false, isDefinition: false, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, isOptimized: false)
!16 = !DISubroutineType(types: !17)
!17 = !{null, !18}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!19 = !DILocation(line: 1, column: 36, scope: !8)
!20 = !DILocation(line: 1, column: 39, scope: !8)
