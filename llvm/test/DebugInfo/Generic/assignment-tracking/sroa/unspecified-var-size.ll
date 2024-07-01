; RUN: opt -S %s -passes=sroa -o - | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -S %s -passes=sroa -o - | FileCheck %s

;; $ cat test.cpp
;; #include <cstddef>
;; void fun(std::nullptr_t) {}
;;
;; Check that migrateDebugInfo doesn't crash when encountering an alloca for a
;; variable with a type of unspecified size (e.g. DW_TAG_unspecified_type).

; CHECK: #dbg_value(ptr %0,{{.+}}, !DIExpression(),
;; There should be no new fragment and the value component should remain as %0.

define dso_local void @_Z3funDn(ptr %0) #0 !dbg !14 {
entry:
  %.addr = alloca ptr, align 8, !DIAssignID !22
  call void @llvm.dbg.assign(metadata i1 undef, metadata !21, metadata !DIExpression(), metadata !22, metadata ptr %.addr, metadata !DIExpression()), !dbg !23
  store ptr %0, ptr %.addr, align 8, !DIAssignID !28
  call void @llvm.dbg.assign(metadata ptr %0, metadata !21, metadata !DIExpression(), metadata !28, metadata ptr %.addr, metadata !DIExpression()), !dbg !23
  ret void, !dbg !29
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11, !12, !1000}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, imports: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !6, file: !9, line: 56)
!5 = !DINamespace(name: "std", scope: null)
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "max_align_t", file: !7, line: 24, baseType: !8)
!7 = !DIFile(filename: "clang/12.0.0/include/__stddef_max_align_t.h", directory: "/")
!8 = !DICompositeType(tag: DW_TAG_structure_type, file: !7, line: 19, size: 256, flags: DIFlagFwdDecl, identifier: "_ZTS11max_align_t")
!9 = !DIFile(filename: "include/c++/7.5.0/cstddef", directory: "")
!10 = !{i32 7, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 12.0.0"}
!14 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funDn", scope: !1, file: !1, line: 20, type: !15, scopeLine: 20, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !20)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17}
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "nullptr_t", scope: !5, file: !18, line: 235, baseType: !19)
!18 = !DIFile(filename: "include/x86_64-linux-gnu/c++/7.5.0/bits/c++config.h", directory: "")
!19 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
!20 = !{!21}
!21 = !DILocalVariable(arg: 1, scope: !14, file: !1, line: 20, type: !17)
!22 = distinct !DIAssignID()
!23 = !DILocation(line: 0, scope: !14)
!28 = distinct !DIAssignID()
!29 = !DILocation(line: 20, column: 27, scope: !14)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
