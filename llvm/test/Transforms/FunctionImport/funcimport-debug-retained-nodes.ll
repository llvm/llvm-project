; REQUIRES: asserts,x86-registered-target
; RUN: opt --bitcode-mdindex-threshold=0 -module-summary %s -o %t.bc
; RUN: opt --bitcode-mdindex-threshold=0 -module-summary %p/Inputs/funcimport-debug-retained-nodes.ll -o %t2.bc

; RUN: llvm-lto2 run %t2.bc %t.bc --save-temps -o %t3                       \
; RUN:  -r=%t.bc,main,px -r=%t.bc,func,px -r=%t2.bc,func,x -r=%t2.bc,foo,rx \
; RUN:  --debug-only=bitcode-reader --thinlto-threads=1 2>&1                \
; RUN: | FileCheck --allow-empty --check-prefix=LTO %s                      \
; RUN:  --implicit-check-not='ignoring invalid debug info'                  \
; RUN:  --implicit-check-not='warning'

; RUN: llvm-dis %t3.2.3.import.bc -o - | FileCheck %s                       \
; RUN:  --implicit-check-not='DISubprogram(name: "inlined_out_clone"'       \
; RUN:  --implicit-check-not='DICompositeType({{.*}}, identifier: "local_type"'

; Check that retained nodes of lazy-loaded DISubprograms are cleaned up
; from incorrectly-scoped local types.

; When DebugTypeODRUniquing feature is enabled (e.g. with ThinLTO),
; local DITypes with the same `identifier` values are uniqued in scope
; of LLVM context during metadata loading.
; DISubprograms may reference their local types via `retainedNodes` attribute.
; Thus, during ThinLTO, the final module may end up having multiple
; DISubprograms referencing the same uniqued local type.
; MetadataLoader should clean up retainedNodes lists of DISubprograms from
; such references after loading subprograms and their local types.
; This test checks that such cleanup is done when metadata nodes are loaded
; in lazy fashion without relying on cleanup performed during
; eager function-level or module-level METADATA_BLOCK loading.

; In order to trigger lazy-loading of DISubprogram "inlined_out_clone"
; from module-level METADATA_BLOCK in %p/Inputs/funcimport-debug-retained-nodes.ll:
; 1. The emission of metadata index is forced by setting
; --bitcode-md-index-threshold. If no MD index is emitted in BC file,
; MetadataLoader loads all metadata from a module-level METADATA_BLOCK eagerly.
; 2. The DISubprogram is referenced by locations inlined in two different
; IR functions, thus, it is emitted in module-level METADATA_BLOCK.
; 3. The DISubprogram is not referenced by any local variable of a function,
; so that it is not loaded eagerly when reading function-level METADATA_BLOCK.
; Otherwise, cleanup would be performed on it during function-level
; METADATA_BLOCK loading.
; 4. No other METADATA_BLOCK should be loaded after lazy-loading the target
; DISubprogram, to avoid cleanup being performed later. We want to observe
; the behavior of MetadataLoader when loading the target DISubprogram lazily
; without interference from metadata blocks loaded later. Therefore, @foo from
; %p/Inputs/funcimport-debug-retained-nodes.ll, that follows @func referencing
; the target DISubprogram, is marked as dso_preemptable => unsafe for LTO
; function import.

; This test should pass if, after ThinLTO function import, the final module
; contains two DISubprograms "inlined_out_clone", and none of them reference
; the local type that doesn't belong to them via `retainedNodes`.
; It should fail if `retainedNodes` field of DISubprogram "inlined_out_clone"
; loaded from %p/Inputs/funcimport-debug-retained-nodes.ll references
; DICompositeType from the scope of DISubprogram "inlined_out_clone" from
;  %p/funcimport-debug-retained-nodes.ll (the type that is uniqued
; due to DebugTypeODRUniquing on).

; Check that lazy loading codepath is triggered, the subprogram is cleaned up,
; and MetadataLoaderImpl::resolveLoadedMetadata() is not called after that.
; LTO:      Lazy metadata loading: Resolved loaded metadata. Cleaned up 1 subprogram(s).
; LTO-NOT:  Resolved loaded metadata

; The module %p/funcimport-debug-retained-nodes.ll contains:
; - DICompositeType "local_type", and
; - DISubprogram "inlined_out_clone" with empty retainedNodes list.
; The module %p/Inputs/funcimport-debug-retained-nodes.ll contains:
; - DICompositeType "local_type", and
; - DISubprogram "inlined_out_clone" with "local_type" in its retainedNodes.
; After function import into module %p/funcimport-debug-retained-nodes.ll,
; the output module contains:
; - a single DICompositeType "local_type" that comes from %p/funcimport-debug-retained-nodes.ll
;   (due to ODR-uniquing, "local_type" from %p/Inputs/funcimport-debug-retained-nodes.ll
;   is not imported during function import),
; - DISubprogram "inlined_out_clone" from %p/funcimport-debug-retained-nodes.ll
;   with empty retainedNodes list, and
; - DISubprogram "inlined_out_clone" from %p/Inputs/funcimport-debug-retained-nodes.ll.
;   This test expects its retaiendNodes to be empty, cleaned up from reference
;   to "local_type" from %p/funcimport-debug-retained-nodes.ll (that, without proper
;   cleanup, would occur because of ODR-uniquing). The following check lines ensure that.

; CHECK: ![[ORIGINAL_FILE:[0-9]+]] = !DIFile(filename: "funcimport_debug.c",
; CHECK: ![[EMPTY:[0-9]+]] = !{}
; CHECK: ![[IMPORTED_FILE:[0-9]+]] = !DIFile(filename: "funcimport_debug2.c",
; CHECK: ![[ORIGINAL_SP:[0-9]+]] = distinct !DISubprogram(name: "inlined_out_clone", {{.*}}, file: ![[ORIGINAL_FILE]], {{.*}}, retainedNodes: ![[EMPTY]]
; CHECK: !DICompositeType(tag: DW_TAG_class_type, scope: ![[ORIGINAL_SP]], {{.*}}, identifier: "local_type"
; CHECK: !DISubprogram(name: "inlined_out_clone", {{.*}}, file: ![[IMPORTED_FILE]], {{.*}}, retainedNodes: ![[EMPTY]]

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() !dbg !5 {
entry:
  %a = alloca i8, align 4
    #dbg_declare(ptr %a, !9, !DIExpression(), !12)
  call void (...) @func()
  ret i32 0
}

declare void @func(...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "funcimport_debug.c", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 2, type: !6, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DILocalVariable(name: "foo_ptr", scope: !10, file: !1, line: 4, type: !11)
!10 = distinct !DISubprogram(name: "inlined_out_clone", scope: !1, file: !1, line: 20, type: !6, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!11 = !DICompositeType(tag: DW_TAG_class_type, scope: !10, file: !1, line: 210, size: 8, flags: DIFlagTypePassByValue, elements: !2, identifier: "local_type")
!12 = !DILocation(line: 3, column: 1, scope: !10, inlinedAt: !13)
!13 = !DILocation(line: 3, column: 3, scope: !5)
