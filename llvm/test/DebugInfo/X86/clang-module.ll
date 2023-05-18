; RUN: llc -mtriple=x86_64-apple-darwin %s -o - -filetype=obj \
; RUN:   | llvm-dwarfdump -debug-info - | FileCheck %s

; Clang modules leave Skeleton CUs as breadcrumbs to point from the object files
; to the pcm containing the module's debug info.

; CHECK: Compile Unit:
; CHECK: DW_TAG_compile_unit
; CHECK:   DW_TAG_imported_declaration
; CHECK: Compile Unit:
; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_name {{.*}}Foo
; CHECK: DW_AT_{{.*}}dwo_id {{.*}}04d2
; CHECK: DW_AT_{{.*}}dwo_name {{.*}}"/Foo.pcm"
source_filename = "modules.m"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

!llvm.dbg.cu = !{!0, !6}
!llvm.module.flags = !{!15, !16}
!llvm.linker.options = !{}

!0 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !1, producer: "clang version 5.0.0 (trunk 308357) (llvm/trunk 308379)", emissionKind: FullDebug, imports: !3, sysroot: "/")
!1 = !DIFile(filename: "modules.m", directory: "/")
!3 = !{!4}
!4 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !5, line: 122)
!5 = !DIModule(scope: null, name: "Foo", includePath: ".")
!6 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !7, producer: "clang version 5.0.0 (trunk 308357) (llvm/trunk 308379)", isOptimized: true, runtimeVersion: 0, splitDebugFilename: "/Foo.pcm", emissionKind: FullDebug, dwoId: 1234)
!7 = !DIFile(filename: "Foo", directory: ".")
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
