; ModuleID = '/Volumes/Data/apple-internal/llvm/tools/clang/test/Modules/debug-info-moduleimport.m'
; RUN: llc %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; CHECK: DW_TAG_module
; CHECK-NEXT: DW_AT_name {{.*}}"DebugModule"
; CHECK-NEXT: DW_AT_LLVM_config_macros {{.*}}"-DMODULES=0"
; CHECK-NEXT: DW_AT_LLVM_include_path {{.*}}"/llvm/tools/clang/test/Modules/Inputs"
; CHECK-NEXT: DW_AT_LLVM_isysroot {{.*}}"/"

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !1, producer: "LLVM version 3.7.0", isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, enums: !2, retainedTypes: !2, subprograms: !2, globals: !2, imports: !3)
!1 = !DIFile(filename: "/llvm/tools/clang/test/Modules/<stdin>", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !5, line: 5)
!5 = !DIModule(scope: null, name: "DebugModule", configMacros: "-DMODULES=0", includePath: "/llvm/tools/clang/test/Modules/Inputs", isysroot: "/")
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"LLVM version 3.7.0"}
