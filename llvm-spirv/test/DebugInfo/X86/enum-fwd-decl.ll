; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-mem2reg=false
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll

; RUN: llc -O0 -mtriple=x86_64-apple-darwin %t.ll -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s

source_filename = "test/DebugInfo/X86/enum-fwd-decl.ll"

@e = global i16 0, align 2, !dbg !0

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!7}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "e", scope: null, file: !2, line: 2, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "foo.cpp", directory: "/tmp")
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E", file: !2, line: 1, size: 16, align: 16, flags: DIFlagFwdDecl)
!4 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.2 (trunk 165274) (llvm/trunk 165272)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
; CHECK: DW_TAG_enumeration_type
; CHECK-NEXT: DW_AT_name
; CHECK-NEXT: DW_AT_byte_size
; CHECK-NEXT: DW_AT_declaration
!6 = !{!0}
!7 = !{i32 1, !"Debug Info Version", i32 3}
target triple = "spir64-unknown-unknown"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
