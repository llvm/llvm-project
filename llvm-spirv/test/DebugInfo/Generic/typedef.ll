; REQUIRES: object-emission

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-mem2reg=false
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll

; RUN: llc -mtriple=%triple -O0 -filetype=obj < %t.ll | llvm-dwarfdump -debug-info - | FileCheck %s

; From source:
; typedef void x;
; x *y;

; Check that a typedef with no DW_AT_type is produced. The absence of a type is used to imply the 'void' type.

; CHECK: DW_TAG_typedef
; CHECK-NOT: DW_AT_type
; CHECK: {{DW_TAG|NULL}}

source_filename = "test/DebugInfo/Generic/typedef.ll"

@y = global i8* null, align 8, !dbg !0

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "y", scope: null, file: !2, line: 2, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "typedef.cpp", directory: "/tmp/dbginfo")
!3 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64, align: 64)
!4 = !DIDerivedType(tag: DW_TAG_typedef, name: "x", file: !2, line: 1, baseType: null)
!5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.5.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !6, retainedTypes: !6, globals: !7, imports: !6)
!6 = !{}
!7 = !{!0}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 1, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.5.0 "}

target triple = "spir64-unknown-unknown"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
