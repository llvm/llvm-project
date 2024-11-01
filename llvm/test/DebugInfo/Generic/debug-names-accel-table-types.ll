; XFAIL: target={{.*}}-aix{{.*}}, target={{.*}}-zos{{.*}}
; RUN: %llc_dwarf -debugger-tune=lldb -accel-tables=Dwarf -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump %t | FileCheck %s
; RUN: llvm-dwarfdump -debug-names %t | FileCheck --check-prefix=SAME-NAME %s
; RUN: llvm-dwarfdump -debug-names %t | FileCheck --check-prefix=DIFFERENT-NAME %s
; RUN: llvm-dwarfdump -debug-names %t | FileCheck --check-prefix=UNIQUE-DIFFERENT-NAME %s
; RUN: llvm-dwarfdump -debug-names -verify %t | FileCheck --check-prefix=VERIFY %s


; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_name ("SameName")
; CHECK: DW_AT_linkage_name ("SameName")

; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_name ("DifferentName")
; CHECK: DW_AT_linkage_name ("UniqueDifferentName")

; The name count should be 5 (the two variables, the two human readable names, one mangled name).
; SAME-NAME: Name count: 5

; The accelarator should only have one entry for the three following names.
; SAME-NAME: "SameName" 
; SAME-NAME-NOT: "SameName" 

; DIFFERENT-NAME: "DifferentName"
; DIFFERENT-NAME-NOT: "DifferentName"

; UNIQUE-DIFFERENT-NAME: "UniqueDifferentName"
; UNIQUE-DIFFERENT-NAME-NOT: "UniqueDifferentName"

; Verification should succeed.
; VERIFY: No errors.

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

@q = common global i8* null, align 8, !dbg !102
@r = common global i8* null, align 8, !dbg !105

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7}

!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, emissionKind: FullDebug, globals: !5)
!3 = !DIFile(filename: "/tmp/p.c", directory: "/")
!4 = !{}
!5 = !{!102, !105}
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}

; marking the types as Swift is necessary because we only emit the linkage names for Swift types.
!11 = !DICompositeType(tag: DW_TAG_structure_type, name: "SameName", file: !3, size: 64, runtimeLang: DW_LANG_Swift, identifier: "SameName")
!12 = !DICompositeType(tag: DW_TAG_structure_type, name: "DifferentName", file: !3, size: 64, runtimeLang: DW_LANG_Swift, identifier: "UniqueDifferentName")

!102 = !DIGlobalVariableExpression(var: !103, expr: !DIExpression())
!103 = distinct !DIGlobalVariable(name: "q", scope: !2, file: !3, line: 1, type: !11, isLocal: false, isDefinition: true)
!104 = distinct !DIGlobalVariable(name: "r", scope: !2, file: !3, line: 1, type: !12, isLocal: false, isDefinition: true)
!105 = !DIGlobalVariableExpression(var: !104, expr: !DIExpression())
