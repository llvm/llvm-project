; RUN: llc -mtriple=x86_64-pc-windows-msvc -O0 -filetype=asm < %s | FileCheck --check-prefixes=CHECK,X86-64 %s
; RUN: llc -mtriple=i686-pc-windows-msvc -O0 -filetype=asm < %s | FileCheck --check-prefixes=CHECK,I686 %s

; CHECK: 	.section	.debug_info

; CHECK: # Abbrev {{.*}} DW_TAG_variable
; CHECK: 	.byte	6                               # DW_AT_location
; DW_OP_const4u
; CHECK-NEXT: 	.byte	12
; X86-64-NEXT: 	.long	tls1@SECREL32
; I686-NEXT:  	.long	_tls1@SECREL32
; DW_OP_GNU_push_tls_address
; CHECK-NEXT: 	.byte	224

; CHECK: # Abbrev {{.*}} DW_TAG_variable
; CHECK: 	.byte	6                               # DW_AT_location
; DW_OP_const4u
; CHECK: 	.byte	12
; X86-64-NEXT: 	.long	tls2@SECREL32
; I686-NEXT:  	.long	_tls2@SECREL32
; DW_OP_GNU_push_tls_address
; CHECK: 	.byte	224

; CHECK: # Abbrev {{.*}} DW_TAG_variable
; CHECK: 	.byte	6                               # DW_AT_location
; DW_OP_const4u
; CHECK-NEXT: 	.byte	12
; X86-64-NEXT: 	.long	tls3@SECREL32
; I686-NEXT:  	.long	_tls3@SECREL32
; DW_OP_GNU_push_tls_address
; CHECK-NEXT: 	.byte	224

source_filename = "test/DebugInfo/X86/tls.ll"

@tls1 = thread_local global i32 1, align 4, !dbg !0
@tls2 = thread_local global i32 2, align 4, !dbg !2
@tls3 = thread_local global i64 3, align 8, !dbg !4

!llvm.dbg.cu = !{!9}
!llvm.module.flags = !{!12, !13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "tls1", scope: null, file: !6, line: 1, type: !7, isLocal: false, isDefinition: true)
!2 = !DIGlobalVariableExpression(var: !3, expr: !DIExpression())
!3 = !DIGlobalVariable(name: "tls2", scope: null, file: !6, line: 2, type: !7, isLocal: false, isDefinition: true)
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = !DIGlobalVariable(name: "tls3", scope: null, file: !6, line: 3, type: !8, isLocal: false, isDefinition: true)
!6 = !DIFile(filename: "tls.cpp", directory: "/tmp")
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !DIBasicType(name: "long long", size: 64, align: 64, encoding: DW_ATE_signed)
!9 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !10, retainedTypes: !10, globals: !11, imports: !10)
!10 = !{}
!11 = !{!0,!2,!4}
!12 = !{i32 7, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
