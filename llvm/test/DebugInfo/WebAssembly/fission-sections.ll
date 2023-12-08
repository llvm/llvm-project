; RUN: llc -split-dwarf-file=baz.dwo -split-dwarf-output=%t.dwo  -O0 %s -mtriple=wasm32-unknown-unknown -filetype=obj -o %t
; RUN: llvm-objdump -h %t | FileCheck --check-prefix=OBJ %s
; RUN: llvm-objdump -h %t.dwo | FileCheck --check-prefix=DWO %s


; This test is derived from test/DebugInfo/X86/fission-cu.ll
; But it checks that the output objects have the expected sections

source_filename = "test/DebugInfo/WebAssembly/fission-cu.ll"

@a = global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!7}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "a", scope: null, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "baz.c", directory: "/usr/local/google/home/echristo/tmp")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 3.3 (trunk 169021) (llvm/trunk 169020)", isOptimized: false, runtimeVersion: 0, splitDebugFilename: "baz.dwo", emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0}
!7 = !{i32 1, !"Debug Info Version", i32 3}

; CHECK-LABEL: Sections:

; OBJ: Idx Name
; OBJ-NEXT: 0 IMPORT
; OBJ-NEXT: DATACOUNT
; OBJ-NEXT: DATA
; OBJ-NEXT: .debug_abbrev
; OBJ-NEXT: .debug_info
; OBJ-NEXT: .debug_str
; OBJ-NEXT: .debug_addr
; OBJ-NEXT: .debug_pubnames
; OBJ-NEXT: .debug_pubtypes
; OBJ-NEXT: .debug_line
; OBJ-NEXT: linking


; DWO: Idx Name
; DWO-NOT: IMPORT
; DWO-NOT: DATA
; DWO: 0 .debug_str.dwo
; DWO-NEXT: .debug_str_offsets.dwo
; DWO-NEXT: .debug_info.dwo
; DWO-NEXT: .debug_abbrev.dwo
; DWO-NEXT: producers
