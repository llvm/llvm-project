; Test checks if llvm-dwarfdump prints the correct PTX virtual register name strings
; (e.g. %rd1) instead of a hex encoding in DW_OP_regx.

; File t.cubin is generated using the following steps.
; llc < %s -o t.ptx
; ptxas t.ptx -o t.cubin

; RUN: llc < %s -mtriple=nvptx64 | FileCheck %s --check-prefix PTX
; prebuilt debug ptx-register-manes.cubin must be kept next to this file.
; RUN: llvm-dwarfdump %S/ptx-register-names.cubin | FileCheck %s --check-prefix DWARF

; PTX: .debug_info
; PTX: .b8 6                                   // DW_AT_location
; ULEB after DW_OP_regx decodes to 0x25726431 = NVPTX packed '%rd1' (encodeRegisterForDwarf).
; PTX: .b8 144
; PTX: .b8 177
; PTX: .b8 200
; PTX: .b8 201
; PTX: .b8 171
; PTX: .b8 2
; PTX: .b8 97                                  // DW_AT_name
; PTX: .b8 6                                   // DW_AT_location
; ULEB after DW_OP_regx decodes to 0x25726432 = NVPTX packed '%rd2' (same encoding as %rd1).
; PTX: .b8 144
; PTX: .b8 178
; PTX: .b8 200
; PTX: .b8 201
; PTX: .b8 171
; PTX: .b8 2
; PTX: .b8 98                                  // DW_AT_name

; DWARF: DW_OP_regx %rd1
; DWARF: DW_OP_regx %rd2
; DWARF-NOT: DW_OP_regx 0x25726431
; DWARF-NOT: DW_OP_regx 0x25726432

@global = common global double 0.000000e+00, align 8, !dbg !0

define i32 @f(double %a, double %b) !dbg !12 {
entry:
  call void @llvm.dbg.value(metadata double %a, metadata !17, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata double %b, metadata !18, metadata !DIExpression()), !dbg !19
  %add = fadd double %a, %b, !dbg !20
  store double %add, ptr @global, align 8, !dbg !20
  tail call void @clobber(), !dbg !21
  ret i32 1, !dbg !22
}

define void @clobber(){
  ret void
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "global", scope: !2, file: !3, line: 4, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "test.cu", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!7 = !{i32 2, !"Dwarf Version", i32 2}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 1, !"min_enum_size", i32 4}
!11 = !{!""}
!12 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 6, type: !13, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !16)
!13 = !DISubroutineType(types: !14)
!14 = !{!15, !6, !6}
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !{!17, !18}
!17 = !DILocalVariable(name: "a", arg: 1, scope: !12, file: !3, line: 6, type: !6)
!18 = !DILocalVariable(name: "b", arg: 2, scope: !12, file: !3, line: 6, type: !6)
!19 = !DILocation(line: 0, scope: !12)
!20 = !DILocation(line: 7, scope: !12)
!21 = !DILocation(line: 8, scope: !12)
!22 = !DILocation(line: 9, scope: !12)