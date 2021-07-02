; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -force-instr-ref-livedebugvalues=1 < %s | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -filetype=obj | llvm-dwarfdump - | FileCheck %s --check-prefix=DWARF
; RUN: llc -force-instr-ref-livedebugvalues=1 < %s -filetype=obj | llvm-dwarfdump - | FileCheck %s --check-prefix=DWARF

; Values in registers should be clobbered by calls, which use a regmask instead
; of individual register def operands.

; ASM: main: # @main
; ASM: #DEBUG_VALUE: main:argc <- $ecx
; ASM: movl $1, x(%rip)
; ASM: callq clobber
; ASM-NEXT: [[argc_range_end:.Ltmp[0-9]+]]:
; ASM: #DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $ecx

; argc is the first debug location.
; ASM: .Ldebug_loc1:
; ASM-NEXT: .quad   .Lfunc_begin0-.Lfunc_begin0
; ASM-NEXT: .quad   [[argc_range_end]]-.Lfunc_begin0
; ASM-NEXT: .short  1                       # Loc expr size
; ASM-NEXT: .byte   82                      # super-register DW_OP_reg2

; argc is the first formal parameter.
; DWARF: .debug_info contents:
; DWARF:  DW_TAG_formal_parameter
; DWARF-NEXT:    DW_AT_location ({{0x.*}}
; DWARF-NEXT:    [0x0000000000000000, 0x0000000000000013): DW_OP_reg2 RCX
; DWARF-NEXT:    [0x0000000000000013, 0x0000000000000043): DW_OP_GNU_entry_value(DW_OP_reg2 RCX), DW_OP_stack_value
; DWARF-NEXT:    DW_AT_name ("argc")

; ModuleID = 't.cpp'
source_filename = "test/DebugInfo/X86/dbg-value-regmask-clobber.ll"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

@x = common global i32 0, align 4, !dbg !0

; Function Attrs: nounwind uwtable
define i32 @main(i32 %argc, i8** nocapture readnone %argv) #0 !dbg !12 {
entry:
  tail call void @llvm.dbg.value(metadata i8** %argv, metadata !19, metadata !21), !dbg !22
  tail call void @llvm.dbg.value(metadata i32 %argc, metadata !20, metadata !21), !dbg !23
  store volatile i32 1, i32* @x, align 4, !dbg !24, !tbaa !25
  tail call void @clobber() #2, !dbg !29
  store volatile i32 2, i32* @x, align 4, !dbg !30, !tbaa !25
  %0 = load volatile i32, i32* @x, align 4, !dbg !31, !tbaa !25
  %tobool = icmp eq i32 %0, 0, !dbg !31
  br i1 %tobool, label %if.else, label %if.then, !dbg !33

if.then:                                          ; preds = %entry
  store volatile i32 3, i32* @x, align 4, !dbg !34, !tbaa !25
  br label %if.end, !dbg !36

if.else:                                          ; preds = %entry

  store volatile i32 4, i32* @x, align 4, !dbg !37, !tbaa !25
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret i32 0, !dbg !39
}

declare void @clobber()

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 3.9.0 (trunk 260617) (llvm/trunk 260619)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !7)
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"PIC Level", i32 2}
!11 = !{!"clang version 3.9.0 (trunk 260617) (llvm/trunk 260619)"}
!12 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 4, type: !13, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !18)
!13 = !DISubroutineType(types: !14)
!14 = !{!7, !7, !15}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64, align: 64)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64, align: 64)
!17 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!18 = !{!19, !20}
!19 = !DILocalVariable(name: "argv", arg: 2, scope: !12, file: !3, line: 4, type: !15)
!20 = !DILocalVariable(name: "argc", arg: 1, scope: !12, file: !3, line: 4, type: !7)
!21 = !DIExpression()
!22 = !DILocation(line: 4, column: 27, scope: !12)
!23 = !DILocation(line: 4, column: 14, scope: !12)
!24 = !DILocation(line: 5, column: 5, scope: !12)
!25 = !{!26, !26, i64 0}
!26 = !{!"int", !27, i64 0}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C/C++ TBAA"}
!29 = !DILocation(line: 6, column: 3, scope: !12)
!30 = !DILocation(line: 7, column: 5, scope: !12)
!31 = !DILocation(line: 8, column: 7, scope: !32)
!32 = distinct !DILexicalBlock(scope: !12, file: !3, line: 8, column: 7)
!33 = !DILocation(line: 8, column: 7, scope: !12)
!34 = !DILocation(line: 9, column: 7, scope: !35)
!35 = distinct !DILexicalBlock(scope: !32, file: !3, line: 8, column: 10)
!36 = !DILocation(line: 10, column: 3, scope: !35)
!37 = !DILocation(line: 11, column: 7, scope: !38)
!38 = distinct !DILexicalBlock(scope: !32, file: !3, line: 10, column: 10)
!39 = !DILocation(line: 13, column: 1, scope: !12)

