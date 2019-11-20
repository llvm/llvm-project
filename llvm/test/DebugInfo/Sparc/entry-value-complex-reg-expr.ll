; RUN: llc -debug-entry-values -filetype=asm -o - %s | FileCheck %s

; Verify that the entry value covers both of the DW_OP_regx pieces. Previously
; the size operand of the entry value would be hardcoded to one.
;
; XXX: Is this really what should be emitted, or should we instead emit one
; entry value operation per DW_OP_regx? GDB can currently not understand
; entry values containing complex expressions like this.

target datalayout = "E-m:e-i64:64-n32:64-S128"
target triple = "sparc64"

; Based on the following C reproducer:
;
; volatile long double global;
; extern void clobber();
; int foo(long double p) {
;   global = p;
;   clobber();
;   return 123;
; }

; CHECK:      .byte   243       ! DW_OP_GNU_entry_value
; CHECK-NEXT: .byte   8         ! 8
; CHECK-NEXT: .byte   144       ! sub-register DW_OP_regx
; CHECK-NEXT: .byte   72        ! 72
; CHECK-NEXT: .byte   147       ! DW_OP_piece
; CHECK-NEXT: .byte   8         ! 8
; CHECK-NEXT: .byte   144       ! sub-register DW_OP_regx
; CHECK-NEXT: .byte   73        ! 73
; CHECK-NEXT: .byte   147       ! DW_OP_piece
; CHECK-NEXT: .byte   8         ! 8
; CHECK-NEXT: .byte   159       ! DW_OP_stack_value

@global = common global fp128 0xL00000000000000000000000000000000, align 16, !dbg !0

; Function Attrs: nounwind
define signext i32 @foo(fp128 %p) #0 !dbg !12 {
entry:
  call void @llvm.dbg.value(metadata fp128 %p, metadata !17, metadata !DIExpression()), !dbg !18
  store volatile fp128 %p, fp128* @global, align 16, !dbg !19
  tail call void @clobber(), !dbg !20
  ret i32 123, !dbg !21
}

declare void @clobber()

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "global", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 10.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "entry-value-complex-reg-expr.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !7)
!7 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{!"clang version 10.0.0"}
!12 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 3, type: !13, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !16)
!13 = !DISubroutineType(types: !14)
!14 = !{!15, !7}
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !{!17}
!17 = !DILocalVariable(name: "p", arg: 1, scope: !12, file: !3, line: 3, type: !7)
!18 = !DILocation(line: 0, scope: !12)
!19 = !DILocation(line: 4, scope: !12)
!20 = !DILocation(line: 5, scope: !12)
!21 = !DILocation(line: 6, scope: !12)
