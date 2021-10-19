;; This test checks for emission of DW_OP_implicit_value operation
;; for double type.

; RUN: llc -O0 -debugger-tune=gdb -filetype=obj %s -o -  | llvm-dwarfdump - | FileCheck %s --check-prefixes=CHECK,BOTH
; RUN: llc -O0 -debugger-tune=lldb -filetype=obj %s -o - | llvm-dwarfdump - | FileCheck %s --check-prefixes=CHECK,BOTH

; CHECK: .debug_info contents:
; CHECK: DW_TAG_variable
; CHECK-NEXT:  DW_AT_location        ({{.*}}
; CHECK-NEXT:                     [{{.*}}): DW_OP_implicit_value 0x8 0x1f 0x85 0xeb 0x51 0xb8 0x1e 0x09 0x40)
; CHECK-NEXT:  DW_AT_name    ("d")

; RUN: llc -O0 -debugger-tune=sce -filetype=obj %s -o -  | llvm-dwarfdump - | FileCheck %s -check-prefixes=SCE-CHECK,BOTH

; SCE-CHECK: .debug_info contents:
; SCE-CHECK: DW_TAG_variable
; SCE-CHECK-NEXT:  DW_AT_location        ({{.*}}
; SCE-CHECK-NEXT:                     [{{.*}}): DW_OP_constu 0x40091eb851eb851f, DW_OP_stack_value)
; SCE-CHECK-NEXT:  DW_AT_name    ("d")

;; Using DW_OP_implicit_value for fragments is not currently supported.
; BOTH: DW_TAG_variable
; BOTH-NEXT:  DW_AT_location        ({{.*}}
; BOTH-NEXT:                     [{{.*}}): DW_OP_constu 0x4047800000000000, DW_OP_stack_value, DW_OP_piece 0x8, DW_OP_constu 0x4052800000000000, DW_OP_stack_value, DW_OP_piece 0x8)
; BOTH-NEXT:  DW_AT_name    ("c")

; ModuleID = 'implicit_value-double.c'
source_filename = "implicit_value-double.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@str = private unnamed_addr constant [6 x i8] c"dummy\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata double 3.140000e+00, metadata !12, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.value(metadata double 4.700000e+01, metadata !17, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg !14
  call void @llvm.dbg.value(metadata double 7.400000e+01, metadata !17, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64)), !dbg !14
  %puts = call i32 @puts(i8* nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @str, i64 0, i64 0)), !dbg !15
  call void @llvm.dbg.value(metadata double undef, metadata !12, metadata !DIExpression()), !dbg !14
  ret i32 0, !dbg !16
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

; Function Attrs: nofree nounwind
declare i32 @puts(i8* nocapture readonly) local_unnamed_addr #2

attributes #0 = { nofree nounwind uwtable }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { nofree nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "implicit_value-double.c", directory: "/home/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 11.0.0"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !17}
!12 = !DILocalVariable(name: "d", scope: !7, file: !1, line: 2, type: !13)
!13 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!14 = !DILocation(line: 0, scope: !7)
!15 = !DILocation(line: 3, column: 2, scope: !7)
!16 = !DILocation(line: 5, column: 2, scope: !7)
!17 = !DILocalVariable(name: "c", scope: !7, file: !1, line: 2, type: !18)
!18 = !DIBasicType(name: "complex", size: 128, encoding: DW_ATE_complex_float)
