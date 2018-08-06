; RUN: llc < %s | FileCheck %s
; RUN: llc < %s -filetype=obj | llvm-dwarfdump -v - --debug-info | FileCheck %s --check-prefix=DWARF

; This should use the frame index side table for allocas, not DBG_VALUE
; instructions. For SDAG ISel, this test would see an SDNode materializing the
; argument to escape_foo and we'd get DBG_VALUE MachineInstr.

; CHECK-LABEL: use_dbg_declare:
; CHECK-NOT: #DEBUG_VALUE

; DWARF: DW_TAG_variable
; DWARF-NEXT:              DW_AT_location [DW_FORM_exprloc]      (DW_OP_fbreg +0)
; DWARF-NEXT:              DW_AT_name [DW_FORM_strp]     ( {{.*}} = "o")


; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux"

%struct.Foo = type { i32 }

; Function Attrs: noinline nounwind uwtable
define void @use_dbg_declare() #0 !dbg !7 {
entry:
  %o = alloca %struct.Foo, align 4
  call void @llvm.dbg.declare(metadata %struct.Foo* %o, metadata !10, metadata !15), !dbg !16
  call void @escape_foo(%struct.Foo* %o), !dbg !17
  ret void, !dbg !18
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @escape_foo(%struct.Foo*)

attributes #0 = { noinline nounwind uwtable }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "C:\5Csrc\5Cllvm-project\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 6.0.0 "}
!7 = distinct !DISubprogram(name: "use_dbg_declare", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "o", scope: !7, file: !1, line: 4, type: !11)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !1, line: 1, size: 32, elements: !12)
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !11, file: !1, line: 1, baseType: !14, size: 32)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DIExpression()
!16 = !DILocation(line: 4, column: 14, scope: !7)
!17 = !DILocation(line: 5, column: 3, scope: !7)
!18 = !DILocation(line: 6, column: 1, scope: !7)
