;; Run twice -- once with DBG_VALUEs, once with instruction referencing.
; RUN: llc %s -o %t.s -experimental-debug-variable-locations=false
; RUN: llvm-mc -triple x86_64--linux %t.s -filetype=obj -o %t.o
; RUN: FileCheck -input-file=%t.s %s
; RUN: llvm-dwarfdump %t.o | FileCheck %s --check-prefix=DWARF
; RUN: llc %s -o %t.s -experimental-debug-variable-locations=true
; RUN: llvm-mc -triple x86_64--linux %t.s -filetype=obj -o %t.o
; RUN: FileCheck -input-file=%t.s %s
; RUN: llvm-dwarfdump %t.o | FileCheck %s --check-prefix=DWARF


; Unlike dbg.declare, dbg.addr should be lowered to DBG_VALUE instructions. It
; is control-dependent.

; CHECK-LABEL: use_dbg_addr:
; CHECK: #DEBUG_VALUE: use_dbg_addr:o <- [$rsp+0]
; CHECK-NOT: #DEBUG_VALUE:

; DWARF: DW_TAG_variable
; DWARF-NEXT:              DW_AT_location (DW_OP_fbreg +0)
; DWARF-NEXT:              DW_AT_name ("o")

; Make sure that in the second case, we properly get a validity range in the
; dwarf for the value. This ensures that we can use this technique to invalidate
; variables.

; CHECK-LABEL: test_dbg_addr_and_dbg_val_undef
; CHECK: #DEBUG_VALUE: test_dbg_addr_and_dbg_val_undef:second_o <- [$rsp+0]
; CHECK: #DEBUG_VALUE: test_dbg_addr_and_dbg_val_undef:second_o <- undef
; CHECK-NOT: #DEBUG_VALUE:

; DWARF: DW_TAG_variable
; DWARF-NEXT:              DW_AT_location (0x{{[0-9a-z][0-9a-z]*}}:
; DWARF-NEXT:                 [0x{{[0-9a-z][0-9a-z]*}}, 0x{{[0-9a-z][0-9a-z]*}}): DW_OP_breg7 RSP+0)
; DWARF-NEXT:              DW_AT_name ("second_o")

; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux"

%struct.Foo = type { i32 }

; Function Attrs: noinline nounwind uwtable
define void @use_dbg_addr() #0 !dbg !7 {
entry:
  %o = alloca %struct.Foo, align 8
  call void @llvm.dbg.addr(metadata ptr %o, metadata !10, metadata !15), !dbg !16
  call void @escape_foo(ptr %o), !dbg !17
  ret void, !dbg !18
}

define void @test_dbg_addr_and_dbg_val_undef() #0 !dbg !117 {
entry:
  %o = alloca %struct.Foo, align 8
  call void @llvm.dbg.addr(metadata ptr %o, metadata !1110, metadata !1115), !dbg !1116
  call void @escape_foo(ptr %o), !dbg !1117
  call void @llvm.dbg.value(metadata ptr undef, metadata !1110, metadata !1115), !dbg !1116
  ret void, !dbg !1118
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.addr(metadata, metadata, metadata) #1
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

declare void @escape_foo(ptr)

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
!7 = distinct !DISubprogram(name: "use_dbg_addr", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
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

!117 = distinct !DISubprogram(name: "test_dbg_addr_and_dbg_val_undef", scope: !1, file: !1, line: 3, type: !118, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!118 = !DISubroutineType(types: !119)
!119 = !{null}
!1110 = !DILocalVariable(name: "second_o", scope: !117, file: !1, line: 4, type: !1111)
!1111 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !1, line: 1, size: 32, elements: !1112)
!1112 = !{!1113}
!1113 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !1111, file: !1, line: 1, baseType: !1114, size: 32)
!1114 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!1115 = !DIExpression()
!1116 = !DILocation(line: 4, column: 14, scope: !117)
!1117 = !DILocation(line: 5, column: 3, scope: !117)
!1118 = !DILocation(line: 6, column: 1, scope: !117)
