; RUN: llc -mtriple=x86_64-unknown-linux-gnu -experimental-debug-variable-locations=false -filetype=obj %s -o %t
; RUN: llvm-dwarfdump --verify %t
; RUN: llvm-dwarfdump --debug-info %t | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -experimental-debug-variable-locations=true -filetype=obj %s -o %t
; RUN: llvm-dwarfdump --verify %t
; RUN: llvm-dwarfdump --debug-info %t | FileCheck %s
;
; Salvaging a truncated value must preserve the distinction between a value
; and an address, including expressions that already have a stack value or
; describe a fragment.
; The shift/mask/extension sequence creates a truncation that SelectionDAG
; removes, requiring the debug values to be salvaged.

; CHECK: DW_TAG_variable
; CHECK: DW_AT_location
; CHECK: DW_OP_convert {{.*}}"DW_ATE_unsigned_16", DW_OP_stack_value{{\)?$}}
; CHECK: DW_AT_name ("value")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location
; CHECK: DW_OP_convert {{.*}}"DW_ATE_unsigned_16", DW_OP_stack_value{{\)?$}}
; CHECK: DW_AT_name ("stack_value")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location
; CHECK: DW_OP_convert {{.*}}"DW_ATE_unsigned_16", DW_OP_stack_value, DW_OP_piece 0x2{{\)?$}}
; CHECK: DW_AT_name ("fragment")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location
; CHECK: DW_OP_convert {{.*}}"DW_ATE_unsigned_16", DW_OP_stack_value{{\)?$}}
; CHECK: DW_AT_name ("arglist")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location
; CHECK: DW_OP_convert {{.*}}"DW_ATE_unsigned_16"{{\)?$}}
; CHECK: DW_AT_name ("indirect")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location
; CHECK: DW_OP_convert {{.*}}"DW_ATE_unsigned_64", DW_OP_breg
; CHECK-SAME: DW_OP_convert {{.*}}"DW_ATE_unsigned_32", DW_OP_convert {{.*}}"DW_ATE_unsigned_16", DW_OP_convert {{.*}}"DW_ATE_unsigned_64", DW_OP_plus{{\)?$}}
; CHECK: DW_AT_name ("memory")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location
; CHECK: DW_OP_convert {{.*}}"DW_ATE_unsigned_64", DW_OP_breg
; CHECK-SAME: DW_OP_convert {{.*}}"DW_ATE_unsigned_32", DW_OP_convert {{.*}}"DW_ATE_unsigned_16", DW_OP_convert {{.*}}"DW_ATE_unsigned_64", DW_OP_plus, DW_OP_piece 0x10{{\)?$}}
; CHECK: DW_AT_name ("memory_fragment")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location
; CHECK: DW_OP_convert {{.*}}"DW_ATE_unsigned_64", DW_OP_breg
; CHECK-SAME: DW_OP_convert {{.*}}"DW_ATE_unsigned_32", DW_OP_convert {{.*}}"DW_ATE_unsigned_16", DW_OP_convert {{.*}}"DW_ATE_unsigned_64", DW_OP_plus, DW_OP_deref, DW_OP_plus_uconst 0x1, DW_OP_stack_value{{\)?$}}
; CHECK: DW_AT_name ("loaded_value")

declare void @observe(ptr, i64)

define i64 @salvage_truncate(ptr %base, i16 %code) !dbg !4 {
entry:
  %shift = lshr i16 %code, 7, !dbg !8
  %value = and i16 %shift, 3, !dbg !8
  #dbg_value(i16 %value, !7, !DIExpression(), !8)
  #dbg_value(i16 %value, !9, !DIExpression(DW_OP_stack_value), !8)
  #dbg_value(i16 %value, !10, !DIExpression(DW_OP_LLVM_fragment, 0, 16), !8)
  #dbg_value(!DIArgList(i16 %value), !14, !DIExpression(DW_OP_LLVM_arg, 0), !8)
  #dbg_declare(i16 %value, !11, !DIExpression(), !8)
  #dbg_value(!DIArgList(ptr %base, i16 %value), !15, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_convert, 64, DW_ATE_unsigned, DW_OP_LLVM_arg, 1, DW_OP_LLVM_convert, 64, DW_ATE_unsigned, DW_OP_plus, DW_OP_deref), !8)
  #dbg_value(!DIArgList(ptr %base, i16 %value), !16, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_convert, 64, DW_ATE_unsigned, DW_OP_LLVM_arg, 1, DW_OP_LLVM_convert, 64, DW_ATE_unsigned, DW_OP_plus, DW_OP_deref, DW_OP_LLVM_fragment, 0, 128), !8)
  #dbg_value(!DIArgList(ptr %base, i16 %value), !17, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_convert, 64, DW_ATE_unsigned, DW_OP_LLVM_arg, 1, DW_OP_LLVM_convert, 64, DW_ATE_unsigned, DW_OP_plus, DW_OP_deref, DW_OP_plus_uconst, 1), !8)
  %wide = zext i16 %value to i64, !dbg !8
  call void @observe(ptr %base, i64 %wide), !dbg !8
  ret i64 %wide, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!1 = !DIFile(filename: "truncate.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = distinct !DISubprogram(name: "salvage_truncate", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !{!13, !18, !12})
!6 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!7 = !DILocalVariable(name: "value", scope: !4, file: !1, line: 2, type: !12)
!8 = !DILocation(line: 2, column: 1, scope: !4)
!9 = !DILocalVariable(name: "stack_value", scope: !4, file: !1, line: 2, type: !12)
!10 = !DILocalVariable(name: "fragment", scope: !4, file: !1, line: 2, type: !6)
!11 = !DILocalVariable(name: "indirect", scope: !4, file: !1, line: 2, type: !12)
!12 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!13 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!14 = !DILocalVariable(name: "arglist", scope: !4, file: !1, line: 2, type: !12)
!15 = !DILocalVariable(name: "memory", scope: !4, file: !1, line: 2, type: !19)
!16 = !DILocalVariable(name: "memory_fragment", scope: !4, file: !1, line: 2, type: !20)
!17 = !DILocalVariable(name: "loaded_value", scope: !4, file: !1, line: 2, type: !13)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Pair", file: !1, line: 1, size: 128)
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Triple", file: !1, line: 1, size: 192)
