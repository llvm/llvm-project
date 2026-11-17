; RUN: llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90 -O0 %s -o - | FileCheck %s
;
; CHECK: .b8 35
; CHECK-NEXT: .b8 4
; CHECK-NEXT: .b8 54
; CHECK-NEXT: .b8 233
; CHECK-NEXT: .b8 2
; CHECK-NEXT: .b8 147
; CHECK-NEXT: .b8 4
; CHECK-NEXT: .b8 109 {{.*}}DW_AT_name
;
; A constant address starts as an implicit value. The extension must turn it
; into a memory location, suppressing an otherwise automatic DW_OP_stack_value.
; CHECK: .b8 60
; CHECK-NEXT: .b8 54
; CHECK-NEXT: .b8 233
; CHECK-NEXT: .b8 2
; CHECK-NEXT: .b8 147
; CHECK-NEXT: .b8 4
; CHECK-NEXT: .b8 110 {{.*}}DW_AT_name
;
; Forming an explicit local-memory location must retain the LLVM user opcode
; and ULEB sub-opcode, followed by a piece rather than DW_OP_stack_value.

define void @local_address() !dbg !7 {
  call void @llvm.dbg.value(metadata i64 12, metadata !11, metadata !DIExpression(DW_OP_constu, 6, DW_OP_LLVM_user, DW_OP_LLVM_form_aspace_address, DW_OP_LLVM_fragment, 0, 32)), !dbg !16
  %o = alloca i64, align 8
  call void @llvm.dbg.declare(metadata ptr %o, metadata !10, metadata !DIExpression(DW_OP_plus_uconst, 4, DW_OP_constu, 6, DW_OP_LLVM_user, DW_OP_LLVM_form_aspace_address, DW_OP_LLVM_fragment, 0, 32)), !dbg !16
  call void @escape(ptr %o), !dbg !16
  ret void, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @escape(ptr)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "local.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "local_address", scope: !1, file: !1, line: 1, type: !8, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DISubroutineType(types: !2)
!10 = !DILocalVariable(name: "member", scope: !7, file: !1, line: 2, type: !14)
!11 = !DILocalVariable(name: "numeric_address", scope: !7, file: !1, line: 2, type: !14)
!14 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!16 = !DILocation(line: 2, column: 1, scope: !7)
