; RUN: llc -global-isel -mtriple=amdgpu9.08-amd-amdhsa -filetype=obj < %s | llvm-dwarfdump --debug-info - | FileCheck %s

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_location (0x{{[0-9a-f]+}}:
; CHECK-NEXT:     [{{.*}}): DW_OP_bregx VGPR0+0, DW_OP_constu 0xffffffff, DW_OP_and, DW_OP_bregx VGPR1+0, DW_OP_constu 0x20, DW_OP_shl, DW_OP_or)
; CHECK-NEXT:   DW_AT_name ("declare")

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_location (0x{{[0-9a-f]+}}:
; CHECK-NEXT:     [{{.*}}): DW_OP_bregx VGPR0+0, DW_OP_constu 0xffffffff, DW_OP_and, DW_OP_bregx VGPR1+0, DW_OP_constu 0x20, DW_OP_shl, DW_OP_or, DW_OP_plus_uconst 0x4, DW_OP_stack_value)
; CHECK-NEXT:   DW_AT_name ("complex")

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_name ("too_wide")

; CHECK:      DW_TAG_formal_parameter
; CHECK-NEXT:   DW_AT_location (DW_OP_bregx VGPR0+0, DW_OP_constu 0xffffffff, DW_OP_and, DW_OP_bregx VGPR1+0, DW_OP_constu 0x20, DW_OP_shl, DW_OP_or)
; CHECK-NEXT:   DW_AT_name ("declare_arg")

define ptr @declare(ptr addrspace(1) %pp) !dbg !10 {
  %p = load ptr, ptr addrspace(1) %pp, !dbg !12
    #dbg_declare(ptr %p, !11, !DIExpression(), !12)
  ret ptr %p, !dbg !12
}

define i64 @complex(ptr addrspace(1) %p) !dbg !20 {
  %v = load i64, ptr addrspace(1) %p, !dbg !22
    #dbg_value(i64 %v, !21, !DIExpression(DW_OP_plus_uconst, 4, DW_OP_stack_value), !22)
  ret i64 %v, !dbg !22
}

define <4 x i32> @too_wide(ptr addrspace(1) %p) !dbg !30 {
  %v = load <4 x i32>, ptr addrspace(1) %p, !dbg !32
    #dbg_value(<4 x i32> %v, !31, !DIExpression(DW_OP_plus_uconst, 4, DW_OP_stack_value), !32)
  ret <4 x i32> %v, !dbg !32
}

define void @declare_arg(ptr %p) !dbg !40 {
    #dbg_declare(ptr %p, !41, !DIExpression(), !42)
  store i32 0, ptr %p, !dbg !42
  ret void, !dbg !42
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DISubroutineType(types: !{})
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!6 = !DIBasicType(name: "int128", size: 128, encoding: DW_ATE_signed)
!10 = distinct !DISubprogram(name: "declare", type: !3, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DILocalVariable(name: "declare", scope: !10, type: !4)
!12 = !DILocation(line: 1, scope: !10)
!20 = distinct !DISubprogram(name: "complex", type: !3, spFlags: DISPFlagDefinition, unit: !0)
!21 = !DILocalVariable(name: "complex", scope: !20, type: !5)
!22 = !DILocation(line: 1, scope: !20)
!30 = distinct !DISubprogram(name: "too_wide", type: !3, spFlags: DISPFlagDefinition, unit: !0)
!31 = !DILocalVariable(name: "too_wide", scope: !30, type: !6)
!32 = !DILocation(line: 1, scope: !30)
!40 = distinct !DISubprogram(name: "declare_arg", type: !3, spFlags: DISPFlagDefinition, unit: !0)
!41 = !DILocalVariable(name: "declare_arg", arg: 1, scope: !40, type: !4)
!42 = !DILocation(line: 1, scope: !40)
