; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_untyped_pointers,+SPV_KHR_relaxed_extended_instruction --spirv-debug-scope-forward-refs=false %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-OFF --implicit-check-not=OpExtInstWithForwardRefsKHR --implicit-check-not=DebugDeclare
; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-ON
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_untyped_pointers,+SPV_KHR_relaxed_extended_instruction --spirv-debug-lexical-block-discriminator-scope --spirv-debug-scope-forward-refs --spirv-debug-declare-untyped-variable %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-ON

; The AMD vendor always enables -spirv-debug-lexical-block-discriminator-scope,
; -spirv-debug-scope-forward-refs and -spirv-debug-declare-untyped-variable by
; default. 
;

; CHECK-ON:     OpExtension "SPV_KHR_relaxed_extended_instruction"
; CHECK-DAG:    [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG:    [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG:    [[STR_S:%[0-9]+]] = OpString "S"
; CHECK-DAG:    [[STR_X:%[0-9]+]] = OpString "x"
; CHECK-ON-DAG: [[STR_NEXT:%[0-9]+]] = OpString "next"

; CHECK: [[LB:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLexicalBlock
; CHECK: [[DISC:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLexicalBlockDiscriminator {{%[0-9]+}} {{%[0-9]+}} [[LB]]

; CHECK:     [[MEM_X:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeMember [[STR_X]]
; CHECK-OFF: [[COMP:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeComposite [[STR_S]] {{.*}} [[MEM_X]]{{$}}
; CHECK-OFF: OpExtInst [[VOID]] [[EXT]] DebugTypePointer [[COMP]]
; CHECK-ON:  [[PTR:%[0-9]+]] = OpExtInstWithForwardRefsKHR [[VOID]] [[EXT]] DebugTypePointer [[COMP:%[0-9]+]]
; CHECK-ON:  [[MEM_NEXT:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeMember [[STR_NEXT]] [[PTR]]
; CHECK-ON:  [[COMP]] = OpExtInst [[VOID]] [[EXT]] DebugTypeComposite [[STR_S]] {{.*}} [[MEM_X]] [[MEM_NEXT]]{{$}}

; CHECK:    [[LV:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[STR_X]]
; CHECK:    [[VAR:%[0-9]+]] = OpUntypedVariableKHR {{%[0-9]+}} Function
; CHECK-ON: OpExtInst [[VOID]] [[EXT]] DebugDeclare [[LV]] [[VAR]]

; CHECK-OFF-NOT: DebugScope [[DISC]]
; CHECK-OFF:     OpExtInst [[VOID]] [[EXT]] DebugScope [[LB]]
; CHECK-OFF-NOT: DebugScope [[DISC]]
; CHECK-ON:      OpExtInst [[VOID]] [[EXT]] DebugScope [[DISC]]
; CHECK-ON-NOT:  DebugScope [[LB]]

define spir_func i32 @foo(i32 %N) !dbg !6 {
entry:
  %x = alloca i32, align 4
    #dbg_declare(ptr %x, !20, !DIExpression(), !14)
  store i32 %N, ptr %x, align 4, !dbg !14
  %cmp = icmp sgt i32 %N, 0, !dbg !11
  br i1 %cmp, label %t, label %f, !dbg !11

t:
  ret i32 1, !dbg !14

f:
  ret i32 0, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !15)
!1 = !DIFile(filename: "foo.c", directory: "/tmp")
!2 = !{null}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !7, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DISubroutineType(types: !2)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 4, scope: !12)
!12 = !DILexicalBlockFile(scope: !13, file: !1, discriminator: 1)
!13 = distinct !DILexicalBlock(scope: !6, file: !1, line: 4, column: 3)
!14 = !DILocation(line: 5, scope: !6)
!15 = !{!16}
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 1, size: 128, elements: !17)
!17 = !{!18, !19}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "x", file: !1, line: 1, baseType: !9, size: 32)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "next", file: !1, line: 1, baseType: !21, size: 64, offset: 64)
!20 = !DILocalVariable(name: "x", scope: !6, file: !1, line: 3, type: !9)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64, dwarfAddressSpace: 4)
