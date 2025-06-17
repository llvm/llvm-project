; RUN: llc -O0 -mcpu=gfx1030 -mtriple=amdgcn-amd-amdhsa -filetype=obj -o - < %s | llvm-dwarfdump --debug-info - | FileCheck %s

; CHECK-LABEL: DW_AT_name ("test_loc_single")
define void @test_loc_single(ptr addrspace(3) %ptr) #0 !dbg !9 {
  ; Verify that the right address class attribute is attached to the variable's
  ; type for a single location:
  ; CHECK: 0x{{[0-9a-f]+}}: DW_TAG_variable
  ; CHECK-NEXT: DW_AT_location (DW_OP_regx {{.*}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset)
  ; CHECK-NEXT: DW_AT_name ("loc_single_ptr")
  ; CHECK-NEXT: DW_AT_decl_file
  ; CHECK-NEXT: DW_AT_decl_line
  ; CHECK-NEXT: DW_AT_type ([[PTR_AS_3:0x[0-9a-f]+]] "int *")

    #dbg_value(ptr addrspace(3) %ptr, !13, !DIExpression(DIOpArg(0, ptr addrspace(3)), DIOpConvert(ptr)), !16)
  ret void, !dbg !17
}

; CHECK-LABEL: DW_AT_name ("test_loc_multi")
define void @test_loc_multi(ptr addrspace(3) %loc_ptr) #0 !dbg !18 {
  ; Verify that no attribute is attached to the variable type if the loclist
  ; contains entries with different address spaces:
  ; CHECK: 0x{{[0-9a-f]+}}: DW_TAG_variable
  ; CHECK-NEXT:   DW_AT_location (indexed ({{0x[0-9a-f]+}}) loclist =
  ; CHECK-NEXT:      [{{0x[0-9a-f]+}}, {{0x[0-9a-f]+}}):{{.*}} DW_OP_LLVM_user DW_OP_LLVM_undefined
  ; CHECK-NEXT:      [{{0x[0-9a-f]+}}, {{0x[0-9a-f]+}}): DW_OP_lit0, DW_OP_stack_value)
  ; CHECK-NEXT:   DW_AT_name ("ptr_as3_as2")
  ; CHECK-NEXT:   DW_AT_decl_file
  ; CHECK-NEXT:   DW_AT_decl_line
  ; CHECK-NEXT:   DW_AT_type ([[PTR_AS_NONE:0x[0-9a-f]+]] "int *")

  ; Verify that an attribute is attached to the variable type if the loclist
  ; contains entries with the same address spaces:
  ; CHECK: 0x{{[0-9a-f]+}}: DW_TAG_variable
  ; CHECK-NEXT:   DW_AT_location (indexed ({{0x[0-9a-f]+}}) loclist =
  ; CHECK-NEXT:      [{{0x[0-9a-f]+}}, {{0x[0-9a-f]+}}): DW_OP_regx
  ; CHECK-NEXT:      [{{0x[0-9a-f]+}}, {{0x[0-9a-f]+}}): DW_OP_lit0, DW_OP_stack_value)
  ; CHECK-NEXT:   DW_AT_name ("ptr_all_as3")
  ; CHECK-NEXT:   DW_AT_decl_file
  ; CHECK-NEXT:   DW_AT_decl_line
  ; CHECK-NEXT:   DW_AT_type ([[PTR_AS_3]] "int *")

    #dbg_value(ptr addrspace(3) %loc_ptr, !21, !DIExpression(DIOpArg(0, ptr addrspace(3)), DIOpConvert(ptr)), !22)
    #dbg_value(ptr addrspace(3) %loc_ptr, !20, !DIExpression(DIOpArg(0, ptr addrspace(3)), DIOpConvert(ptr)), !22)
  tail call void asm sideeffect "s_nop 1", ""(), !dbg !22
    #dbg_value(ptr null, !21, !DIExpression(DIOpArg(0, ptr)), !23)
    #dbg_value(ptr addrspace(3) null, !20, !DIExpression(DIOpArg(0, ptr addrspace(3)), DIOpConvert(ptr)), !23)
  ret void, !dbg !23
}

; CHECK-LABEL: DW_AT_name ("test_loc_mmi")
define void @test_loc_mmi() #0 !dbg !24 {
  ; CHECK: 0x{{[0-9a-f]+}}: DW_TAG_variable
  ; CHECK-NEXT:   DW_AT_location (indexed ({{0x[0-9a-f]+}}) loclist =
  ; CHECK-NEXT:      [{{0x[0-9a-f]+}}, {{0x[0-9a-f]+}}): DW_OP_regx SGPR{{.*}}, DW_OP_deref_size 0x4, DW_OP_lit5, DW_OP_shr, DW_OP_lit0, DW_OP_plus, DW_OP_stack_value)
  ; CHECK-NEXT:   DW_AT_name ("ptr_as5")
  ; CHECK-NEXT:   DW_AT_decl_file
  ; CHECK-NEXT:   DW_AT_decl_line
  ; CHECK-NEXT:   DW_AT_type ([[PTR_AS_5:0x[0-9a-f]+]] "int *")

  %ptr = alloca i32, align 4, addrspace(5), !dbg !27
    #dbg_value(ptr addrspace(5) %ptr, !26, !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpConvert(ptr)), !27)
  ret void, !dbg !28
}

; CHECK-LABEL: DW_AT_name ("test_divergent")
define void @test_divergent(ptr addrspace(5) %p5, ptr addrspace(3) %p3) #0 !dbg !29 {
  ; CHECK: 0x{{[0-9a-f]+}}: DW_TAG_variable
  ; CHECK-NEXT:   DW_AT_location (DW_OP_regx {{.*}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset)
  ; CHECK-NEXT:   DW_AT_name ("ptr_div_as5")
  ; CHECK-NEXT:   DW_AT_decl_file
  ; CHECK-NEXT:   DW_AT_decl_line
  ; CHECK-NEXT:   DW_AT_type ([[PTR_AS_5]] "int *")
    #dbg_value(ptr addrspace(5) %p5, !31, !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpConvert(ptr)), !30)

  ; CHECK: 0x{{[0-9a-f]+}}: DW_TAG_variable
  ; CHECK-NEXT:   DW_AT_location (DW_OP_regx {{.*}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset)
  ; CHECK-NEXT:   DW_AT_name ("ptr_div_as3")
  ; CHECK-NEXT:   DW_AT_decl_file
  ; CHECK-NEXT:   DW_AT_decl_line
  ; CHECK-NEXT:   DW_AT_type ([[PTR_AS_3]] "int *")
    #dbg_value(ptr addrspace(3) %p3, !32, !DIExpression(DIOpArg(0, ptr addrspace(3)), DIOpConvert(ptr), DIOpReinterpret(i64), DIOpReinterpret(ptr)), !30)

  ; CHECK: 0x{{[0-9a-f]+}}: DW_TAG_variable
  ; CHECK-NEXT:   DW_AT_location ({{.*}} DW_OP_LLVM_user DW_OP_LLVM_undefined)
  ; CHECK-NEXT:   DW_AT_name ("ptr_div_invalid")
    #dbg_value(ptr addrspace(5) %p5, !33, !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpConvert(ptr), DIOpReinterpret(i64), DIOpConstant(i64 42), DIOpAdd(), DIOpReinterpret(ptr)), !30)

  ret void, !dbg !30
}

; CHECK-LABEL: DW_AT_name ("test_noop_convert")
define void @test_noop_convert(ptr addrspace(1) %p1) #0 !dbg !34 {
 ; Verify that a noop address space conversion doesn't produce a divergent
 ; address space.
 ; CHECK: 0x{{[0-9a-f]+}}: DW_TAG_variable
 ; CHECK-NEXT: DW_AT_location
 ; CHECK-NEXT: DW_AT_name ("not_divergent")
 ; CHECK-NEXT: DW_AT_decl_file
 ; CHECK-NEXT: DW_AT_decl_line
 ; CHECK-NEXT: DW_AT_type ([[PTR_AS_NONE]] "int *")
    #dbg_value(ptr addrspace(1) %p1, !36, !DIExpression(DIOpArg(0, ptr addrspace(1)), DIOpConvert(ptr addrspace(1)), DIOpReinterpret(ptr)), !37)
  ret void, !dbg !37
}

attributes #0 = { "frame-pointer"="all" }

; CHECK: [[PTR_AS_3]]: DW_TAG_pointer_type
; CHECK-NEXT: DW_AT_type
; CHECK-NEXT: DW_AT_address_class (0x00000003)
; CHECK-NEXT: DW_AT_LLVM_address_space (0x00000003 "DW_ASPACE_LLVM_AMDGPU_local")

; CHECK: [[PTR_AS_NONE]]: DW_TAG_pointer_type
; CHECK-NEXT: DW_AT_type
; CHECK-EMPTY:

; CHECK: [[PTR_AS_5]]: DW_TAG_pointer_type
; CHECK-NEXT: DW_AT_type
; CHECK-NEXT: DW_AT_address_class (0x00000005)
; CHECK-NEXT: DW_AT_LLVM_address_space (0x00000005 "DW_ASPACE_LLVM_AMDGPU_private_lane")

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 19.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.cpp", directory: "/")
!2 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 8, !"PIC Level", i32 2}
!7 = !{i32 7, !"frame-pointer", i32 2}
!8 = !{!"clang version 19.0.0"}
!9 = distinct !DISubprogram(name: "test_loc_single", linkageName: "test_loc_single", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!10 = !DISubroutineType(types: !11)
!11 = !{}
!12 = !{!13}
!13 = !DILocalVariable(name: "loc_single_ptr", scope: !9, file: !1, line: 1, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !DILocation(line: 1, column: 14, scope: !9)
!17 = !DILocation(line: 2, column: 1, scope: !9)
!18 = distinct !DISubprogram(name: "test_loc_multi", linkageName: "test_loc_multi", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
!19 = !{!20, !21}
!20 = !DILocalVariable(name: "ptr_all_as3", scope: !18, file: !1, line: 1, type: !14)
!21 = !DILocalVariable(name: "ptr_as3_as2", scope: !18, file: !1, line: 1, type: !14)
!22 = !DILocation(line: 1, column: 1, scope: !18)
!23 = !DILocation(line: 2, column: 1, scope: !18)
!24 = distinct !DISubprogram(name: "test_loc_mmi", linkageName: "test_loc_mmi", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !25)
!25 = !{!26}
!26 = !DILocalVariable(name: "ptr_as5", scope: !24, file: !1, line: 1, type: !14)
!27 = !DILocation(line: 1, column: 1, scope: !24)
!28 = !DILocation(line: 2, column: 1, scope: !24)
!29 = distinct !DISubprogram(name: "test_divergent", linkageName: "test_divergent", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
!30 = !DILocation(line: 1, column: 1, scope: !29)
!31 = !DILocalVariable(name: "ptr_div_as5", scope: !29, file: !1, line: 1, type: !14)
!32 = !DILocalVariable(name: "ptr_div_as3", scope: !29, file: !1, line: 1, type: !14)
!33 = !DILocalVariable(name: "ptr_div_invalid", scope: !29, file: !1, line: 1, type: !14)
!34 = distinct !DISubprogram(name: "test_noop_convert", linkageName: "test_noop_convert", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !35)
!35 = !{!36}
!36 = !DILocalVariable(name: "not_divergent", scope: !34, file: !1, line: 1, type: !14)
!37 = !DILocation(line: 1, column: 1, scope: !34)
