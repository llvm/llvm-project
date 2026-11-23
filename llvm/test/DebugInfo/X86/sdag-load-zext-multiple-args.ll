; This test checks that after SelectionDAG runs, it preserves the debug info that is lost due to the DAGCombiner combining a load and a zext instruction, where the #dbg_value is pointing to the result of the load. However, this test also ensures that the DIExpression, which has multiple DW_OP_LLVM_arg's is handled correctly when the debug info is preserved in selectioDAG.
; RUN: llc %s -mtriple=x86_64-unkown-linux-gnu -start-before=x86-isel -stop-after=x86-isel -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -O2 %s -start-before=x86-isel -mtriple=x86_64-unkown-linux --filetype=obj -o %t.o 

; MIR: ![[V:[0-9]+]] = !DILocalVariable(name: "v"
; MIR-LABEL: bb.0
; MIR: %{{[0-9a-f]+}}{{.*}} = MOVZX32rm8 {{.*}}, 1, $noreg, 0, $noreg, debug-instr-number [[INSTR_NUM:[0-9]+]]
; MIR-NEXT: DBG_INSTR_REF ![[V]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_LLVM_convert, 8, DW_ATE_unsigned, DW_OP_constu, 45, DW_OP_eq, DW_OP_LLVM_arg, 1, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_LLVM_convert, 8, DW_ATE_unsigned, DW_OP_constu, 114, DW_OP_eq, DW_OP_or, DW_OP_stack_value), dbg-instr-ref([[INSTR_NUM]], 0), dbg-instr-ref([[INSTR_NUM]], 0)

  @.str = private unnamed_addr constant [105 x i8] c"/Users/srastogi/Development/llvm-project-2/compiler-rt/lib/sanitizer_common/sanitizer_procmaps_linux.cpp\00"
  @.str.1 = private unnamed_addr constant [45 x i8] c"((IsOneOf(*data_.current, '-', 'r'))) != (0)\00"
  define hidden noundef zeroext i1 @_ZN11__sanitizer19MemoryMappingLayout4NextEPNS_19MemoryMappedSegmentE(ptr noundef nonnull readonly align 8 captures(none) dereferenceable(32) %this, ptr noundef readonly captures(none) %segment) unnamed_addr #0 align 2 !dbg !64 {
    %current = getelementptr inbounds nuw i8, ptr %this, i64 24
    %3 = load ptr, ptr %current, !dbg !95
    %4 = load i8, ptr %3
      #dbg_value(!DIArgList(i8 %4, i8 %4), !71, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_constu, 45, DW_OP_eq, DW_OP_LLVM_arg, 1, DW_OP_constu, 114, DW_OP_eq, DW_OP_or, DW_OP_stack_value), !98)
    %5 = zext i8 %4 to i32
    switch i32 %5, label %if.then [
      i32 114, label %if.end
      i32 45, label %if.end
    ]
  if.then:                                          ; preds = %entry
    tail call void @_ZN11__sanitizer11CheckFailedEPKciS1_yy(ptr noundef nonnull @.str, i32 noundef 45, ptr noundef nonnull @.str.1, i64 noundef 0, i64 noundef 0) #5
    unreachable
  if.end:                                           ; preds = %entry, %entry
    ret i1 true
  }
  declare void @_ZN11__sanitizer11CheckFailedEPKciS1_yy(ptr noundef, i32 noundef, ptr noundef, i64 noundef, i64 noundef) local_unnamed_addr #2
  !llvm.dbg.cu = !{!15}
  !llvm.module.flags = !{!55, !58}
  !0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
  !1 = distinct !DIGlobalVariable()
  !2 = !DIFile(filename: "san.pp.cpp", directory: "/Users/srastogi/Development/Delta", checksumkind: CSK_MD5, checksum: "386698ddde6c66899ec76581efaeabe2")
  !15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !16, emissionKind: FullDebug, nameTableKind: None)
  !16 = !DIFile(filename: "/Users/srastogi/Development/Delta/san.pp.cpp", directory: "/Users/srastogi/Development/Delta", checksumkind: CSK_MD5, checksum: "386698ddde6c66899ec76581efaeabe2")
  !19 = !DIDerivedType(tag: DW_TAG_typedef,baseType: !21)
  !21 = !DIBasicType()
  !22 = !DIDerivedType(tag: DW_TAG_typedef, baseType: !23)
  !23 = !DIBasicType()
  !46 = !DISubroutineType(types: !47)
  !47 = !{}
  !55 = !{i32 2, !"Debug Info Version", i32 3}
  !58 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
  !64 = distinct !DISubprogram( type: !46, unit: !15, keyInstructions: true)
  !71 = !DILocalVariable(name: "v", scope: !72, type: !19)
  !72 = distinct !DILexicalBlock(scope: !64, line: 187, column: 6)
  !95 = !DILocation( scope: !72)
  !98 = !DILocation(scope: !72)
