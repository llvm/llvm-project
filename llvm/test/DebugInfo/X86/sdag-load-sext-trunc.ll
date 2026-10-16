; This test checks that after SelectionDAG runs, it preserves the debug info that is lost due to the DAGCombiner combining a load and a sext instruction, where the #dbg_value is pointing to the result of the load.
; However, in this case, the load has multiple uses.

; RUN: llc %s -mtriple=x86_64-unkown-linux -start-before=x86-isel -stop-after=x86-isel -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -O2 %s -start-before=x86-isel -mtriple=x86_64-unkown-linux --filetype=obj -o %t.o 
; RUN: llvm-dwarfdump %t.o --name Idx | FileCheck %s --check-prefix=DUMP
; RUN: llvm-dwarfdump %t.o --name Idx2 | FileCheck %s --check-prefix=DUMP2

; MIR: ![[IDX:[0-9]+]] = !DILocalVariable(name: "Idx"
; MIR: ![[IDX2:[0-9]+]] = !DILocalVariable(name: "Idx2"
; MIR: name: _Z8useValuei
; MIR: name: main
; MIR: debugValueSubstitutions
; MIR-NEXT: - { srcinst: [[INSTR_NUM2:[0-9]+]], srcop: 0, dstinst: [[INSTR_NUM:[0-9]+]], dstop: 0, subreg: 6 }
; MIR-LABEL: bb.0 (%ir-block.0)
; MIR: %{{[0-9a-f]+}}{{.*}} = MOVSX64rm32 ${{.*}}, 1, $noreg, @GlobArr, $noreg, debug-instr-number [[INSTR_NUM]]
; MIR-NEXT: {{.*}} = COPY %0.sub_32bit
; MIR-NEXT DBG_INSTR_REF ![[IDX]], !DIExpression(DW_OP_LLVM_arg, 0), dbg-instr-ref([[INSTR_NUM2]], 0)
; MIR-NEXT DBG_INSTR_REF ![[IDX2]], !DIExpression(DW_OP_LLVM_arg, 0), dbg-instr-ref([[INSTR_NUM]], 0)

; DUMP: DW_AT_location	(indexed ({{[0-9a-f]+}}x{{[0-9a-f]+}}) loclist = 0x{{[0-9a-f]+}}: 
; DUMP-NEXT: [0x{{[0-9a-f]+}}, 0x{{[0-9a-f]+}}):  DW_OP_reg3 RBX)

; DUMP2: DW_AT_location	(indexed ({{[0-9a-f]+}}x{{[0-9a-f]+}}) loclist = 0x{{[0-9a-f]+}}: 
; DUMP2-NEXT: [0x{{[0-9a-f]+}}, 0x{{[0-9a-f]+}}):  DW_OP_reg3 RBX)



  @GlobArr = dso_local local_unnamed_addr global [5 x i32] [i32 1, i32 1, i32 2, i32 3, i32 5], align 16, !dbg !0
  @__const.main.Data = private unnamed_addr constant [7 x i32] [i32 10, i32 20, i32 30, i32 40, i32 50, i32 60, i32 70], align 16
  define dso_local void @_Z8useValuei(i32 noundef %0) local_unnamed_addr #0 !dbg !22 {
    ret void, !dbg !28
  }
  define dso_local noundef i32 @main() local_unnamed_addr #1 !dbg !29 {
    %1 = load i32, ptr @GlobArr
      #dbg_value(i32 %1, !43, !DIExpression(), !52)
    %2 = sext i32 %1 to i64
      #dbg_value(i64 %2, !57, !DIExpression(), !52)
    tail call void @_Z8useValuei(i32 noundef %1), !dbg !56
    %3 = getelementptr inbounds i32, ptr @__const.main.Data, i64 %2
    %4 = load i32, ptr %3
    tail call void @_Z8useValuei(i32 noundef %4), !dbg !56
    ret i32 0
  }
    !llvm.dbg.cu = !{!2}  
  !llvm.module.flags = !{!10, !11, !16}
  !0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
  !1 = distinct !DIGlobalVariable(type: !6, isDefinition: true)
  !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, emissionKind: FullDebug, nameTableKind: None)
  !3 = !DIFile(filename: "/tmp/test.cpp", directory: "/Users/srastogi/Development/llvm-project/build_ninja", checksumkind: CSK_MD5, checksum: "0fe735937e606b4db3e3b2e9253eff90")
  !6 = !DICompositeType(tag: DW_TAG_array_type, elements: !8)
  !7 = !DIBasicType()
  !8 = !{}
  !10 = !{i32 7, !"Dwarf Version", i32 5}
  !11 = !{i32 2, !"Debug Info Version", i32 3}
  !16 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
  !22 = distinct !DISubprogram(type: !23, unit: !2, keyInstructions: true)
  !23 = !DISubroutineType(types: !24)
  !24 = !{}
  !28 = !DILocation(scope: !22, atomRank: 1)
  !29 = distinct !DISubprogram(type: !30, unit: !2, keyInstructions: true)
  !30 = !DISubroutineType(types: !31)
  !31 = !{}
  !38 = distinct !DILexicalBlock(scope: !29, line: 5, column: 3)
  !43 = !DILocalVariable(name: "Idx", scope: !44, type: !7)
  !44 = distinct !DILexicalBlock(scope: !38, line: 5, column: 3)
  !46 = distinct !DILexicalBlock(scope: !44, line: 5, column: 27)
  !52 = !DILocation(scope: !44)
  !56 = !DILocation(scope: !46)
  !57 = !DILocalVariable(name: "Idx2", scope: !44, type: !7)
