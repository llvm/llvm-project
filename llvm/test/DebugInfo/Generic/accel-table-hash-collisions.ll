; REQUIRES: object-emission
; RUN: %llc_dwarf -dwarf-accel-tables=Enable -filetype=obj -o - < %s | llvm-dwarfdump -debug-dump=apple_names - | FileCheck %s

; Generated from the following C code using
; clang -S -emit-llvm hash-collision.c
;
; The names of the variables have been chosen so that they produce hash collisions.
; There are 12 names here that are hashed to only 6 hashes (each pair of lines
; hashes to the same value, see the CHECK lines below).
;
; int ForceTopDown;
; int _ZNSt3__116allocator_traitsINS_9allocatorINS_11__tree_nodeINS_12__value_typeIPN4llvm10BasicBlockEPNS4_10RegionNodeEEEPvEEEEE11__constructIS9_JNS_4pairIS6_S8_EEEEEvNS_17integral_constantIbLb1EEERSC_PT_DpOT0_;
; int _ZN5clang23DataRecursiveASTVisitorIN12_GLOBAL__N_124UnusedBackingIvarCheckerEE26TraverseCUDAKernelCallExprEPNS_18CUDAKernelCallExprE; 
; int _ZN4llvm16DenseMapIteratorIPNS_10MDLocationENS_6detail13DenseSetEmptyENS_10MDNodeInfoIS1_EENS3_12DenseSetPairIS2_EELb0EE23AdvancePastEmptyBucketsEv;
; int _ZNK4llvm12LivePhysRegs5printERNS_11raw_ostreamE;
; int _ZN4llvm15ScalarEvolution14getSignedRangeEPKNS_4SCEVE;
; int k1;
; int is;
; int setStmt;
; int _ZN4llvm5TwineC1Ei;
; int _ZNK5clang12OverrideAttr5cloneERNS_10ASTContextE;
; int _ZN4llvm22MachineModuleInfoMachOD2Ev;

; Check that we have the right amount of hashes.
; CHECK: Bucket count = 6
; CHECK: Hashes count = 6

; Check that all the names are present in the output
; CHECK:  Hash = 0x00597841
; CHECK:    Name: {{[0-9a-f]*}} "is"
; CHECK:    Name: {{[0-9a-f]*}} "k1"

; CHECK: Hash = 0xa4b42a1e
; CHECK:    Name: {{[0-9a-f]*}} "_ZN5clang23DataRecursiveASTVisitorIN12_GLOBAL__N_124UnusedBackingIvarCheckerEE26TraverseCUDAKernelCallExprEPNS_18CUDAKernelCallExprE"
; CHECK:    Name: {{[0-9a-f]*}} "_ZN4llvm16DenseMapIteratorIPNS_10MDLocationENS_6detail13DenseSetEmptyENS_10MDNodeInfoIS1_EENS3_12DenseSetPairIS2_EELb0EE23AdvancePastEmptyBucketsEv"

; CHECK: Hash = 0xeee7c0b2
; CHECK:    Name: {{[0-9a-f]*}} "_ZNK4llvm12LivePhysRegs5printERNS_11raw_ostreamE"
; CHECK:    Name: {{[0-9a-f]*}} "_ZN4llvm15ScalarEvolution14getSignedRangeEPKNS_4SCEVE"

; CHECK: Hash = 0xea48ac5f
; CHECK:    Name: {{[0-9a-f]*}} "ForceTopDown"
; CHECK:    Name: {{[0-9a-f]*}} "_ZNSt3__116allocator_traitsINS_9allocatorINS_11__tree_nodeINS_12__value_typeIPN4llvm10BasicBlockEPNS4_10RegionNodeEEEPvEEEEE11__constructIS9_JNS_4pairIS6_S8_EEEEEvNS_17integral_constantIbLb1EEERSC_PT_DpOT0_"

; CHECK:  Hash = 0x6b22f71f
; CHECK:    Name: {{[0-9a-f]*}} "_ZNK5clang12OverrideAttr5cloneERNS_10ASTContextE"
; CHECK:    Name: {{[0-9a-f]*}} "_ZN4llvm22MachineModuleInfoMachOD2Ev"

; CHECK:  Hash = 0x8c248979
; CHECK:    Name: {{[0-9a-f]*}} "setStmt"
; CHECK:    Name: {{[0-9a-f]*}} "_ZN4llvm5TwineC1Ei"



@ForceTopDown = common global i32 0, align 4
@_ZNSt3__116allocator_traitsINS_9allocatorINS_11__tree_nodeINS_12__value_typeIPN4llvm10BasicBlockEPNS4_10RegionNodeEEEPvEEEEE11__constructIS9_JNS_4pairIS6_S8_EEEEEvNS_17integral_constantIbLb1EEERSC_PT_DpOT0_ = common global i32 0, align 4
@_ZN5clang23DataRecursiveASTVisitorIN12_GLOBAL__N_124UnusedBackingIvarCheckerEE26TraverseCUDAKernelCallExprEPNS_18CUDAKernelCallExprE = common global i32 0, align 4
@_ZN4llvm16DenseMapIteratorIPNS_10MDLocationENS_6detail13DenseSetEmptyENS_10MDNodeInfoIS1_EENS3_12DenseSetPairIS2_EELb0EE23AdvancePastEmptyBucketsEv = common global i32 0, align 4
@_ZNK4llvm12LivePhysRegs5printERNS_11raw_ostreamE = common global i32 0, align 4
@_ZN4llvm15ScalarEvolution14getSignedRangeEPKNS_4SCEVE = common global i32 0, align 4
@k1 = common global i32 0, align 4
@is = common global i32 0, align 4
@setStmt = common global i32 0, align 4
@_ZN4llvm5TwineC1Ei = common global i32 0, align 4
@_ZNK5clang12OverrideAttr5cloneERNS_10ASTContextE = common global i32 0, align 4
@_ZN4llvm22MachineModuleInfoMachOD2Ev = common global i32 0, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18, !19}
!llvm.ident = !{!20}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.7.0 (trunk 231548) (llvm/trunk 231547)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, subprograms: !2, globals: !3, imports: !2)
!1 = !DIFile(filename: "hash-collisions.c", directory: "/tmp")
!2 = !{}
!3 = !{!4, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16}
!4 = !DIGlobalVariable(name: "ForceTopDown", scope: !0, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, variable: i32* @ForceTopDown)
!5 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !DIGlobalVariable(name: "_ZNSt3__116allocator_traitsINS_9allocatorINS_11__tree_nodeINS_12__value_typeIPN4llvm10BasicBlockEPNS4_10RegionNodeEEEPvEEEEE11__constructIS9_JNS_4pairIS6_S8_EEEEEvNS_17integral_constantIbLb1EEERSC_PT_DpOT0_", scope: !0, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, variable: i32* @_ZNSt3__116allocator_traitsINS_9allocatorINS_11__tree_nodeINS_12__value_typeIPN4llvm10BasicBlockEPNS4_10RegionNodeEEEPvEEEEE11__constructIS9_JNS_4pairIS6_S8_EEEEEvNS_17integral_constantIbLb1EEERSC_PT_DpOT0_)
!7 = !DIGlobalVariable(name: "_ZN5clang23DataRecursiveASTVisitorIN12_GLOBAL__N_124UnusedBackingIvarCheckerEE26TraverseCUDAKernelCallExprEPNS_18CUDAKernelCallExprE", scope: !0, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, variable: i32* @_ZN5clang23DataRecursiveASTVisitorIN12_GLOBAL__N_124UnusedBackingIvarCheckerEE26TraverseCUDAKernelCallExprEPNS_18CUDAKernelCallExprE)
!8 = !DIGlobalVariable(name: "_ZN4llvm16DenseMapIteratorIPNS_10MDLocationENS_6detail13DenseSetEmptyENS_10MDNodeInfoIS1_EENS3_12DenseSetPairIS2_EELb0EE23AdvancePastEmptyBucketsEv", scope: !0, file: !1, line: 4, type: !5, isLocal: false, isDefinition: true, variable: i32* @_ZN4llvm16DenseMapIteratorIPNS_10MDLocationENS_6detail13DenseSetEmptyENS_10MDNodeInfoIS1_EENS3_12DenseSetPairIS2_EELb0EE23AdvancePastEmptyBucketsEv)
!9 = !DIGlobalVariable(name: "_ZNK4llvm12LivePhysRegs5printERNS_11raw_ostreamE", scope: !0, file: !1, line: 5, type: !5, isLocal: false, isDefinition: true, variable: i32* @_ZNK4llvm12LivePhysRegs5printERNS_11raw_ostreamE)
!10 = !DIGlobalVariable(name: "_ZN4llvm15ScalarEvolution14getSignedRangeEPKNS_4SCEVE", scope: !0, file: !1, line: 6, type: !5, isLocal: false, isDefinition: true, variable: i32* @_ZN4llvm15ScalarEvolution14getSignedRangeEPKNS_4SCEVE)
!11 = !DIGlobalVariable(name: "k1", scope: !0, file: !1, line: 7, type: !5, isLocal: false, isDefinition: true, variable: i32* @k1)
!12 = !DIGlobalVariable(name: "is", scope: !0, file: !1, line: 8, type: !5, isLocal: false, isDefinition: true, variable: i32* @is)
!13 = !DIGlobalVariable(name: "setStmt", scope: !0, file: !1, line: 9, type: !5, isLocal: false, isDefinition: true, variable: i32* @setStmt)
!14 = !DIGlobalVariable(name: "_ZN4llvm5TwineC1Ei", scope: !0, file: !1, line: 10, type: !5, isLocal: false, isDefinition: true, variable: i32* @_ZN4llvm5TwineC1Ei)
!15 = !DIGlobalVariable(name: "_ZNK5clang12OverrideAttr5cloneERNS_10ASTContextE", scope: !0, file: !1, line: 11, type: !5, isLocal: false, isDefinition: true, variable: i32* @_ZNK5clang12OverrideAttr5cloneERNS_10ASTContextE)
!16 = !DIGlobalVariable(name: "_ZN4llvm22MachineModuleInfoMachOD2Ev", scope: !0, file: !1, line: 12, type: !5, isLocal: false, isDefinition: true, variable: i32* @_ZN4llvm22MachineModuleInfoMachOD2Ev)
!17 = !{i32 2, !"Dwarf Version", i32 2}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 1, !"PIC Level", i32 2}
!20 = !{!"clang version 3.7.0 (trunk 231548) (llvm/trunk 231547)"}
