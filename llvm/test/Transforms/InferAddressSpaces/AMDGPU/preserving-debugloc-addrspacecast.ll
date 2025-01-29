; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=infer-address-spaces -o - %s | FileCheck %s

; Check that InferAddressSpaces's cloneInstructionWithNewAddressSpace() propagates
; the debug location to new addrspacecast instruction which casts `%p` in the following test.

@c0 = addrspace(4) global ptr poison

define float @generic_ptr_from_constant() !dbg !5 {
; CHECK-LABEL: define float @generic_ptr_from_constant(
; CHECK:    [[TMP1:%.*]] = addrspacecast ptr [[P:%.*]] to ptr addrspace(1), !dbg [[DBG8:![0-9]+]]
;
  %p = load ptr, ptr addrspace(4) @c0, align 8, !dbg !8
  %v = load float, ptr %p, align 4, !dbg !9
  ret float %v, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

; CHECK: [[DBG8]] = !DILocation(line: 1,
;
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "temp.ll", directory: "/")
!2 = !{i32 3}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "generic_ptr_from_constant", linkageName: "generic_ptr_from_constant", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5)
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DILocation(line: 3, column: 1, scope: !5)
