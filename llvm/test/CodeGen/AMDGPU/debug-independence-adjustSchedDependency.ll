; RUN: opt %s  -strip-debug -o %t.no_debug.ll -S
; RUN: llc -mcpu=gfx1250 < %s             -filetype=obj -o %t.with_debug.o
; RUN: llc -mcpu=gfx1250 < %t.no_debug.ll -filetype=obj -o %t.no_debug.o
; RUN: llvm-strip %t.with_debug.o %t.no_debug.o
; RUN: cmp %t.with_debug.o %t.no_debug.o
; Ensure that compiling with and without debug generates identical code.
; Test that adjustSchedDependency does not count debug instructions in bundles.

target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @_test_adjustSchedDependency(ptr addrspace(1) %AA.coerce, i64 %shiftA, i32 %lda, ptr addrspace(3) %stPtr) !dbg !4 {
entry:
    #dbg_value(i32 0, !10, !DIExpression(), !13)
    #dbg_value(ptr addrspace(1) %AA.coerce, !14, !DIExpression(), !13)
  %add.ptr1.i = getelementptr float, ptr addrspace(1) %AA.coerce, i64 %shiftA
  %mul15.13 = mul i32 %lda, 13
  %idxprom.13 = sext i32 %mul15.13 to i64
  %arrayidx.13 = getelementptr float, ptr addrspace(1) %add.ptr1.i, i64 %idxprom.13
  %floatval = load float, ptr addrspace(1) %arrayidx.13, align 4
  %floatpair = insertelement <2 x float> zeroinitializer, float %floatval, i64 0
  store <2 x float> %floatpair, ptr addrspace(3) %stPtr, align 4
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "AMD clang version 22.0.0git (ssh://github-emu/AMD-Lightning-Internal/llvm-project  25425 c51a87b7a53a3e8f308402aaffa3ecbc2953305a)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, imports: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/tmp", checksumkind: CSK_MD5, checksum: "cc205700bf3536fe4ff21a07daf7e01d")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test_adjustSchedDependency", linkageName: "_test_adjustSchedDependency", scope: !6, file: !5, line: 142, type: !8, scopeLine: 150, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, templateParams: !2, retainedNodes: !2)
!5 = !DIFile(filename: "kernels.hpp", directory: "/tmp")
!6 = !DINamespace(name: "v33200", scope: !7, exportSymbols: true)
!7 = !DINamespace(name: "solve", scope: null)
!8 = distinct !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "m", arg: 1, scope: !4, file: !5, line: 142, type: !11)
!11 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 0, scope: !4)
!14 = !DILocalVariable(name: "AA", arg: 2, scope: !4, file: !5, line: 143, type: !15)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
