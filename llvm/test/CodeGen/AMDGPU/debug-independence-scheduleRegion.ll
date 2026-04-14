; Ensure that compiling with and without debug generates identical code.
; Test that scheduleRegion in iterative scheduler only updates LiveIntervals if non-debug
; instructions are reordered.

; RUN: opt %s  -strip-debug -o %t.no_debug.ll -S
; RUN: llc -O2 -mcpu=gfx1250 < %s             -misched=gcn-iterative-ilp -filetype=obj -o %t.with_debug.o
; RUN: llc -O2 -mcpu=gfx1250 < %t.no_debug.ll -misched=gcn-iterative-ilp -filetype=obj -o %t.no_debug.o
; RUN: llvm-strip %t.with_debug.o %t.no_debug.o
; RUN: cmp %t.with_debug.o %t.no_debug.o

target triple = "amdgcn-amd-amdhsa"

declare void @llvm.amdgcn.s.barrier() #0

define amdgpu_kernel void @_test_scheduleRegion(<2 x float> %f9, <2 x float> %f15, <14 x float> %f16, <2 x float> %f14, <2 x float> %f33, i1 %cmp59.13, <2 x float> %f48, ptr addrspace(1) %arrayidx.1) {
entry:
    #dbg_value(ptr addrspace(1) null, !4, !DIExpression(), !13)
  tail call void @llvm.amdgcn.s.barrier()
  fence acquire
  store float 0.000000e+00, ptr addrspace(3) zeroinitializer, align 4
  %f26 = fsub <2 x float> %f14, %f33
  %f27 = shufflevector <2 x float> %f26, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f28 = shufflevector <14 x float> zeroinitializer, <14 x float> %f27, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 14, i32 15>
  %f31 = shufflevector <14 x float> %f28, <14 x float> zeroinitializer, <2 x i32> <i32 11, i32 12>
  %f34 = fsub <2 x float> %f31, %f15
  store float 0.000000e+00, ptr addrspace(1) null, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.1, align 4
  store float 0.000000e+00, ptr addrspace(1) null, align 4
  %f49 = fsub <2 x float> %f34, %f48
  %f61 = fsub <2 x float> %f49, %f9
  %f64 = shufflevector <2 x float> %f61, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f65 = shufflevector <14 x float> zeroinitializer, <14 x float> %f64, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 14, i32 15, i32 poison, i32 poison, i32 poison>
  %promotealloca.13 = select i1 %cmp59.13, <14 x float> zeroinitializer, <14 x float> %f65
  %f67 = extractelement <14 x float> %promotealloca.13, i64 9
  store float %f67, ptr addrspace(1) %arrayidx.1, align 4
  ret void
}

attributes #0 = { convergent nocallback nofree nounwind willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "AMD clang version 22.0.0git (ssh://github-emu/AMD-Lightning-Internal/llvm-project  25425 c51a87b7a53a3e8f308402aaffa3ecbc2953305a)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, imports: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/tmp", checksumkind: CSK_MD5, checksum: "cc205700bf3536fe4ff21a07daf7e01d")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DILocalVariable(name: "info", scope: !5, file: !6, line: 162, type: !11)
!5 = distinct !DISubprogram(name: "test_scheduleRegion", linkageName: "_test_scheduleRegion", scope: !7, file: !6, line: 142, type: !9, scopeLine: 150, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, templateParams: !2, retainedNodes: !2)
!6 = !DIFile(filename: "kernels.hpp", directory: "/tmp")
!7 = !DINamespace(name: "v33200", scope: !8, exportSymbols: true)
!8 = !DINamespace(name: "solve", scope: null)
!9 = distinct !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 0, scope: !5)
