; RUN: opt %s  -strip-debug -o %t.no_debug.ll -S
; RUN: llc -O3 -mcpu=gfx1250 < %s             -filetype=obj -o %t.with_debug.o
; RUN: llc -O3 -mcpu=gfx1250 < %t.no_debug.ll -filetype=obj -o %t.no_debug.o
; RUN: llvm-strip %t.with_debug.o %t.no_debug.o
; RUN: cmp %t.with_debug.o %t.no_debug.o
; Ensure that compiling with and without debug generates identical code.
; Test that revertScheduling only updates LiveIntervals if non-debug
; instructions are reordered.

target triple = "amdgcn-amd-amdhsa"

declare void @llvm.amdgcn.s.barrier() #0

define amdgpu_kernel void @_test_revertScheduling(i32 %lda, ptr addrspace(1) %infoA.coerce, i32 %add.ptr13.idx, ptr addrspace(3) %add.ptr10, i32 %i0, i1 %cmp59.not, ptr addrspace(3) %arrayidx53, ptr addrspace(3) %arrayidx75.3.6, <2 x float> %f1, float %f2, <2 x float> %f3, <14 x float> %f4) {
entry:
  %cond13.in.i10.i.i.i = load i16, ptr addrspace(4) null, align 2
  %f5 = tail call i32 @llvm.amdgcn.workitem.id.x()
    #dbg_value(ptr addrspace(1) %add.ptr, !4, !DIExpression(), !13)
  %idxprom = zext i32 %f5 to i64
  %arrayidx = getelementptr float, ptr addrspace(1) null, i64 %idxprom
  %add16.1 = add i32 %lda, %f5
  %idxprom.1 = sext i32 %add16.1 to i64
  %arrayidx.1 = getelementptr float, ptr addrspace(1) null, i64 %idxprom.1
  %mul15.2 = shl i32 %lda, 1
  %add16.2 = add i32 %mul15.2, %f5
  %idxprom.2 = sext i32 %add16.2 to i64
  %arrayidx.2 = getelementptr float, ptr addrspace(1) null, i64 %idxprom.2
  %mul15.3 = mul i32 %lda, 3
  %add16.3 = add i32 %mul15.3, %f5
  %idxprom.3 = sext i32 %add16.3 to i64
  %arrayidx.3 = getelementptr float, ptr addrspace(1) null, i64 %idxprom.3
  %mul15.4 = shl i32 %lda, 2
  %add16.4 = add i32 %mul15.4, %f5
  %idxprom.4 = sext i32 %add16.4 to i64
  %arrayidx.4 = getelementptr float, ptr addrspace(1) null, i64 %idxprom.4
  %mul15.5 = mul i32 %lda, 5
  %add16.5 = add i32 %mul15.5, %f5
  %idxprom.5 = sext i32 %add16.5 to i64
  %arrayidx.5 = getelementptr float, ptr addrspace(1) null, i64 %idxprom.5
  %mul15.6 = mul i32 %lda, 6
  %add16.6 = add i32 %mul15.6, %f5
  %idxprom.6 = sext i32 %add16.6 to i64
  %arrayidx.6 = getelementptr float, ptr addrspace(1) null, i64 %idxprom.6
  %mul15.7 = mul i32 %lda, 7
  %add16.7 = add i32 %mul15.7, %f5
  %idxprom.7 = sext i32 %add16.7 to i64
  %arrayidx.7 = getelementptr float, ptr addrspace(1) null, i64 %idxprom.7
  %mul15.8 = shl i32 %lda, 3
  %add16.8 = add i32 %mul15.8, %f5
  %idxprom.8 = sext i32 %add16.8 to i64
  %arrayidx.8 = getelementptr float, ptr addrspace(1) null, i64 %idxprom.8
  %mul15.9 = mul i32 %lda, 9
  %add16.9 = add i32 %mul15.9, %f5
  %idxprom.9 = sext i32 %add16.9 to i64
  %arrayidx.9 = getelementptr float, ptr addrspace(1) null, i64 %idxprom.9
  %mul15.10 = mul i32 %lda, 10
  %add16.10 = add i32 %mul15.10, %f5
  %idxprom.10 = sext i32 %add16.10 to i64
  %arrayidx.10 = getelementptr float, ptr addrspace(1) null, i64 %idxprom.10
  %mul15.11 = mul i32 %lda, 11
  %add16.11 = add i32 %mul15.11, %f5
  %idxprom.11 = sext i32 %add16.11 to i64
  %arrayidx.11 = getelementptr float, ptr addrspace(1) null, i64 %idxprom.11
  %mul15.12 = mul i32 %lda, 12
  %add16.12 = add i32 %mul15.12, %f5
  %idxprom.12 = sext i32 %add16.12 to i64
  %arrayidx.12 = getelementptr float, ptr addrspace(1) null, i64 %idxprom.12
  %mul15.13 = mul i32 %lda, 13
  %add16.13 = add i32 %mul15.13, %f5
  %idxprom.13 = sext i32 %add16.13 to i64
  %arrayidx.13 = getelementptr float, ptr addrspace(1) null, i64 %idxprom.13
  %f6 = load float, ptr addrspace(1) %arrayidx.13, align 4
  %cmp54 = fcmp oeq float %f6, 0.000000e+00
  br i1 %cmp59.not, label %if.end82, label %for.body71.preheader

for.body71.preheader:                             ; preds = %entry
  %f7 = load float, ptr addrspace(1) %arrayidx, align 4
  %mul65 = fmul float %f7, 0.000000e+00
  %f8 = insertelement <14 x float> zeroinitializer, float %mul65, i64 0
  br label %if.end82

if.end82:                                         ; preds = %for.body71.preheader, %entry
  %f9 = load <2 x float>, ptr addrspace(3) zeroinitializer, align 8
  tail call void @llvm.amdgcn.s.barrier()
  %f10 = load float, ptr addrspace(3) %add.ptr10, align 4
  %f11 = load float, ptr addrspace(3) zeroinitializer, align 4
  fence acquire
  %f12 = load float, ptr addrspace(3) %add.ptr10, align 4
  %f13 = load float, ptr addrspace(3) %arrayidx53, align 4
  %cmp54.2 = fcmp oeq float %f10, 0.000000e+00
  %cmp54.1 = fcmp une float %f13, 0.000000e+00
  %spec.select.1 = select i1 %cmp54.1, i32 %add16.13, i32 0
  %spec.select.2 = select i1 %cmp54.2, i32 0, i32 %spec.select.1
  %mul65.3 = fmul float 0.000000e+00, %f2
  %f14 = insertelement <2 x float> zeroinitializer, float %mul65.3, i64 0
  %f15 = fmul <2 x float> zeroinitializer, %f9
  %f16 = shufflevector <2 x float> %f15, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f17 = shufflevector <14 x float> zeroinitializer, <14 x float> %f16, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 14, i32 15>
  %f18 = insertelement <14 x float> %f17, float %f11, i64 13
  %f19 = shufflevector <14 x float> %f18, <14 x float> zeroinitializer, <2 x i32> <i32 12, i32 13>
  %f20 = load <2 x float>, ptr addrspace(3) zeroinitializer, align 8
  %f21 = fmul contract <2 x float> zeroinitializer, %f20
  store float 0.000000e+00, ptr addrspace(3) zeroinitializer, align 4
  %f22 = load <2 x float>, ptr addrspace(3) %arrayidx75.3.6, align 8
  %f23 = fmul <2 x float> zeroinitializer, %f22
  %f24 = shufflevector <2 x float> %f23, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f25 = shufflevector <14 x float> %f4, <14 x float> %f24, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 14, i32 15, i32 poison, i32 poison>
  %f26 = fsub contract <2 x float> %f19, %f21
  %f27 = shufflevector <2 x float> %f26, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f28 = shufflevector <14 x float> %f25, <14 x float> %f27, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 14, i32 15>
  %f29 = extractelement <14 x float> %f28, i64 0
  %f30 = load <2 x float>, ptr addrspace(3) zeroinitializer, align 4
  %f31 = shufflevector <14 x float> %f28, <14 x float> zeroinitializer, <2 x i32> <i32 11, i32 12>
  %f32 = fsub <2 x float> zeroinitializer, %f14
  %f33 = fmul contract <2 x float> %f32, zeroinitializer
  %f34 = fsub contract <2 x float> %f31, %f33
  fence release
  %f35 = load <2 x float>, ptr addrspace(3) inttoptr (i32 32 to ptr addrspace(3)), align 8
  %f36 = fmul contract <2 x float> zeroinitializer, %f35
  %f37 = load <2 x float>, ptr addrspace(3) inttoptr (i32 24 to ptr addrspace(3)), align 8
  %f38 = fmul contract <2 x float> %f3, %f37
  %f39 = fsub contract <2 x float> %f33, %f38
  %f40 = shufflevector <2 x float> %f39, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f41 = shufflevector <14 x float> zeroinitializer, <14 x float> %f40, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f42 = fsub contract <2 x float> %f1, %f36
  %f43 = shufflevector <2 x float> %f42, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f44 = shufflevector <14 x float> %f41, <14 x float> %f43, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison>
  %f45 = shufflevector <14 x float> %f44, <14 x float> zeroinitializer, <2 x i32> <i32 7, i32 8>
  %f46 = fmul contract <2 x float> zeroinitializer, %f3
  %f47 = fsub contract <2 x float> %f45, %f46
  %f48 = load <2 x float>, ptr addrspace(3) inttoptr (i32 48 to ptr addrspace(3)), align 8
  %f49 = fsub <2 x float> %f34, %f48
  %f50 = fmul <2 x float> zeroinitializer, %f30
  %f51 = shufflevector <2 x float> %f47, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f52 = shufflevector <14 x float> %f44, <14 x float> %f51, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f53 = fsub <2 x float> %f49, %f50
  %f54 = shufflevector <2 x float> %f53, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f55 = shufflevector <14 x float> %f52, <14 x float> %f54, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 14, i32 15, i32 poison>
  %f56 = extractelement <14 x float> %f55, i64 7
  %mul65.7 = fmul float %f56, 0.000000e+00
  %f57 = load <2 x float>, ptr addrspace(3) inttoptr (i32 40 to ptr addrspace(3)), align 8
  store float 0.000000e+00, ptr addrspace(3) %add.ptr10, align 4
  %f58 = insertelement <2 x float> zeroinitializer, float %mul65.7, i64 0
  %f59 = shufflevector <14 x float> %f55, <14 x float> zeroinitializer, <2 x i32> <i32 10, i32 11>
  %f60 = fmul <2 x float> %f58, %f57
  %f61 = fsub <2 x float> %f59, %f60
  %f62 = load <2 x float>, ptr addrspace(3) %arrayidx53, align 4
  %f63 = fsub <2 x float> %f61, %f62
  %f64 = shufflevector <2 x float> %f63, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %cmp59.13 = icmp ugt i32 %f5, 0
  %f65 = shufflevector <14 x float> zeroinitializer, <14 x float> %f64, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 14, i32 15, i32 poison, i32 poison, i32 poison>
  %promotealloca.13 = select i1 %cmp59.13, <14 x float> zeroinitializer, <14 x float> %f65
  %cmp54.3 = fcmp oeq float %f12, 0.000000e+00
  %cmp54.4 = fcmp oeq float %f29, 0.000000e+00
  %spec.select.3 = select i1 %cmp54.3, i32 0, i32 %spec.select.2
  %spec.select.4 = select i1 %cmp54.4, i32 0, i32 %spec.select.3
  %conv.i.i = zext i16 %cond13.in.i10.i.i.i to i32
  %f66 = tail call i32 @llvm.amdgcn.workitem.id.y()
  %add = or i32 %conv.i.i, %f66
  %conv.i = sext i32 %add to i64
  %add.ptr = getelementptr i32, ptr addrspace(1) %infoA.coerce, i64 %conv.i
  store i32 %spec.select.4, ptr addrspace(1) %add.ptr, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.1, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.2, align 4
  store float %mul65.3, ptr addrspace(1) %arrayidx.3, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.4, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.5, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.6, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.7, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.8, align 4
  %f67 = extractelement <14 x float> %promotealloca.13, i64 9
  store float %f67, ptr addrspace(1) %arrayidx.9, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.10, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.11, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.12, align 4
  store float 0.000000e+00, ptr addrspace(1) null, align 4
  ret void
}

declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.y() #1

attributes #0 = { convergent nocallback nofree nounwind willreturn }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "AMD clang version 22.0.0git (ssh://github-emu/AMD-Lightning-Internal/llvm-project  25425 c51a87b7a53a3e8f308402aaffa3ecbc2953305a)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, imports: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/tmp", checksumkind: CSK_MD5, checksum: "cc205700bf3536fe4ff21a07daf7e01d")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DILocalVariable(name: "info", scope: !5, file: !6, line: 162, type: !11)
!5 = distinct !DISubprogram(name: "test_revertScheduling", linkageName: "_test_revertScheduling", scope: !7, file: !6, line: 142, type: !9, scopeLine: 150, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, templateParams: !2, retainedNodes: !2)
!6 = !DIFile(filename: "kernels.hpp", directory: "/tmp")
!7 = !DINamespace(name: "v33200", scope: !8, exportSymbols: true)
!8 = !DINamespace(name: "solve", scope: null)
!9 = distinct !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 0, scope: !5)
