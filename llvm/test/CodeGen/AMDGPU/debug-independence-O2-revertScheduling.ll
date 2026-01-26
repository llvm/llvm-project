; RUN: opt %s  -strip-debug -o %t.no_debug.ll -S
; RUN: llc -O2 -mcpu=gfx1250 < %s             -filetype=obj -amdgpu-force-revert-scheduling -o %t.with_debug.o
; RUN: llc -O2 -mcpu=gfx1250 < %t.no_debug.ll -filetype=obj -amdgpu-force-revert-scheduling -o %t.no_debug.o
; RUN: llvm-strip %t.with_debug.o %t.no_debug.o
; RUN: cmp %t.with_debug.o %t.no_debug.o
; Ensure that compiling with and without debug generates identical code.
; Test that revertScheduling only updates LiveIntervals if non-debug
; instructions are reordered.

target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @_test_revertScheduling(i32 %lda, ptr addrspace(1) %infoA.coerce, ptr addrspace(3) %add.ptr13, ptr addrspace(3) %arrayidx75.1.2, ptr addrspace(3) %arrayidx75.1.4, <2 x float> %f0, <14 x float> %f1, <2 x float> %f2, ptr addrspace(3) %arrayidx75.1.8, <2 x float> %f3, <2 x float> %f4, ptr addrspace(3) %arrayidx75.2.2, ptr addrspace(3) %arrayidx75.2.4) {
entry:
    #dbg_value(i32 0, !4, !DIExpression(), !13)
  %f5 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %idxprom = zext i32 %f5 to i64
  %arrayidx = getelementptr float, ptr addrspace(1) null, i64 %idxprom
  %add16.1 = add nsw i32 %lda, %f5
  %idxprom.1 = sext i32 %add16.1 to i64
  %arrayidx.1 = getelementptr float, ptr addrspace(1) null, i64 %idxprom.1
  %f6 = load float, ptr addrspace(1) %arrayidx.1, align 4
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
  %arrayidx.81 = getelementptr float, ptr addrspace(1) null, i64 %idxprom.8
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
  %f7 = load float, ptr addrspace(1) %arrayidx.4, align 4
  %f8 = insertelement <14 x float> zeroinitializer, float %f7, i64 4
  %f9 = load float, ptr addrspace(1) %arrayidx.5, align 4
  %f10 = insertelement <14 x float> %f8, float %f9, i64 5
  %f11 = shufflevector <14 x float> %f10, <14 x float> zeroinitializer, <2 x i32> <i32 4, i32 5>
  %f12 = load <2 x float>, ptr addrspace(3) %add.ptr13, align 8
  %f13 = fmul contract <2 x float> zeroinitializer, %f12
  %f14 = load float, ptr addrspace(1) %arrayidx.6, align 4
  %f15 = insertelement <14 x float> %f8, float %f14, i64 6
  %f16 = load float, ptr addrspace(1) %arrayidx.7, align 4
  %f17 = insertelement <14 x float> %f15, float %f16, i64 7
  %f18 = load <2 x float>, ptr addrspace(3) null, align 8
  %f19 = fsub contract <2 x float> %f11, %f13
  %f20 = shufflevector <2 x float> %f19, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f21 = shufflevector <14 x float> %f17, <14 x float> zeroinitializer, <2 x i32> <i32 6, i32 7>
  %f22 = fmul contract <2 x float> zeroinitializer, %f18
  %f23 = fsub contract <2 x float> %f21, %f22
  %f24 = fmul contract <2 x float> zeroinitializer, %f3
  %arrayidx.13 = getelementptr float, ptr addrspace(1) null, i64 %idxprom.13
  %f25 = load float, ptr addrspace(1) %arrayidx.13, align 4
  %f26 = insertelement <14 x float> zeroinitializer, float %f25, i64 13
  %f27 = shufflevector <14 x float> %f26, <14 x float> zeroinitializer, <2 x i32> <i32 12, i32 13>
  %f28 = shufflevector <14 x float> zeroinitializer, <14 x float> %f20, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f29 = shufflevector <2 x float> %f23, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f30 = shufflevector <14 x float> %f28, <14 x float> %f29, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f31 = fsub <2 x float> %f2, %f0
  %f32 = shufflevector <2 x float> %f31, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f33 = shufflevector <14 x float> %f30, <14 x float> %f32, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison>
  %f34 = fsub <2 x float> %f27, splat (float 1.000000e+00)
  %f35 = shufflevector <2 x float> %f34, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f36 = shufflevector <14 x float> %f33, <14 x float> %f35, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 14, i32 15>
  %f37 = extractelement <14 x float> %f36, i64 13
  store float %f37, ptr addrspace(3) null, align 4
  %f38 = shufflevector <14 x float> %f30, <14 x float> zeroinitializer, <2 x i32> <i32 5, i32 6>
  %f39 = load <2 x float>, ptr addrspace(3) %arrayidx75.2.2, align 4
  %f40 = fmul contract <2 x float> zeroinitializer, %f39
  %f41 = load <2 x float>, ptr addrspace(3) %arrayidx75.2.4, align 4
  %f42 = fmul <2 x float> zeroinitializer, %f41
  %f43 = shufflevector <2 x float> %f42, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f44 = fsub contract <2 x float> %f4, %f24
  %f45 = shufflevector <2 x float> %f44, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f46 = shufflevector <14 x float> %f33, <14 x float> %f45, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 14, i32 15, i32 poison, i32 poison>
  %f47 = shufflevector <14 x float> %f46, <14 x float> zeroinitializer, <2 x i32> <i32 9, i32 10>
  %f48 = load <2 x float>, ptr addrspace(3) %arrayidx75.1.4, align 4
  %f49 = fsub <2 x float> %f47, %f48
  %f50 = load <2 x float>, ptr addrspace(3) %arrayidx75.1.2, align 4
  %f51 = fmul <2 x float> zeroinitializer, %f50
  %f52 = load <2 x float>, ptr addrspace(3) %add.ptr13, align 8
  %f53 = fmul contract <2 x float> zeroinitializer, %f52
  %f54 = shufflevector <14 x float> zeroinitializer, <14 x float> %f43, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f55 = shufflevector <2 x float> %f49, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f56 = shufflevector <14 x float> %f54, <14 x float> %f55, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 14, i32 15, i32 poison, i32 poison, i32 poison>
  %f57 = shufflevector <14 x float> %f56, <14 x float> zeroinitializer, <2 x i32> <i32 8, i32 9>
  %f58 = fsub contract <2 x float> %f38, %f40
  %f59 = shufflevector <2 x float> %f58, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f60 = shufflevector <14 x float> zeroinitializer, <14 x float> %f59, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f61 = fsub contract <2 x float> %f57, %f53
  %f62 = shufflevector <2 x float> %f61, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f63 = shufflevector <14 x float> %f60, <14 x float> %f62, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison>
  %f64 = shufflevector <2 x float> %f51, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f65 = shufflevector <14 x float> %f56, <14 x float> %f64, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 14, i32 15, i32 poison>
  %f66 = shufflevector <14 x float> %f65, <14 x float> zeroinitializer, <2 x i32> <i32 10, i32 11>
  %f67 = load <2 x float>, ptr addrspace(3) null, align 8
  %f68 = fmul contract <2 x float> zeroinitializer, %f67
  %f69 = fsub contract <2 x float> %f66, %f68
  store float 0.000000e+00, ptr addrspace(3) null, align 4
  %f70 = shufflevector <2 x float> %f69, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f71 = shufflevector <14 x float> %f63, <14 x float> %f70, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 14, i32 15, i32 poison, i32 poison>
  %f72 = shufflevector <14 x float> %f71, <14 x float> zeroinitializer, <2 x i32> <i32 5, i32 6>
  %f73 = load <2 x float>, ptr addrspace(3) null, align 4
  %f74 = fsub <2 x float> %f72, %f73
  %f75 = shufflevector <2 x float> %f74, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f76 = shufflevector <14 x float> zeroinitializer, <14 x float> %f75, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %f77 = shufflevector <14 x float> %f71, <14 x float> zeroinitializer, <2 x i32> <i32 9, i32 10>
  %f78 = fmul contract <2 x float> zeroinitializer, %f0
  %f79 = fsub contract <2 x float> %f77, %f78
  %f80 = shufflevector <2 x float> %f79, <2 x float> zeroinitializer, <14 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  fence release
  %f81 = shufflevector <14 x float> %f76, <14 x float> %f80, <14 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 14, i32 15, i32 poison, i32 poison, i32 poison>
  %f82 = extractelement <14 x float> %f81, i64 5
  store float %f82, ptr addrspace(3) %arrayidx75.1.2, align 4
  %f83 = shufflevector <14 x float> %f81, <14 x float> zeroinitializer, <2 x i32> <i32 10, i32 11>
  store <2 x float> %f83, ptr addrspace(3) null, align 8
  store <2 x float> zeroinitializer, ptr addrspace(3) %add.ptr13, align 8
  %f84 = tail call i32 @llvm.amdgcn.workitem.id.y()
  %conv.i = sext i32 %f84 to i64
  %add.ptr = getelementptr i32, ptr addrspace(1) %infoA.coerce, i64 %conv.i
  store i32 0, ptr addrspace(1) %add.ptr, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.1, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.2, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.3, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.4, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.5, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.6, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.7, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.81, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.9, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.10, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.11, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.12, align 4
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.13, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.y() #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "AMD clang version 22.0.0git (ssh://github-emu/AMD-Lightning-Internal/llvm-project  25425 c51a87b7a53a3e8f308402aaffa3ecbc2953305a)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, imports: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/tmp", checksumkind: CSK_MD5, checksum: "cc205700bf3536fe4ff21a07daf7e01d")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DILocalVariable(name: "m", arg: 1, scope: !5, file: !6, line: 142, type: !11)
!5 = distinct !DISubprogram(name: "getf2_npvt_small_kernel<14, float, int, int, float *>", linkageName: "_test_revertScheduling", scope: !7, file: !6, line: 142, type: !9, scopeLine: 150, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, templateParams: !2, retainedNodes: !2)
!6 = !DIFile(filename: "kernels.hpp", directory: "/tmp")
!7 = !DINamespace(name: "v33200", scope: !8, exportSymbols: true)
!8 = !DINamespace(name: "solver", scope: null)
!9 = distinct !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 0, scope: !5)
