; This test was obtained from swift source code and then automatically reducing it via Delta.
; The swift source code was from the test test/DebugInfo/debug_scope_distinct.swift.

; RUN: opt %s -S -p=sroa -o - | FileCheck %s

; CHECK: [[SROA_5_SROA_21:%.*]] = alloca [7 x i8], align 8
; CHECK-NEXT: #dbg_value(ptr [[SROA_5_SROA_21]], [[META59:![0-9]+]], !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 72, 56), [[DBG72:![0-9]+]])

; CHECK: #dbg_value(ptr [[REG1:%[0-9]+]], [[META54:![0-9]+]], !DIExpression(DW_OP_deref), [[DBG78:![0-9]+]]
; CHECK-NEXT: #dbg_value(ptr [[REG2:%[0-9]+]], [[META56:![0-9]+]], !DIExpression(DW_OP_deref), [[DBG78]]
; CHECK-NEXT: #dbg_value(i64 0, [[META57:![0-9]+]], !DIExpression(), [[DBG78]]

; CHECK: [[SROA_418_SROA_COPYLOAD:%.*]] = load i8, ptr [[SROA_418_0_U1_IDX:%.*]], align 8, !dbg [[DBG78]]
; CHECK-NEXT: #dbg_value(i8 [[SROA_418_SROA_COPYLOAD]], [[META59]], !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 64, 8), [[DBG72]]

%T4main1TV13TangentVectorV = type <{ %T4main1UV13TangentVectorV, [7 x i8], %T4main1UV13TangentVectorV }>
%T4main1UV13TangentVectorV = type <{ %T1M1SVySfG, [7 x i8], %T4main1VV13TangentVectorV }>
%T1M1SVySfG = type <{ ptr, %Ts4Int8V }>
%Ts4Int8V = type <{ i8 }>
%T4main1VV13TangentVectorV = type <{ %T1M1SVySfG }>
define hidden swiftcc void @"$s4main1TV13TangentVectorV1poiyA2E_AEtFZ"(ptr noalias nocapture sret(%T4main1TV13TangentVectorV) %0, ptr noalias nocapture dereferenceable(57) %1, ptr noalias nocapture dereferenceable(57) %2) #0 !dbg !44 {
entry:
  %3 = alloca %T4main1VV13TangentVectorV
  %4 = alloca %T4main1UV13TangentVectorV
  call void @llvm.dbg.value(metadata ptr %1, metadata !54, metadata !DIExpression(DW_OP_deref)), !dbg !61
  call void @llvm.dbg.value(metadata ptr %2, metadata !56, metadata !DIExpression(DW_OP_deref)), !dbg !61
  call void @llvm.dbg.value(metadata i64 0, metadata !57, metadata !DIExpression()), !dbg !61
  %.u1 = getelementptr inbounds %T4main1TV13TangentVectorV, ptr %1, i32 0, i32 0
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %4, ptr align 8 %.u1, i64 25, i1 false), !dbg !61
  call void @llvm.dbg.value(metadata ptr %4, metadata !62, metadata !DIExpression(DW_OP_deref)), !dbg !75
  %.s = getelementptr inbounds %T4main1UV13TangentVectorV, ptr %4, i32 0, i32 0
  %.s.b = getelementptr inbounds %T1M1SVySfG, ptr %.s, i32 0, i32 1
  %.s.b._value = getelementptr inbounds %Ts4Int8V, ptr %.s.b, i32 0, i32 0
  %12 = load i8, ptr %.s.b._value
  %.v = getelementptr inbounds %T4main1UV13TangentVectorV, ptr %4, i32 0, i32 2
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %3, ptr align 8 %.v, i64 9, i1 false)
  %.s4 = getelementptr inbounds %T4main1VV13TangentVectorV, ptr %3, i32 0, i32 0
  %.s4.c = getelementptr inbounds %T1M1SVySfG, ptr %.s4, i32 0, i32 0
  %18 = load ptr, ptr %.s4.c
  ret void
}
!llvm.module.flags = !{ !7, !15}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"Swift Minor Version", i8 0}
!16 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !17, sdk: "MacOSX14.4.sdk")
!17 = !DIFile(filename: "/Users/debug_scope_distinct.swift", directory: "/Users/")
!44 = distinct !DISubprogram( unit: !16, retainedNodes: !53)
!53 = !{}
!54 = !DILocalVariable(name: "lhs", scope: !44, flags: DIFlagArtificial)
!56 = !DILocalVariable(name: "rhs", scope: !44, flags: DIFlagArtificial)
!57 = !DILocalVariable(name: "self", scope: !44, flags: DIFlagArtificial)
!61 = !DILocation( scope: !44)
!62 = !DILocalVariable( scope: !63, flags: DIFlagArtificial)
!63 = distinct !DISubprogram( unit: !16, retainedNodes: !70)
!70 = !{}
!75 = !DILocation( scope: !63, inlinedAt: !76)
!76 = distinct !DILocation( scope: !44)
