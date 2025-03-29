; RUN: opt -passes=instcombine -S %s | FileCheck %s

define dso_local void @_Z1fv() {
  %1 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %1)
  call void @llvm.dbg.declare(metadata ptr %1, metadata !16, metadata !DIExpression()), !dbg !19
; CHECK: %[[A:.*]] = alloca i32, align 4
; CHECK: #dbg_value(ptr %[[A]], {{.*}}, !DIExpression(DW_OP_deref)
; CHECK: call void @_Z1gPv
  call void @_Z1gPv(ptr nonnull %1)
; CHECK-NOT: #dbg_value
; CHECK: call void @_Z1gPv
  call void @_Z1gPv(ptr nonnull %1)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %1)
  ret void, !dbg !21
}

define dso_local void @_Z2fv() {
  %1 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %1)
  call void @llvm.dbg.declare(metadata ptr %1, metadata !16, metadata !DIExpression()), !dbg !19
; CHECK: %[[A:.*]] = alloca i32, align 4
; CHECK: #dbg_value(ptr %[[A]], {{.*}}, !DIExpression(DW_OP_deref)
; CHECK: call void @_Z1gPv
  call void @_Z1gPv(ptr nonnull %1)
  br label %block2

block2:
; CHECK: #dbg_value(ptr %[[A]], {{.*}}, !DIExpression(DW_OP_deref)
; CHECK: call void @_Z1gPv
  call void @_Z1gPv(ptr nonnull %1)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %1)
  ret void, !dbg !21
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare dso_local void @_Z1gPv(ptr)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 10.0.0 (git@github.com:llvm/llvm-project.git cab64b5708f614c71d275ec9d134e68b8c3baedd)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "dbg.cc", directory: "/tmp")
!2 = !{}
!3 = !{!4, !5}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !DISubprogram(name: "g", linkageName: "_Z1gPv", scope: !1, file: !1, line: 1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !4}
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!12 = distinct !DISubprogram(name: "f", linkageName: "_Z1fv", scope: !1, file: !1, line: 2, type: !13, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = !{!16}
!16 = !DILocalVariable(name: "x", scope: !12, file: !1, line: 3, type: !17)
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !DILocation(line: 3, column: 3, scope: !12)
!19 = !DILocation(line: 3, column: 7, scope: !12)
!20 = !DILocation(line: 4, column: 3, scope: !12)
!21 = !DILocation(line: 5, column: 1, scope: !12)
