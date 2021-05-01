; RUN: opt < %s -reassociate -S | FileCheck %s

; Check that reassociate pass now undefs debug intrinsics that reference a value
; that gets dropped and cannot be salvaged.

define hidden i32 @main() local_unnamed_addr {
entry:
  %foo = alloca i32, align 4, !dbg !20
  %foo.0.foo.0..sroa_cast = bitcast i32* %foo to i8*, !dbg !20
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %foo.0.foo.0..sroa_cast), !dbg !20
  store volatile i32 4, i32* %foo, align 4, !dbg !20, !tbaa !21
  %foo.0.foo.0. = load volatile i32, i32* %foo, align 4, !dbg !25, !tbaa !21
  %foo.0.foo.0.15 = load volatile i32, i32* %foo, align 4, !dbg !27, !tbaa !21
  %foo.0.foo.0.16 = load volatile i32, i32* %foo, align 4, !dbg !28, !tbaa !21
  ; CHECK-NOT: %add = add nsw i32 %foo.0.foo.0., %foo.0.foo.0.15
  %add = add nsw i32 %foo.0.foo.0., %foo.0.foo.0.15, !dbg !29
  ; CHECK: call void @llvm.dbg.value(metadata i32 undef, metadata [[VAR_A:![0-9]+]], metadata !DIExpression())
  call void @llvm.dbg.value(metadata i32 %add, metadata !19, metadata !DIExpression()), !dbg !26
  %foo.0.foo.0.17 = load volatile i32, i32* %foo, align 4, !dbg !30, !tbaa !21
  %cmp = icmp eq i32 %foo.0.foo.0.17, 4, !dbg !30
  br i1 %cmp, label %if.then, label %if.end, !dbg !32

  ; CHECK-LABEL: if.then:
if.then:
  ; CHECK-NOT: %add1 = add nsw i32 %add, %foo.0.foo.0.16
  %add1 = add nsw i32 %add, %foo.0.foo.0.16, !dbg !33
  ; CHECK: call void @llvm.dbg.value(metadata i32 undef, metadata [[VAR_A]], metadata !DIExpression())
  call void @llvm.dbg.value(metadata i32 %add1, metadata !19, metadata !DIExpression()), !dbg !26
  ; CHECK: call void @llvm.dbg.value(metadata i32 undef, metadata [[VAR_CHEESE:![0-9]+]], metadata !DIExpression())
  call void @llvm.dbg.value(metadata i32 %add, metadata !18, metadata !DIExpression()), !dbg !26
  %sub = add nsw i32 %add, -12, !dbg !34
  %sub3 = sub nsw i32 %add1, %sub, !dbg !34
  %mul = mul nsw i32 %sub3, 20, !dbg !36
  %div = sdiv i32 %mul, 3, !dbg !37
  br label %if.end, !dbg !38

if.end:
  %a.0 = phi i32 [ %div, %if.then ], [ 0, %entry ], !dbg !39
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %foo.0.foo.0..sroa_cast), !dbg !40
  ret i32 %a.0, !dbg !41
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 10.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "F:\")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 10.0.0"}
!8 = distinct !DISubprogram(name: "main", scope: !9, file: !9, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!9 = !DIFile(filename: "./test.cpp", directory: "F:\")
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{!14, !16, !17, !18, !19}
!14 = !DILocalVariable(name: "foo", scope: !8, file: !9, line: 2, type: !15)
!15 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !12)
!16 = !DILocalVariable(name: "read1", scope: !8, file: !9, line: 3, type: !12)
!17 = !DILocalVariable(name: "read2", scope: !8, file: !9, line: 4, type: !12)
; CHECK: [[VAR_CHEESE]] = !DILocalVariable(name: "cheese"
!18 = !DILocalVariable(name: "cheese", scope: !8, file: !9, line: 6, type: !12)
; CHECK: [[VAR_A]] = !DILocalVariable(name: "a"
!19 = !DILocalVariable(name: "a", scope: !8, file: !9, line: 7, type: !12)
!20 = !DILocation(line: 2, scope: !8)
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !DILocation(line: 3, scope: !8)
!26 = !DILocation(line: 0, scope: !8)
!27 = !DILocation(line: 4, scope: !8)
!28 = !DILocation(line: 6, scope: !8)
!29 = !DILocation(line: 7, scope: !8)
!30 = !DILocation(line: 10, scope: !31)
!31 = distinct !DILexicalBlock(scope: !8, file: !9, line: 10)
!32 = !DILocation(line: 10, scope: !8)
!33 = !DILocation(line: 8, scope: !8)
!34 = !DILocation(line: 12, scope: !35)
!35 = distinct !DILexicalBlock(scope: !31, file: !9, line: 10)
!36 = !DILocation(line: 13, scope: !35)
!37 = !DILocation(line: 14, scope: !35)
!38 = !DILocation(line: 15, scope: !35)
!39 = !DILocation(line: 0, scope: !31)
!40 = !DILocation(line: 20, scope: !8)
!41 = !DILocation(line: 19, scope: !8)
