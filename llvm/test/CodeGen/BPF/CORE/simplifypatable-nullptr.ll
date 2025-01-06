; RUN: llc -O2 -mtriple=bpf -mcpu=v3 < %s | FileCheck %s
; Source code:
;   struct t3 {
;     int i;
;   } __attribute__((preserve_access_index));
;   struct t2 {
;     void *pad;
;     struct t3 *f;
;  } __attribute__((preserve_access_index));
;  struct t1 {
;    void *pad;
;    struct t2 *q;
;  } __attribute__((preserve_access_index));
;
;  int g;
;  int test(struct t1 *p) {
;    struct t2 *q = p->q;
;    if (q)
;      return 0;
;    struct t3 *f = q->f;
;    if (!f) g = 5;
;    return 0;
;  }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

@g = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0
@"llvm.t2:0:8$0:1" = external global i64, !llvm.preserve.access.index !6 #0
@"llvm.t1:0:8$0:1" = external global i64, !llvm.preserve.access.index !15 #0

; Function Attrs: mustprogress nofree nosync nounwind willreturn
define dso_local i32 @test(ptr noundef readonly %p) local_unnamed_addr #1 !dbg !25 {
entry:
  call void @llvm.dbg.value(metadata ptr %p, metadata !30, metadata !DIExpression()), !dbg !33
  %0 = load i64, ptr @"llvm.t1:0:8$0:1", align 8
  %1 = getelementptr i8, ptr %p, i64 %0
  %2 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 1, ptr %1)
  %3 = load ptr, ptr %2, align 8, !dbg !34, !tbaa !35
  call void @llvm.dbg.value(metadata ptr %3, metadata !31, metadata !DIExpression()), !dbg !33
  %tobool.not = icmp eq ptr %3, null, !dbg !40
  br i1 %tobool.not, label %if.end, label %cleanup, !dbg !42

; CHECK-LABEL: test
; CHECK:       r1 = *(u64 *)(r1 + 8)
; CHECK:       if r1 != 0 goto

if.end:                                           ; preds = %entry
  %4 = load i64, ptr @"llvm.t2:0:8$0:1", align 8
  %5 = getelementptr i8, ptr null, i64 %4
  %6 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 0, ptr %5)
  %7 = load ptr, ptr %6, align 8, !dbg !43, !tbaa !44
  call void @llvm.dbg.value(metadata ptr %7, metadata !32, metadata !DIExpression()), !dbg !33
  %tobool1.not = icmp eq ptr %7, null, !dbg !46
  br i1 %tobool1.not, label %if.then2, label %cleanup, !dbg !48

; CHECK:       r1 = 8
; CHECK:       r1 = *(u64 *)(r1 + 0)
; CHECK:       if r1 != 0 goto

if.then2:                                         ; preds = %if.end
  store i32 5, ptr @g, align 4, !dbg !49, !tbaa !50
  br label %cleanup, !dbg !52

cleanup:                                          ; preds = %if.end, %if.then2, %entry
  ret i32 0, !dbg !53
}

; Function Attrs: nofree nosync nounwind readnone
declare ptr @llvm.bpf.passthrough.p0.p0(i32, ptr) #2

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { "btf_ama" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nofree nosync nounwind readnone }
attributes #3 = { nocallback nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!20, !21, !22, !23}
!llvm.ident = !{!24}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 13, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git ca2be81e34a6d87edb8e555dfac94ab68ee20f70)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/tmp/home/yhs/work/tests/llvm/nullptr", checksumkind: CSK_MD5, checksum: "2c0ea9b3c647baf31f56992f9142b0df")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t2", file: !3, line: 4, size: 128, elements: !7)
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "pad", scope: !6, file: !3, line: 5, baseType: !9, size: 64)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !6, file: !3, line: 6, baseType: !11, size: 64, offset: 64)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t3", file: !3, line: 1, size: 32, elements: !13)
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !12, file: !3, line: 2, baseType: !5, size: 32)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", file: !3, line: 8, size: 128, elements: !16)
!16 = !{!17, !18}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "pad", scope: !15, file: !3, line: 9, baseType: !9, size: 64)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "q", scope: !15, file: !3, line: 10, baseType: !19, size: 64, offset: 64)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!20 = !{i32 7, !"Dwarf Version", i32 5}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !{i32 1, !"wchar_size", i32 4}
!23 = !{i32 7, !"frame-pointer", i32 2}
!24 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git ca2be81e34a6d87edb8e555dfac94ab68ee20f70)"}
!25 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 14, type: !26, scopeLine: 14, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !29)
!26 = !DISubroutineType(types: !27)
!27 = !{!5, !28}
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!29 = !{!30, !31, !32}
!30 = !DILocalVariable(name: "p", arg: 1, scope: !25, file: !3, line: 14, type: !28)
!31 = !DILocalVariable(name: "q", scope: !25, file: !3, line: 15, type: !19)
!32 = !DILocalVariable(name: "f", scope: !25, file: !3, line: 18, type: !11)
!33 = !DILocation(line: 0, scope: !25)
!34 = !DILocation(line: 15, column: 21, scope: !25)
!35 = !{!36, !37, i64 8}
!36 = !{!"t1", !37, i64 0, !37, i64 8}
!37 = !{!"any pointer", !38, i64 0}
!38 = !{!"omnipotent char", !39, i64 0}
!39 = !{!"Simple C/C++ TBAA"}
!40 = !DILocation(line: 16, column: 7, scope: !41)
!41 = distinct !DILexicalBlock(scope: !25, file: !3, line: 16, column: 7)
!42 = !DILocation(line: 16, column: 7, scope: !25)
!43 = !DILocation(line: 18, column: 21, scope: !25)
!44 = !{!45, !37, i64 8}
!45 = !{!"t2", !37, i64 0, !37, i64 8}
!46 = !DILocation(line: 19, column: 8, scope: !47)
!47 = distinct !DILexicalBlock(scope: !25, file: !3, line: 19, column: 7)
!48 = !DILocation(line: 19, column: 7, scope: !25)
!49 = !DILocation(line: 19, column: 13, scope: !47)
!50 = !{!51, !51, i64 0}
!51 = !{!"int", !38, i64 0}
!52 = !DILocation(line: 19, column: 11, scope: !47)
!53 = !DILocation(line: 21, column: 1, scope: !25)
