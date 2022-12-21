; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; Source code:
;   struct data_t {
;     int d1;
;     int d2;
;   };
;   struct info_t {
;     int pid;
;     int flags;
;   } __attribute__((preserve_access_index));
;
;   extern void output(ptr);
;   void test(struct info_t * args) {
;     int is_mask2 = args->flags & 0x10000;
;     struct data_t data = {};
;
;     data.d1 = is_mask2 ? 2 : args->pid;
;     data.d2 = (is_mask2 || (args->flags & 0x8000)) ? 1 : 2;
;     output(&data);
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

%struct.info_t = type { i32, i32 }
%struct.data_t = type { i32, i32 }

; Function Attrs: nounwind
define dso_local void @test(ptr readonly %args) local_unnamed_addr #0 !dbg !12 {
entry:
  %data = alloca i64, align 8
  call void @llvm.dbg.value(metadata ptr %args, metadata !22, metadata !DIExpression()), !dbg !29
  %0 = tail call ptr @llvm.preserve.struct.access.index.p0.p0.info_ts(ptr elementtype(%struct.info_t) %args, i32 1, i32 1), !dbg !30, !llvm.preserve.access.index !16
  %1 = load i32, ptr %0, align 4, !dbg !30, !tbaa !31
  %and = and i32 %1, 65536, !dbg !36
  call void @llvm.dbg.value(metadata i32 %and, metadata !23, metadata !DIExpression()), !dbg !29
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %data) #5, !dbg !37
  call void @llvm.dbg.declare(metadata ptr %data, metadata !24, metadata !DIExpression()), !dbg !38
  store i64 0, ptr %data, align 8, !dbg !38
  %tobool = icmp eq i32 %and, 0, !dbg !39
  br i1 %tobool, label %cond.false, label %lor.end.critedge, !dbg !39

cond.false:                                       ; preds = %entry
  %2 = tail call ptr @llvm.preserve.struct.access.index.p0.p0.info_ts(ptr elementtype(%struct.info_t) %args, i32 0, i32 0), !dbg !40, !llvm.preserve.access.index !16
  %3 = load i32, ptr %2, align 4, !dbg !40, !tbaa !41
  store i32 %3, ptr %data, align 8, !dbg !43, !tbaa !44
  %4 = load i32, ptr %0, align 4, !dbg !46, !tbaa !31
  %and2 = and i32 %4, 32768, !dbg !47
  %tobool3 = icmp eq i32 %and2, 0, !dbg !48
  %phitmp = select i1 %tobool3, i32 2, i32 1, !dbg !48
  br label %lor.end, !dbg !48

lor.end.critedge:                                 ; preds = %entry
  store i32 2, ptr %data, align 8, !dbg !43, !tbaa !44
  br label %lor.end, !dbg !48

lor.end:                                          ; preds = %lor.end.critedge, %cond.false
  %5 = phi i32 [ %phitmp, %cond.false ], [ 1, %lor.end.critedge ]
  %d2 = getelementptr inbounds %struct.data_t, ptr %data, i64 0, i32 1, !dbg !49
  store i32 %5, ptr %d2, align 4, !dbg !50, !tbaa !51
  call void @output(ptr nonnull %data) #5, !dbg !52
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %data) #5, !dbg !53
  ret void, !dbg !53
}

; CHECK: r[[LOAD1:[0-9]+]] = *(u32 *)(r{{[0-9]+}} + 4)
; CHECK: r[[LOAD1]] &= 65536
; CHECK: r[[LOAD2:[0-9]+]] = *(u32 *)(r{{[0-9]+}} + 4)
; CHECK: r[[LOAD2]] &= 32768

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: nounwind readnone
declare ptr @llvm.preserve.struct.access.index.p0.p0.info_ts(ptr, i32 immarg, i32 immarg) #3

declare !dbg !4 dso_local void @output(ptr) local_unnamed_addr #4

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { argmemonly nounwind willreturn }
attributes #3 = { nounwind readnone }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git 5884aae58f56786475bbc0f13ad8bd35f7f1ce69)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/core")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "output", scope: !1, file: !1, line: 10, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 5884aae58f56786475bbc0f13ad8bd35f7f1ce69)"}
!12 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 11, type: !13, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !21)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !15}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "info_t", file: !1, line: 5, size: 64, elements: !17)
!17 = !{!18, !20}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "pid", scope: !16, file: !1, line: 6, baseType: !19, size: 32)
!19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "flags", scope: !16, file: !1, line: 7, baseType: !19, size: 32, offset: 32)
!21 = !{!22, !23, !24}
!22 = !DILocalVariable(name: "args", arg: 1, scope: !12, file: !1, line: 11, type: !15)
!23 = !DILocalVariable(name: "is_mask2", scope: !12, file: !1, line: 12, type: !19)
!24 = !DILocalVariable(name: "data", scope: !12, file: !1, line: 13, type: !25)
!25 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "data_t", file: !1, line: 1, size: 64, elements: !26)
!26 = !{!27, !28}
!27 = !DIDerivedType(tag: DW_TAG_member, name: "d1", scope: !25, file: !1, line: 2, baseType: !19, size: 32)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "d2", scope: !25, file: !1, line: 3, baseType: !19, size: 32, offset: 32)
!29 = !DILocation(line: 0, scope: !12)
!30 = !DILocation(line: 12, column: 24, scope: !12)
!31 = !{!32, !33, i64 4}
!32 = !{!"info_t", !33, i64 0, !33, i64 4}
!33 = !{!"int", !34, i64 0}
!34 = !{!"omnipotent char", !35, i64 0}
!35 = !{!"Simple C/C++ TBAA"}
!36 = !DILocation(line: 12, column: 30, scope: !12)
!37 = !DILocation(line: 13, column: 3, scope: !12)
!38 = !DILocation(line: 13, column: 17, scope: !12)
!39 = !DILocation(line: 15, column: 13, scope: !12)
!40 = !DILocation(line: 15, column: 34, scope: !12)
!41 = !{!32, !33, i64 0}
!42 = !DILocation(line: 15, column: 8, scope: !12)
!43 = !DILocation(line: 15, column: 11, scope: !12)
!44 = !{!45, !33, i64 0}
!45 = !{!"data_t", !33, i64 0, !33, i64 4}
!46 = !DILocation(line: 16, column: 33, scope: !12)
!47 = !DILocation(line: 16, column: 39, scope: !12)
!48 = !DILocation(line: 16, column: 23, scope: !12)
!49 = !DILocation(line: 16, column: 8, scope: !12)
!50 = !DILocation(line: 16, column: 11, scope: !12)
!51 = !{!45, !33, i64 4}
!52 = !DILocation(line: 17, column: 3, scope: !12)
!53 = !DILocation(line: 18, column: 1, scope: !12)
