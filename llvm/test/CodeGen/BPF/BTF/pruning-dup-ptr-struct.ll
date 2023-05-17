; RUN: llc -march=bpfel -o - %s | FileCheck %s
; RUN: llc -march=bpfeb -o - %s | FileCheck %s
; Source:
;   struct t1 {
;     int a;
;   };
;   struct t2 {
;     struct t1 *p1;
;     struct t1 *p2;
;     int b;
;   };
;   int foo(struct t2 *arg) {
;     return arg->b;
;   }
; Compilation flags:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

%struct.t2 = type { ptr, ptr, i32 }

; Function Attrs: argmemonly mustprogress nofree norecurse nosync nounwind readonly willreturn
define dso_local i32 @foo(ptr nocapture noundef readonly %arg) local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata ptr %arg, metadata !22, metadata !DIExpression()), !dbg !23
  %b = getelementptr inbounds %struct.t2, ptr %arg, i64 0, i32 2, !dbg !24
  %0 = load i32, ptr %b, align 8, !dbg !24, !tbaa !25
  ret i32 %0, !dbg !31
}

; CHECK:             .long   0                               # BTF_KIND_PTR(id = 1)
; CHECK-NEXT:        .long   33554432                        # 0x2000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1                               # BTF_KIND_STRUCT(id = 2)
; CHECK-NEXT:        .long   67108867                        # 0x4000003
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   0                               # 0x0
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   64                              # 0x40
; CHECK-NEXT:        .long   10
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   128                             # 0x80
; CHECK-NEXT:        .long   0                               # BTF_KIND_PTR(id = 3)
; CHECK-NEXT:        .long   33554432                        # 0x2000000
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   12                              # BTF_KIND_INT(id = 4)
; CHECK-NEXT:        .long   16777216                        # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                        # 0x1000020
; CHECK-NEXT:        .long   0                               # BTF_KIND_FUNC_PROTO(id = 5)
; CHECK-NEXT:        .long   218103809                       # 0xd000001
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   20                              # BTF_KIND_FUNC(id = 6)
; CHECK-NEXT:        .long   201326593                       # 0xc000001
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   60                              # BTF_KIND_FWD(id = 7)
; CHECK-NEXT:        .long   117440512                       # 0x7000000
; CHECK-NEXT:        .long   0

; CHECK:             .ascii  "t2"                            # string offset=1
; CHECK:             .ascii  "p1"                            # string offset=4
; CHECK:             .ascii  "p2"                            # string offset=7
; CHECK:             .byte   98                              # string offset=10
; CHECK:             .ascii  "int"                           # string offset=12
; CHECK:             .ascii  "arg"                           # string offset=16
; CHECK:             .ascii  "foo"                           # string offset=20
; CHECK:             .ascii  "t1"                            # string offset=60

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { argmemonly mustprogress nofree norecurse nosync nounwind readonly willreturn "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 15.0.7 (https://github.com/llvm/llvm-project.git 8dfdcc7b7bf66834a761bd8de445840ef68e4d1a)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t1.c", directory: "/tmp/home/yhs/tmp3/tests", checksumkind: CSK_MD5, checksum: "9a79ff24a6244249e556afd85288af94")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!"clang version 15.0.7 (https://github.com/llvm/llvm-project.git 8dfdcc7b7bf66834a761bd8de445840ef68e4d1a)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 9, type: !8, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !21)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t2", file: !1, line: 4, size: 192, elements: !13)
!13 = !{!14, !19, !20}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "p1", scope: !12, file: !1, line: 5, baseType: !15, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", file: !1, line: 1, size: 32, elements: !17)
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !16, file: !1, line: 2, baseType: !10, size: 32)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "p2", scope: !12, file: !1, line: 6, baseType: !15, size: 64, offset: 64)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !12, file: !1, line: 7, baseType: !10, size: 32, offset: 128)
!21 = !{!22}
!22 = !DILocalVariable(name: "arg", arg: 1, scope: !7, file: !1, line: 9, type: !11)
!23 = !DILocation(line: 0, scope: !7)
!24 = !DILocation(line: 10, column: 15, scope: !7)
!25 = !{!26, !30, i64 16}
!26 = !{!"t2", !27, i64 0, !27, i64 8, !30, i64 16}
!27 = !{!"any pointer", !28, i64 0}
!28 = !{!"omnipotent char", !29, i64 0}
!29 = !{!"Simple C/C++ TBAA"}
!30 = !{!"int", !28, i64 0}
!31 = !DILocation(line: 10, column: 3, scope: !7)
