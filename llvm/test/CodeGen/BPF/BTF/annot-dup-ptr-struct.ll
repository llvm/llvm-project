; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK-BTF %s
; Source:
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;
;   struct t1 {
;     int a;
;   };
;   struct t2 {
;     struct t1 *p1;
;     struct t1 __tag1 *p2;
;     int b;
;   };
;   int foo(struct t2 *arg) {
;     return arg->b;
;   }
; Compilation flags:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
define dso_local i32 @foo(ptr nocapture noundef readonly %0) local_unnamed_addr #0 !dbg !8 {
    #dbg_value(ptr %0, !26, !DIExpression(), !27)
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 16, !dbg !28
  %3 = load i32, ptr %2, align 8, !dbg !28, !tbaa !29
  ret i32 %3, !dbg !36
}

; CHECK-BTF: [1] PTR '(anon)' type_id=2
; CHECK-BTF: [2] STRUCT 't2' size=24 vlen=3
; CHECK-BTF:     'p1' type_id=3 bits_offset=0
; CHECK-BTF:     'p2' type_id=5 bits_offset=64
; CHECK-BTF:     'b' type_id=7 bits_offset=128
; CHECK-BTF: [3] PTR '(anon)' type_id=6
; CHECK-BTF: [4] TYPE_TAG 'tag1' type_id=6
; CHECK-BTF: [5] PTR '(anon)' type_id=4
; CHECK-BTF: [6] STRUCT 't1' size=4 vlen=1
; CHECK-BTF:     'a' type_id=7 bits_offset=0
; CHECK-BTF: [7] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-BTF: [8] FUNC_PROTO '(anon)' ret_type_id=7 vlen=1
; CHECK-BTF:     'arg' type_id=1
; CHECK-BTF: [9] FUNC 'foo' type_id=8 linkage=global

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 20.1.6 (https://github.com/llvm/llvm-project.git aa804fd3e624cb92c6e7665182504c6049387f35)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/tmp/home/yhs/tmp3/tests", checksumkind: CSK_MD5, checksum: "73ef5b93654cc4e2667f383d5c3a9cd3")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!7 = !{!"clang version 20.1.6 (https://github.com/llvm/llvm-project.git aa804fd3e624cb92c6e7665182504c6049387f35)"}
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 11, type: !9, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !25)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !12}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t2", file: !1, line: 6, size: 192, elements: !14)
!14 = !{!15, !20, !24}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "p1", scope: !13, file: !1, line: 7, baseType: !16, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", file: !1, line: 3, size: 32, elements: !18)
!18 = !{!19}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !17, file: !1, line: 4, baseType: !11, size: 32)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "p2", scope: !13, file: !1, line: 8, baseType: !21, size: 64, offset: 64)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64, annotations: !22)
!22 = !{!23}
!23 = !{!"btf_type_tag", !"tag1"}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !13, file: !1, line: 9, baseType: !11, size: 32, offset: 128)
!25 = !{!26}
!26 = !DILocalVariable(name: "arg", arg: 1, scope: !8, file: !1, line: 11, type: !12)
!27 = !DILocation(line: 0, scope: !8)
!28 = !DILocation(line: 12, column: 15, scope: !8)
!29 = !{!30, !35, i64 16}
!30 = !{!"t2", !31, i64 0, !31, i64 8, !35, i64 16}
!31 = !{!"p1 _ZTS2t1", !32, i64 0}
!32 = !{!"any pointer", !33, i64 0}
!33 = !{!"omnipotent char", !34, i64 0}
!34 = !{!"Simple C/C++ TBAA"}
!35 = !{!"int", !33, i64 0}
!36 = !DILocation(line: 12, column: 3, scope: !8)
