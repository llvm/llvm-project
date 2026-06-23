; RUN: llc -mtriple=bpfel -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK-BTF %s
; RUN: llc -mtriple=bpfeb -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK-BTF %s
;
; Source:
;  #define __tag1 __attribute__((btf_type_tag("tag1")))
;  #define __tag2 __attribute__((btf_type_tag("tag2")))
;
;  struct bar {
;    int c;
;  };
;  typedef struct bar __tag1 __tag2 bar_t1;
;  typedef const struct bar __tag1 __tag2 bar_t2;
;  typedef volatile struct bar __tag1 __tag2 bar_t3;
;  typedef volatile struct bar * __tag1 __tag2 bar_t4;
;
;  typedef const int __tag1 __tag2 int_v;
;
;  int use(bar_t1 *v1, bar_t2 *v2, bar_t3 *v3, bar_t4 v4, int_v v5)
;  {
;    return v1->c + v2->c + v3->c + v4.c + v5;
;  }
; Compilation flag:
;  clang -target bpf -O2 -g -S -emit-llvm t.c

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local i32 @use(ptr nofree noundef readonly captures(none) %0, ptr nofree noundef readonly captures(none) %1, ptr nofree noundef captures(address) %2, ptr nofree noundef captures(address) %3, i32 noundef %4) local_unnamed_addr #0 !dbg !11 {
    #dbg_value(ptr %0, !34, !DIExpression(), !39)
    #dbg_value(ptr %1, !35, !DIExpression(), !39)
    #dbg_value(ptr %2, !36, !DIExpression(), !39)
    #dbg_value(ptr %3, !37, !DIExpression(), !39)
    #dbg_value(i32 %4, !38, !DIExpression(), !39)
  %6 = load i32, ptr %0, align 4, !dbg !40, !tbaa !41
  %7 = load i32, ptr %1, align 4, !dbg !43, !tbaa !41
  %8 = load volatile i32, ptr %2, align 4, !dbg !44, !tbaa !41
  %9 = load volatile i32, ptr %3, align 4, !dbg !45, !tbaa !41
  %10 = add i32 %6, %4, !dbg !46
  %11 = add i32 %10, %7, !dbg !47
  %12 = add i32 %11, %8, !dbg !48
  %13 = add i32 %12, %9, !dbg !49
  ret i32 %13, !dbg !50
}

; CHECK-BTF: [1] PTR '(anon)' type_id=4
; CHECK-BTF: [2] TYPE_TAG 'tag1' type_id=5
; CHECK-BTF: [3] TYPE_TAG 'tag2' type_id=2
; CHECK-BTF: [4] TYPEDEF 'bar_t1' type_id=3
; CHECK-BTF: [5] STRUCT 'bar' size=4 vlen=1
; CHECK-BTF: 	'c' type_id=6 bits_offset=0
; CHECK-BTF: [6] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-BTF: [7] PTR '(anon)' type_id=10
; CHECK-BTF: [8] TYPE_TAG 'tag1' type_id=11
; CHECK-BTF: [9] TYPE_TAG 'tag2' type_id=8
; CHECK-BTF: [10] TYPEDEF 'bar_t2' type_id=9
; CHECK-BTF: [11] CONST '(anon)' type_id=5
; CHECK-BTF: [12] PTR '(anon)' type_id=15
; CHECK-BTF: [13] TYPE_TAG 'tag1' type_id=16
; CHECK-BTF: [14] TYPE_TAG 'tag2' type_id=13
; CHECK-BTF: [15] TYPEDEF 'bar_t3' type_id=14
; CHECK-BTF: [16] VOLATILE '(anon)' type_id=5
; CHECK-BTF: [17] TYPE_TAG 'tag1' type_id=20
; CHECK-BTF: [18] TYPE_TAG 'tag2' type_id=17
; CHECK-BTF: [19] TYPEDEF 'bar_t4' type_id=18
; CHECK-BTF: [20] PTR '(anon)' type_id=16
; CHECK-BTF: [21] TYPE_TAG 'tag1' type_id=24
; CHECK-BTF: [22] TYPE_TAG 'tag2' type_id=21
; CHECK-BTF: [23] TYPEDEF 'int_v' type_id=22
; CHECK-BTF: [24] CONST '(anon)' type_id=6
; CHECK-BTF: [25] FUNC_PROTO '(anon)' ret_type_id=6 vlen=5
; CHECK-BTF: 	'v1' type_id=1
; CHECK-BTF: 	'v2' type_id=7
; CHECK-BTF: 	'v3' type_id=12
; CHECK-BTF: 	'v4' type_id=19
; CHECK-BTF: 	'v5' type_id=23
; CHECK-BTF: [26] FUNC 'use' type_id=25 linkage=global

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}
!llvm.errno.tbaa = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 23.0.0git (git@github.com:yonghong-song/llvm-project.git 19865cd9403e3a25d2ab36f87d28f3d212342a7e)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/tmp/home/yhs/tests/typedef-tag-noptr", checksumkind: CSK_MD5, checksum: "837ae8c69aac4a5303c9a401fdefea04")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!6 = !{!"clang version 23.0.0git (git@github.com:yonghong-song/llvm-project.git 19865cd9403e3a25d2ab36f87d28f3d212342a7e)"}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = distinct !DISubprogram(name: "use", scope: !1, file: !1, line: 14, type: !12, scopeLine: 15, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !33, keyInstructions: true)
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !15, !23, !26, !29, !31}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "bar_t1", file: !1, line: 7, baseType: !17, annotations: !20)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !1, line: 4, size: 32, elements: !18)
!18 = !{!19}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !17, file: !1, line: 5, baseType: !14, size: 32)
!20 = !{!21, !22}
!21 = !{!"btf_type_tag", !"tag1"}
!22 = !{!"btf_type_tag", !"tag2"}
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !24, size: 64)
!24 = !DIDerivedType(tag: DW_TAG_typedef, name: "bar_t2", file: !1, line: 8, baseType: !25, annotations: !20)
!25 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !17)
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !27, size: 64)
!27 = !DIDerivedType(tag: DW_TAG_typedef, name: "bar_t3", file: !1, line: 9, baseType: !28, annotations: !20)
!28 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !17)
!29 = !DIDerivedType(tag: DW_TAG_typedef, name: "bar_t4", file: !1, line: 10, baseType: !30, annotations: !20)
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !28, size: 64)
!31 = !DIDerivedType(tag: DW_TAG_typedef, name: "int_v", file: !1, line: 12, baseType: !32, annotations: !20)
!32 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !14)
!33 = !{!34, !35, !36, !37, !38}
!34 = !DILocalVariable(name: "v1", arg: 1, scope: !11, file: !1, line: 14, type: !15)
!35 = !DILocalVariable(name: "v2", arg: 2, scope: !11, file: !1, line: 14, type: !23)
!36 = !DILocalVariable(name: "v3", arg: 3, scope: !11, file: !1, line: 14, type: !26)
!37 = !DILocalVariable(name: "v4", arg: 4, scope: !11, file: !1, line: 14, type: !29)
!38 = !DILocalVariable(name: "v5", arg: 5, scope: !11, file: !1, line: 14, type: !31)
!39 = !DILocation(line: 0, scope: !11)
!40 = !DILocation(line: 16, column: 20, scope: !11)
!41 = !{!42, !8, i64 0}
!42 = !{!"bar", !8, i64 0}
!43 = !DILocation(line: 16, column: 28, scope: !11)
!44 = !DILocation(line: 16, column: 36, scope: !11)
!45 = !DILocation(line: 16, column: 44, scope: !11)
!46 = !DILocation(line: 16, column: 22, scope: !11)
!47 = !DILocation(line: 16, column: 30, scope: !11)
!48 = !DILocation(line: 16, column: 38, scope: !11)
!49 = !DILocation(line: 16, column: 46, scope: !11, atomGroup: 1, atomRank: 2)
!50 = !DILocation(line: 16, column: 9, scope: !11, atomGroup: 1, atomRank: 1)
