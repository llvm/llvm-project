; RUN: llc -march=bpfel -filetype=obj -o - %s \
; RUN: | llvm-objcopy --dump-section .BTF=- - | %python %S/print_btf.py - | FileCheck %s
; RUN: llc -march=bpfeb -filetype=obj -o - %s \
; RUN: | llvm-objcopy --dump-section .BTF=- - | %python %S/print_btf.py - | FileCheck %s
;
; Source:
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;   #define __tag2 __attribute__((btf_type_tag("tag2")))
;   
;   typedef int foo;
;   
;   struct bar {
;     foo __tag1 aa;
;     foo __tag2 bb;
;     foo cc;
;   };
;   
;   void root(struct bar *bar) {}
;
; Compilation flag:
;   clang -mllvm -btf-type-tag-v2 -S -g -emit-llvm test.c -o test.ll

; Check that typedef entry is not duplicated in BTF despite duplication in DWARF
; (entries in DWARF are duplicated because of the presence of type tags).

; CHECK:      [[[#]]] STRUCT 'bar' size=12 vlen=3
; CHECK-NEXT:   'aa' type_id=[[#tag1:]] bits_offset=0
; CHECK-NEXT:   'bb' type_id=[[#tag2:]] bits_offset=32
; CHECK-NEXT:   'cc' type_id=[[#foo:]] bits_offset=64
; CHECK-NEXT: [[[#foo]]] TYPEDEF 'foo' type_id=6
; CHECK-NEXT: [[[#tag1]]] TYPE_TAG 'tag1' type_id=[[#foo]]
; CHECK-NEXT: [[[#]]]     INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-NEXT: [[[#tag2]]] TYPE_TAG 'tag2' type_id=[[#foo]]

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @root(ptr noundef %bar) #0 !dbg !10 {
entry:
  %bar.addr = alloca ptr, align 8
  store ptr %bar, ptr %bar.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %bar.addr, metadata !28, metadata !DIExpression()), !dbg !29
  ret void, !dbg !30
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang, some version", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang, some version"}
!10 = distinct !DISubprogram(name: "root", scope: !1, file: !1, line: 12, type: !11, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !27)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !1, line: 6, size: 96, elements: !15)
!15 = !{!16, !21, !25}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "aa", scope: !14, file: !1, line: 7, baseType: !17, size: 32)
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "foo", file: !1, line: 4, baseType: !18, annotations: !19)
!18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!19 = !{!20}
!20 = !{!"btf:type_tag", !"tag1"}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "bb", scope: !14, file: !1, line: 8, baseType: !22, size: 32, offset: 32)
!22 = !DIDerivedType(tag: DW_TAG_typedef, name: "foo", file: !1, line: 4, baseType: !18, annotations: !23)
!23 = !{!24}
!24 = !{!"btf:type_tag", !"tag2"}
!25 = !DIDerivedType(tag: DW_TAG_member, name: "cc", scope: !14, file: !1, line: 9, baseType: !26, size: 32, offset: 64)
!26 = !DIDerivedType(tag: DW_TAG_typedef, name: "foo", file: !1, line: 4, baseType: !18)
!27 = !{}
!28 = !DILocalVariable(name: "bar", arg: 1, scope: !10, file: !1, line: 12, type: !13)
!29 = !DILocation(line: 12, column: 23, scope: !10)
!30 = !DILocation(line: 12, column: 29, scope: !10)
