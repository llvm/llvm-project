; RUN: llc -march=bpfel -filetype=obj -o - %s \
; RUN: | llvm-objcopy --dump-section .BTF=- - | %python %S/print_btf.py - | FileCheck %s
; RUN: llc -march=bpfeb -filetype=obj -o - %s \
; RUN: | llvm-objcopy --dump-section .BTF=- - | %python %S/print_btf.py - | FileCheck %s
;
; Source:
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;   #define __tag2 __attribute__((btf_type_tag("tag2")))
;   
;   enum foo { FOO };
;   
;   struct bar {
;     enum foo __tag1 aa;
;     enum foo __tag2 bb;
;     enum foo cc;
;   };
;   
;   void root(struct bar *bar) {}
;
; Compilation flag:
;   clang -mllvm -btf-type-tag-v2 -S -g -emit-llvm test.c -o test.ll

; Check that enum entry is not duplicated in BTF despite duplication in DWARF
; (entries in DWARF are duplicated because of the presence of type tags).

; CHECK:      [[[#]]] STRUCT 'bar' size=12 vlen=3
; CHECK-NEXT: 	'aa' type_id=[[#tag1:]] bits_offset=0
; CHECK-NEXT: 	'bb' type_id=[[#tag2:]] bits_offset=32
; CHECK-NEXT: 	'cc' type_id=[[#foo:]] bits_offset=64
; CHECK-NEXT: [[[#foo]]] ENUM 'foo' encoding=UNSIGNED size=4 vlen=1
; CHECK-NEXT: 	'FOO' val=0
; CHECK-NEXT: [[[#tag1]]] TYPE_TAG 'tag1' type_id=4
; CHECK-NEXT: [[[#tag2]]] TYPE_TAG 'tag2' type_id=4

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @root(ptr noundef %bar) #0 !dbg !15 {
entry:
  %bar.addr = alloca ptr, align 8
  store ptr %bar, ptr %bar.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %bar.addr, metadata !31, metadata !DIExpression()), !dbg !32
  ret void, !dbg !33
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang, some version", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "foo", file: !1, line: 4, baseType: !4, size: 32, elements: !5)
!4 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!5 = !{!6}
!6 = !DIEnumerator(name: "FOO", value: 0)
!7 = !{i32 7, !"Dwarf Version", i32 5}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 8, !"PIC Level", i32 2}
!11 = !{i32 7, !"PIE Level", i32 2}
!12 = !{i32 7, !"uwtable", i32 2}
!13 = !{i32 7, !"frame-pointer", i32 2}
!14 = !{!"clang, some version"}
!15 = distinct !DISubprogram(name: "root", scope: !1, file: !1, line: 12, type: !16, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !30)
!16 = !DISubroutineType(types: !17)
!17 = !{null, !18}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !1, line: 6, size: 96, elements: !20)
!20 = !{!21, !25, !29}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "aa", scope: !19, file: !1, line: 7, baseType: !22, size: 32)
!22 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "foo", file: !1, line: 4, baseType: !4, size: 32, elements: !5, annotations: !23)
!23 = !{!24}
!24 = !{!"btf:type_tag", !"tag1"}
!25 = !DIDerivedType(tag: DW_TAG_member, name: "bb", scope: !19, file: !1, line: 8, baseType: !26, size: 32, offset: 32)
!26 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "foo", file: !1, line: 4, baseType: !4, size: 32, elements: !5, annotations: !27)
!27 = !{!28}
!28 = !{!"btf:type_tag", !"tag2"}
!29 = !DIDerivedType(tag: DW_TAG_member, name: "cc", scope: !19, file: !1, line: 9, baseType: !3, size: 32, offset: 64)
!30 = !{}
!31 = !DILocalVariable(name: "bar", arg: 1, scope: !15, file: !1, line: 12, type: !18)
!32 = !DILocation(line: 12, column: 23, scope: !15)
!33 = !DILocation(line: 12, column: 29, scope: !15)
