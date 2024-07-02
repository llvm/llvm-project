; RUN: llc -march=bpfel -filetype=obj -o - %s | llvm-objdump -dr --no-show-raw-insn - | FileCheck %s
; RUN: llc -march=bpfeb -filetype=obj -o - %s | llvm-objdump -dr --no-show-raw-insn - | FileCheck %s
;
; Source:
;
;   #define __pai  __attribute__((preserve_access_index));
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;
;   struct alpha {
;     int zulu;
;   } __pai;
;
;   struct bravo {
;     struct alpha __tag1 *yankee;
;   } __pai;
;
;   int func(struct bravo *xray) {
;     return xray->yankee->zulu;
;   }
;
; Compilation command:
; 
;   cat test.c | clang -mllvm -btf-type-tag-v2 -x c -target bpf -O2 -g -emit-llvm -S - -o -
;
; The relocation entry for zulu should point to STRUCT 'alpha',
; not TYPE_TAG 'tag1' -> STRUCT 'alpha'.

@"llvm.alpha:0:0$0:0" = external global i64, !llvm.preserve.access.index !0 #0
@"llvm.bravo:0:0$0:0" = external global i64, !llvm.preserve.access.index !8 #0

; Function Attrs: nofree nosync nounwind memory(read, inaccessiblemem: none)
define dso_local i32 @func(ptr noundef readonly %xray) local_unnamed_addr #1 !dbg !20 {
entry:
  tail call void @llvm.dbg.value(metadata ptr %xray, metadata !25, metadata !DIExpression()), !dbg !26
  %0 = load i64, ptr @"llvm.bravo:0:0$0:0", align 8
  %1 = getelementptr i8, ptr %xray, i64 %0
  %2 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 1, ptr %1)
  %3 = load ptr, ptr %2, align 8, !dbg !27, !tbaa !28
  %4 = load i64, ptr @"llvm.alpha:0:0$0:0", align 8
  %5 = getelementptr i8, ptr %3, i64 %4
  %6 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 0, ptr %5)
  %7 = load i32, ptr %6, align 4, !dbg !33, !tbaa !34
  ret i32 %7, !dbg !37
}

; CHECK:      r[[#]] = *(u64 *)(r[[#]] + 0x0)
; CHECK-NEXT:   CO-RE <byte_off> [[[#]]] struct bravo::yankee (0:0)
; CHECK-NEXT: r[[#]] = *(u32 *)(r[[#]] + 0x0)
; CHECK-NEXT:   CO-RE <byte_off> [[[#]]] struct alpha::zulu (0:0)

; Function Attrs: nofree nosync nounwind memory(none)
declare ptr @llvm.bpf.passthrough.p0.p0(i32, ptr) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { "btf_ama" }
attributes #1 = { nofree nosync nounwind memory(read, inaccessiblemem: none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nofree nosync nounwind memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!12}
!llvm.module.flags = !{!14, !15, !16, !17, !18}
!llvm.ident = !{!19}

!0 = !DICompositeType(tag: DW_TAG_structure_type, name: "alpha", file: !1, line: 4, size: 32, elements: !2, annotations: !6)
!1 = !DIFile(filename: "some-file.c", directory: "/some/dir")
!2 = !{!3}
!3 = !DIDerivedType(tag: DW_TAG_member, name: "zulu", scope: !4, file: !1, line: 5, baseType: !5, size: 32)
!4 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "alpha", file: !1, line: 4, size: 32, elements: !2)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{!7}
!7 = !{!"btf:type_tag", !"tag1"}
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bravo", file: !1, line: 8, size: 64, elements: !9)
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "yankee", scope: !8, file: !1, line: 9, baseType: !11, size: 64)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !0, size: 64)
!12 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang some version", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !13, splitDebugInlining: false, nameTableKind: None)
!13 = !{!0}
!14 = !{i32 7, !"Dwarf Version", i32 5}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{i32 7, !"frame-pointer", i32 2}
!18 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!19 = !{!"clang, some version"}
!20 = distinct !DISubprogram(name: "func", scope: !1, file: !1, line: 12, type: !21, scopeLine: 12, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !12, retainedNodes: !24)
!21 = !DISubroutineType(types: !22)
!22 = !{!5, !23}
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!24 = !{!25}
!25 = !DILocalVariable(name: "xray", arg: 1, scope: !20, file: !1, line: 12, type: !23)
!26 = !DILocation(line: 0, scope: !20)
!27 = !DILocation(line: 13, column: 16, scope: !20)
!28 = !{!29, !30, i64 0}
!29 = !{!"bravo", !30, i64 0}
!30 = !{!"any pointer", !31, i64 0}
!31 = !{!"omnipotent char", !32, i64 0}
!32 = !{!"Simple C/C++ TBAA"}
!33 = !DILocation(line: 13, column: 24, scope: !20)
!34 = !{!35, !36, i64 0}
!35 = !{!"alpha", !36, i64 0}
!36 = !{!"int", !31, i64 0}
!37 = !DILocation(line: 13, column: 3, scope: !20)
