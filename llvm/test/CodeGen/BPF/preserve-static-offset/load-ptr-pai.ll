; RUN: opt -passes=bpf-preserve-static-offset -mtriple=bpf-pc-linux -S -o - %s | FileCheck %s
;
; Source:
;    #define __ctx __attribute__((preserve_static_offset))
;    #define __pai __attribute__((preserve_access_index))
;    
;    struct bar {
;      int a;
;      int b;
;    } __pai;
;    
;    struct buz {
;      int _1;
;      struct bar *b;
;    } __pai __ctx;
;    
;    void foo(struct buz *p) {
;      p->b->b = 42;
;    }
;    
;
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes \
;         -debug-info-kind=limited -o - \
;       | opt -passes=function(sroa) -S -o -

%struct.buz = type { i32, ptr }
%struct.bar = type { i32, i32 }

; Function Attrs: nounwind
define dso_local void @foo(ptr noundef %p) #0 !dbg !5 {
entry:
  call void @llvm.dbg.value(metadata ptr %p, metadata !20, metadata !DIExpression()), !dbg !21
  %0 = call ptr @llvm.preserve.static.offset(ptr %p), !dbg !22
  %1 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.buz) %0, i32 1, i32 1), !dbg !22, !llvm.preserve.access.index !9
  %2 = load ptr, ptr %1, align 8, !dbg !22, !tbaa !23
  %3 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.bar) %2, i32 1, i32 1), !dbg !29, !llvm.preserve.access.index !15
  store i32 42, ptr %3, align 4, !dbg !30, !tbaa !31
  ret void, !dbg !33
}

; CHECK:      define dso_local void @foo(ptr noundef %[[p:.*]]) {{.*}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   #dbg_value
; CHECK-NEXT:   %[[v5:.*]] = call ptr (ptr, i1, i8, i8, i8, i1, ...)
; CHECK-SAME:     @llvm.bpf.getelementptr.and.load.p0
; CHECK-SAME:       (ptr readonly elementtype(%struct.buz) %[[p]],
; CHECK-SAME:        i1 false, i8 0, i8 1, i8 3, i1 true, i32 immarg 0, i32 immarg 1)
; CHECK-SAME:      #[[v6:.*]], !tbaa
; CHECK-NEXT:   %[[v8:.*]] =
; CHECK-SAME:     call ptr @llvm.preserve.struct.access.index.p0.p0
; CHECK-SAME:       (ptr elementtype(%struct.bar) %[[v5]], i32 1, i32 1),
; CHECK-SAME:        !dbg ![[#]], !llvm.preserve.access.index ![[#]]
; CHECK-NEXT:   store i32 42, ptr %[[v8]], align 4, !dbg ![[#]], !tbaa
; CHECK-NEXT:   ret void, !dbg
; CHECK-NEXT: }

; CHECK     : attributes #[[v6]] = { memory(argmem: read) }


; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.preserve.static.offset(ptr readnone) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32 immarg, i32 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{!"clang"}
!5 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 14, type: !6, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !19)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "buz", file: !1, line: 9, size: 128, elements: !10)
!10 = !{!11, !13}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "_1", scope: !9, file: !1, line: 10, baseType: !12, size: 32)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !9, file: !1, line: 11, baseType: !14, size: 64, offset: 64)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !1, line: 4, size: 64, elements: !16)
!16 = !{!17, !18}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !15, file: !1, line: 5, baseType: !12, size: 32)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !15, file: !1, line: 6, baseType: !12, size: 32, offset: 32)
!19 = !{!20}
!20 = !DILocalVariable(name: "p", arg: 1, scope: !5, file: !1, line: 14, type: !8)
!21 = !DILocation(line: 0, scope: !5)
!22 = !DILocation(line: 15, column: 6, scope: !5)
!23 = !{!24, !28, i64 8}
!24 = !{!"buz", !25, i64 0, !28, i64 8}
!25 = !{!"int", !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C/C++ TBAA"}
!28 = !{!"any pointer", !26, i64 0}
!29 = !DILocation(line: 15, column: 9, scope: !5)
!30 = !DILocation(line: 15, column: 11, scope: !5)
!31 = !{!32, !25, i64 4}
!32 = !{!"bar", !25, i64 0, !25, i64 4}
!33 = !DILocation(line: 16, column: 1, scope: !5)
