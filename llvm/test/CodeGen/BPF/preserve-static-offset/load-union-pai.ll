; RUN: opt -passes=bpf-preserve-static-offset -mtriple=bpf-pc-linux -S -o - %s | FileCheck %s
;
; Source:
;    #define __ctx __attribute__((preserve_static_offset))
;    #define __pai __attribute__((preserve_access_index))
;    
;    struct foo {
;      char a[10];
;    } __pai;
;    
;    struct bar {
;      int a;
;      int b;
;    } __pai;
;    
;    union buz {
;      struct foo a;
;      struct bar b;
;    } __pai __ctx;
;    
;    int quux(union buz *p) {
;      return p->b.b;
;    }
;
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes -debug-info-kind=limited -o - \
;       | opt -passes=function(sroa) -S -o -

%struct.bar = type { i32, i32 }

; Function Attrs: nounwind
define dso_local i32 @quux(ptr noundef %p) #0 !dbg !5 {
entry:
  call void @llvm.dbg.value(metadata ptr %p, metadata !26, metadata !DIExpression()), !dbg !27
  %0 = call ptr @llvm.preserve.static.offset(ptr %p), !dbg !28
  %1 = call ptr @llvm.preserve.union.access.index.p0.p0(ptr %0, i32 1), !dbg !28, !llvm.preserve.access.index !10
  %2 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.bar) %1, i32 1, i32 1), !dbg !29, !llvm.preserve.access.index !21
  %3 = load i32, ptr %2, align 4, !dbg !29, !tbaa !30
  ret i32 %3, !dbg !33
}

; CHECK:      define dso_local i32 @quux(ptr noundef %[[p:.*]]) {{.*}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.dbg.value
; CHECK-NEXT:   %[[v5:.*]] = call i32 (ptr, i1, i8, i8, i8, i1, ...)
; CHECK-SAME:     @llvm.bpf.getelementptr.and.load.i32
; CHECK-SAME:       (ptr readonly elementtype(%struct.bar) %[[p]],
; CHECK-SAME:        i1 false, i8 0, i8 1, i8 2, i1 true, i32 immarg 0, i32 immarg 1)
; CHECK-SAME:      #[[v6:.*]], !tbaa
; CHECK-NEXT:   ret i32 %[[v5]]
; CHECK-NEXT: }
; CHECK:      attributes #[[v6]] = { memory(argmem: read) }

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.preserve.static.offset(ptr readnone) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.union.access.index.p0.p0(ptr, i32 immarg) #2

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
!5 = distinct !DISubprogram(name: "quux", scope: !1, file: !1, line: 18, type: !6, scopeLine: 18, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !25)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "buz", file: !1, line: 13, size: 96, elements: !11)
!11 = !{!12, !20}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !10, file: !1, line: 14, baseType: !13, size: 80)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !1, line: 4, size: 80, elements: !14)
!14 = !{!15}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !13, file: !1, line: 5, baseType: !16, size: 80)
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !17, size: 80, elements: !18)
!17 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!18 = !{!19}
!19 = !DISubrange(count: 10)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !10, file: !1, line: 15, baseType: !21, size: 64)
!21 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !1, line: 8, size: 64, elements: !22)
!22 = !{!23, !24}
!23 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !21, file: !1, line: 9, baseType: !8, size: 32)
!24 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !21, file: !1, line: 10, baseType: !8, size: 32, offset: 32)
!25 = !{!26}
!26 = !DILocalVariable(name: "p", arg: 1, scope: !5, file: !1, line: 18, type: !9)
!27 = !DILocation(line: 0, scope: !5)
!28 = !DILocation(line: 19, column: 13, scope: !5)
!29 = !DILocation(line: 19, column: 15, scope: !5)
!30 = !{!31, !31, i64 0}
!31 = !{!"omnipotent char", !32, i64 0}
!32 = !{!"Simple C/C++ TBAA"}
!33 = !DILocation(line: 19, column: 3, scope: !5)
