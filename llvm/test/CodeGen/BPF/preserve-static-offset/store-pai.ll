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
;    struct buz {
;      int _1;
;      int _2;
;      int _3;
;      union {
;        struct foo a;
;        struct bar b[7];
;      };
;    } __pai __ctx;
;    
;    void quux(struct buz *p) {
;      p->b[5].b = 42;
;    }
;    
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes \
;         -debug-info-kind=limited -o - \
;       | opt -passes=function(sroa) -S -o -

%struct.buz = type { i32, i32, i32, %union.anon }
%union.anon = type { [7 x %struct.bar] }
%struct.bar = type { i32, i32 }

; Function Attrs: nounwind
define dso_local void @quux(ptr noundef %p) #0 !dbg !31 {
entry:
  call void @llvm.dbg.value(metadata ptr %p, metadata !36, metadata !DIExpression()), !dbg !37
  %0 = call ptr @llvm.preserve.static.offset(ptr %p), !dbg !38
  %1 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.buz) %0, i32 3, i32 3), !dbg !38, !llvm.preserve.access.index !4
  %2 = call ptr @llvm.preserve.union.access.index.p0.p0(ptr %1, i32 1), !dbg !38, !llvm.preserve.access.index !3
  %3 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([7 x %struct.bar]) %2, i32 1, i32 5), !dbg !39, !llvm.preserve.access.index !21
  %4 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.bar) %3, i32 1, i32 1), !dbg !40, !llvm.preserve.access.index !22
  store i32 42, ptr %4, align 4, !dbg !41, !tbaa !42
  ret void, !dbg !45
}

; CHECK:      define dso_local void @quux(ptr noundef %[[p:.*]]) {{.*}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.dbg.value
; CHECK-NEXT:   call void (i32, ptr, i1, i8, i8, i8, i1, ...)
; CHECK-SAME:     @llvm.bpf.getelementptr.and.store.i32
; CHECK-SAME:       (i32 42,
; CHECK-SAME:        ptr writeonly elementtype(i8) %[[p]],
; CHECK-SAME:        i1 false, i8 0, i8 1, i8 2, i1 true, i64 immarg 56)
; CHECK-SAME:      #[[v5:.*]], !tbaa
; CHECK-NEXT:   ret void, !dbg
; CHECK-NEXT: }
; CHECK:      attributes #[[v5]] = { memory(argmem: write) }

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.preserve.static.offset(ptr readnone) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32 immarg, i32 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.union.access.index.p0.p0(ptr, i32 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.array.access.index.p0.p0(ptr, i32 immarg, i32 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!28, !29}
!llvm.ident = !{!30}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!2 = !{!3, !21}
!3 = distinct !DICompositeType(tag: DW_TAG_union_type, scope: !4, file: !1, line: 17, size: 448, elements: !11)
!4 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "buz", file: !1, line: 13, size: 544, elements: !5)
!5 = !{!6, !8, !9, !10}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "_1", scope: !4, file: !1, line: 14, baseType: !7, size: 32)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DIDerivedType(tag: DW_TAG_member, name: "_2", scope: !4, file: !1, line: 15, baseType: !7, size: 32, offset: 32)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "_3", scope: !4, file: !1, line: 16, baseType: !7, size: 32, offset: 64)
!10 = !DIDerivedType(tag: DW_TAG_member, scope: !4, file: !1, line: 17, baseType: !3, size: 448, offset: 96)
!11 = !{!12, !20}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !3, file: !1, line: 18, baseType: !13, size: 80)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !1, line: 4, size: 80, elements: !14)
!14 = !{!15}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !13, file: !1, line: 5, baseType: !16, size: 80)
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !17, size: 80, elements: !18)
!17 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!18 = !{!19}
!19 = !DISubrange(count: 10)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !3, file: !1, line: 19, baseType: !21, size: 448)
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !22, size: 448, elements: !26)
!22 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !1, line: 8, size: 64, elements: !23)
!23 = !{!24, !25}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !22, file: !1, line: 9, baseType: !7, size: 32)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !22, file: !1, line: 10, baseType: !7, size: 32, offset: 32)
!26 = !{!27}
!27 = !DISubrange(count: 7)
!28 = !{i32 2, !"Debug Info Version", i32 3}
!29 = !{i32 1, !"wchar_size", i32 4}
!30 = !{!"clang"}
!31 = distinct !DISubprogram(name: "quux", scope: !1, file: !1, line: 23, type: !32, scopeLine: 23, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !35)
!32 = !DISubroutineType(types: !33)
!33 = !{null, !34}
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)
!35 = !{!36}
!36 = !DILocalVariable(name: "p", arg: 1, scope: !31, file: !1, line: 23, type: !34)
!37 = !DILocation(line: 0, scope: !31)
!38 = !DILocation(line: 24, column: 6, scope: !31)
!39 = !DILocation(line: 24, column: 3, scope: !31)
!40 = !DILocation(line: 24, column: 11, scope: !31)
!41 = !DILocation(line: 24, column: 13, scope: !31)
!42 = !{!43, !43, i64 0}
!43 = !{!"omnipotent char", !44, i64 0}
!44 = !{!"Simple C/C++ TBAA"}
!45 = !DILocation(line: 25, column: 1, scope: !31)
