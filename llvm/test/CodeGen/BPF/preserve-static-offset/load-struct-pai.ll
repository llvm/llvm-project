; RUN: opt -passes=bpf-preserve-static-offset -mtriple=bpf-pc-linux -S -o - %s | FileCheck %s
;
; Source:
;    #define __ctx __attribute__((preserve_static_offset))
;    #define __pai __attribute__((preserve_access_index))
;    
;    struct foo {
;      int a;
;      int b;
;    };
;    
;    struct bar {
;      int _1;
;      int _2;
;      struct foo c;
;    } __pai __ctx;
;    
;    int buz(struct bar *p) {
;      return p->c.b;
;    }
;    
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes \
;         -debug-info-kind=limited -o - \
;       | opt -passes=function(sroa) -S -o -

%struct.bar = type { i32, i32, %struct.foo }
%struct.foo = type { i32, i32 }

; Function Attrs: nounwind
define dso_local i32 @buz(ptr noundef %p) #0 !dbg !5 {
entry:
  call void @llvm.dbg.value(metadata ptr %p, metadata !20, metadata !DIExpression()), !dbg !21
  %0 = call ptr @llvm.preserve.static.offset(ptr %p), !dbg !22
  %1 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.bar) %0, i32 2, i32 2), !dbg !22, !llvm.preserve.access.index !10
  %b = getelementptr inbounds %struct.foo, ptr %1, i32 0, i32 1, !dbg !23
  %2 = load i32, ptr %b, align 4, !dbg !23, !tbaa !24
  ret i32 %2, !dbg !30
}

; CHECK:      define dso_local i32 @buz(ptr noundef %[[p:.*]]) {{.*}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   #dbg_value
; CHECK-NEXT:   %[[b1:.*]] = call i32 (ptr, i1, i8, i8, i8, i1, ...)
; CHECK-SAME:     @llvm.bpf.getelementptr.and.load.i32
; CHECK-SAME:       (ptr readonly elementtype(%struct.bar) %[[p]],
; CHECK-SAME:        i1 false, i8 0, i8 1, i8 2, i1 true, i32 immarg 0, i32 immarg 2, i32 immarg 1)
; CHECK-SAME:      #[[v5:.*]], !tbaa
; CHECK-NEXT:   ret i32 %[[b1]]
; CHECK-NEXT: }

; CHECK:      attributes #[[v5]] = { memory(argmem: read) }


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
!5 = distinct !DISubprogram(name: "buz", scope: !1, file: !1, line: 15, type: !6, scopeLine: 15, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !19)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !1, line: 9, size: 128, elements: !11)
!11 = !{!12, !13, !14}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "_1", scope: !10, file: !1, line: 10, baseType: !8, size: 32)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "_2", scope: !10, file: !1, line: 11, baseType: !8, size: 32, offset: 32)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !10, file: !1, line: 12, baseType: !15, size: 64, offset: 64)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !1, line: 4, size: 64, elements: !16)
!16 = !{!17, !18}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !15, file: !1, line: 5, baseType: !8, size: 32)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !15, file: !1, line: 6, baseType: !8, size: 32, offset: 32)
!19 = !{!20}
!20 = !DILocalVariable(name: "p", arg: 1, scope: !5, file: !1, line: 15, type: !9)
!21 = !DILocation(line: 0, scope: !5)
!22 = !DILocation(line: 16, column: 13, scope: !5)
!23 = !DILocation(line: 16, column: 15, scope: !5)
!24 = !{!25, !26, i64 12}
!25 = !{!"bar", !26, i64 0, !26, i64 4, !29, i64 8}
!26 = !{!"int", !27, i64 0}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C/C++ TBAA"}
!29 = !{!"foo", !26, i64 0, !26, i64 4}
!30 = !DILocation(line: 16, column: 3, scope: !5)
