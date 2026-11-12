; REQUIRES: bpf-registered-target

;; Verify that llvm-objdump can use .BTF.ext to show CO-RE relocation data.

; RUN: llc --mtriple bpfel %s --filetype=obj -o - | \
; RUN:   llvm-objdump --no-addresses --no-show-raw-insn -dr - | \
; RUN:   FileCheck %s

; RUN: llc --mtriple bpfeb %s --filetype=obj -o - | \
; RUN:   llvm-objdump --no-addresses --no-show-raw-insn -dr - | \
; RUN:   FileCheck %s

;; Input generated from the following C code:
;;
;;  #define __pai __attribute__((preserve_access_index))
;;
;;  struct buz {
;;    int a;
;;    int b;
;;  } __pai;
;;
;;  struct foo {
;;    int :4;
;;    int i;
;;    struct buz k[10];
;;  } __pai;
;;
;;  struct bar {
;;    struct foo f;
;;  } __pai;
;;
;;  void * volatile g;
;;
;;  void root(void) {
;;    struct bar *bar = (void *)0;
;;    g = &bar->f;
;;    g = &bar->f.i;
;;    g = &bar->f.k;
;;    g = &bar->f.k[7].a;
;;    g = &bar->f.k[7].b;
;;    g = &bar[1].f.k[7].b;
;;  }
;;
;; Using the following command:
;;
;;  clang --target=bpf -g -O2 -emit-llvm -S t.c

; CHECK: CO-RE <byte_off> [[[#bar:]]] struct bar::f (0:0)
; CHECK: CO-RE <byte_off> [[[#bar]]] struct bar::f.i (0:0:0)
; CHECK: CO-RE <byte_off> [[[#bar]]] struct bar::f.k (0:0:1)
; CHECK: CO-RE <byte_off> [[[#bar]]] struct bar::f.k[7].a (0:0:1:7:0)
; CHECK: CO-RE <byte_off> [[[#bar]]] struct bar::f.k[7].b (0:0:1:7:1)
; CHECK: CO-RE <byte_off> [[[#bar]]] struct bar::[1].f.k[7].b (1:0:1:7:1)

@g = dso_local global ptr null, align 8, !dbg !0
@"llvm.bar:0:0$0:0" = external global i64, !llvm.preserve.access.index !14 #0
@"llvm.bar:0:8$0:0:1" = external global i64, !llvm.preserve.access.index !14 #0
@"llvm.bar:0:4$0:0:0" = external global i64, !llvm.preserve.access.index !14 #0
@"llvm.bar:0:64$0:0:1:7:0" = external global i64, !llvm.preserve.access.index !14 #0
@"llvm.bar:0:68$0:0:1:7:1" = external global i64, !llvm.preserve.access.index !14 #0
@"llvm.bar:0:156$1:0:1:7:1" = external global i64, !llvm.preserve.access.index !14 #0

; Function Attrs: nofree nounwind memory(readwrite, argmem: none)
define dso_local void @root() local_unnamed_addr #1 !dbg !29 {
entry:
  call void @llvm.dbg.value(metadata ptr null, metadata !33, metadata !DIExpression()), !dbg !34
  %0 = load i64, ptr @"llvm.bar:0:0$0:0", align 8
  %1 = getelementptr i8, ptr null, i64 %0
  %2 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 0, ptr %1)
  store volatile ptr %2, ptr @g, align 8, !dbg !35, !tbaa !36
  %3 = load i64, ptr @"llvm.bar:0:4$0:0:0", align 8
  %4 = getelementptr i8, ptr null, i64 %3
  %5 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 2, ptr %4)
  store volatile ptr %5, ptr @g, align 8, !dbg !40, !tbaa !36
  %6 = load i64, ptr @"llvm.bar:0:8$0:0:1", align 8
  %7 = getelementptr i8, ptr null, i64 %6
  %8 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 1, ptr %7)
  store volatile ptr %8, ptr @g, align 8, !dbg !41, !tbaa !36
  %9 = load i64, ptr @"llvm.bar:0:64$0:0:1:7:0", align 8
  %10 = getelementptr i8, ptr null, i64 %9
  %11 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 3, ptr %10)
  store volatile ptr %11, ptr @g, align 8, !dbg !42, !tbaa !36
  %12 = load i64, ptr @"llvm.bar:0:68$0:0:1:7:1", align 8
  %13 = getelementptr i8, ptr null, i64 %12
  %14 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 4, ptr %13)
  store volatile ptr %14, ptr @g, align 8, !dbg !43, !tbaa !36
  %15 = load i64, ptr @"llvm.bar:0:156$1:0:1:7:1", align 8
  %16 = getelementptr i8, ptr null, i64 %15
  %17 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 5, ptr %16)
  store volatile ptr %17, ptr @g, align 8, !dbg !44, !tbaa !36
  ret void, !dbg !45
}

; Function Attrs: nofree nosync nounwind memory(none)
declare ptr @llvm.bpf.passthrough.p0.p0(i32, ptr) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { "btf_ama" }
attributes #1 = { nofree nounwind memory(readwrite, argmem: none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nofree nosync nounwind memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!24, !25, !26, !27}
!llvm.ident = !{!28}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 18, type: !22, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 17.0.0 (/home/eddy/work/llvm-project/clang 2f8c5c0afd1d79a771dd74c8fb1e5bbae6d04eb7)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4, globals: !21, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/eddy/work/tmp", checksumkind: CSK_MD5, checksum: "f7c638151153f385e69bef98e88c80ef")
!4 = !{!5, !13}
!5 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 640, elements: !11)
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "buz", file: !3, line: 3, size: 64, elements: !7)
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !6, file: !3, line: 4, baseType: !9, size: 32)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !6, file: !3, line: 5, baseType: !9, size: 32, offset: 32)
!11 = !{!12}
!12 = !DISubrange(count: 10)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !3, line: 14, size: 704, elements: !15)
!15 = !{!16}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !14, file: !3, line: 15, baseType: !17, size: 704)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !3, line: 8, size: 704, elements: !18)
!18 = !{!19, !20}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !17, file: !3, line: 10, baseType: !9, size: 32, offset: 32)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "k", scope: !17, file: !3, line: 11, baseType: !5, size: 640, offset: 64)
!21 = !{!0}
!22 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !23)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!24 = !{i32 7, !"Dwarf Version", i32 5}
!25 = !{i32 2, !"Debug Info Version", i32 3}
!26 = !{i32 1, !"wchar_size", i32 4}
!27 = !{i32 7, !"frame-pointer", i32 2}
!28 = !{!"clang version 17.0.0 (/home/eddy/work/llvm-project/clang 2f8c5c0afd1d79a771dd74c8fb1e5bbae6d04eb7)"}
!29 = distinct !DISubprogram(name: "root", scope: !3, file: !3, line: 20, type: !30, scopeLine: 20, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !32)
!30 = !DISubroutineType(types: !31)
!31 = !{null}
!32 = !{!33}
!33 = !DILocalVariable(name: "bar", scope: !29, file: !3, line: 21, type: !13)
!34 = !DILocation(line: 0, scope: !29)
!35 = !DILocation(line: 22, column: 5, scope: !29)
!36 = !{!37, !37, i64 0}
!37 = !{!"any pointer", !38, i64 0}
!38 = !{!"omnipotent char", !39, i64 0}
!39 = !{!"Simple C/C++ TBAA"}
!40 = !DILocation(line: 23, column: 5, scope: !29)
!41 = !DILocation(line: 24, column: 5, scope: !29)
!42 = !DILocation(line: 25, column: 5, scope: !29)
!43 = !DILocation(line: 26, column: 5, scope: !29)
!44 = !DILocation(line: 27, column: 5, scope: !29)
!45 = !DILocation(line: 28, column: 1, scope: !29)
