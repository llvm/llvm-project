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
;;  struct bar { int a; } __pai;
;;  volatile unsigned long g;
;;  void root(void) {
;;    struct bar *bar = (void *)0;
;;    g = __builtin_preserve_field_info(bar->a, 1);
;;    g = __builtin_preserve_field_info(bar->a, 2);
;;    g = __builtin_preserve_field_info(bar->a, 3);
;;    g = __builtin_preserve_field_info(bar->a, 4);
;;    g = __builtin_preserve_field_info(bar->a, 5);
;;  }
;;
;; Using the following command:
;;
;;  clang --target=bpf -g -O2 -emit-llvm -S t.c

; CHECK: CO-RE <byte_sz> [[[#]]] struct bar::a
; CHECK: CO-RE <field_exists> [[[#]]] struct bar::a
; CHECK: CO-RE <signed> [[[#]]] struct bar::a
; CHECK: CO-RE <lshift_u64> [[[#]]] struct bar::a
; CHECK: CO-RE <rshift_u64> [[[#]]] struct bar::a

@g = dso_local global i64 0, align 8, !dbg !0
@"llvm.bar:1:4$0:0" = external global i32, !llvm.preserve.access.index !7 #0
@"llvm.bar:2:1$0:0" = external global i32, !llvm.preserve.access.index !7 #0
@"llvm.bar:3:1$0:0" = external global i32, !llvm.preserve.access.index !7 #0
@"llvm.bar:4:32$0:0" = external global i32, !llvm.preserve.access.index !7 #0
@"llvm.bar:5:32$0:0" = external global i32, !llvm.preserve.access.index !7 #0

; Function Attrs: nofree nounwind memory(readwrite, argmem: none)
define dso_local void @root() local_unnamed_addr #1 !dbg !16 {
entry:
  call void @llvm.dbg.value(metadata ptr null, metadata !20, metadata !DIExpression()), !dbg !22
  %0 = load i32, ptr @"llvm.bar:1:4$0:0", align 4
  %1 = tail call i32 @llvm.bpf.passthrough.i32.i32(i32 0, i32 %0)
  %conv = zext i32 %1 to i64, !dbg !23
  store volatile i64 %conv, ptr @g, align 8, !dbg !24, !tbaa !25
  %2 = load i32, ptr @"llvm.bar:2:1$0:0", align 4
  %3 = tail call i32 @llvm.bpf.passthrough.i32.i32(i32 1, i32 %2)
  %conv1 = zext i32 %3 to i64, !dbg !29
  store volatile i64 %conv1, ptr @g, align 8, !dbg !30, !tbaa !25
  %4 = load i32, ptr @"llvm.bar:3:1$0:0", align 4
  %5 = tail call i32 @llvm.bpf.passthrough.i32.i32(i32 2, i32 %4)
  %conv2 = zext i32 %5 to i64, !dbg !31
  store volatile i64 %conv2, ptr @g, align 8, !dbg !32, !tbaa !25
  %6 = load i32, ptr @"llvm.bar:4:32$0:0", align 4
  %7 = tail call i32 @llvm.bpf.passthrough.i32.i32(i32 3, i32 %6)
  %conv3 = zext i32 %7 to i64, !dbg !33
  store volatile i64 %conv3, ptr @g, align 8, !dbg !34, !tbaa !25
  %8 = load i32, ptr @"llvm.bar:5:32$0:0", align 4
  %9 = tail call i32 @llvm.bpf.passthrough.i32.i32(i32 4, i32 %8)
  %conv4 = zext i32 %9 to i64, !dbg !35
  store volatile i64 %conv4, ptr @g, align 8, !dbg !36, !tbaa !25
  ret void, !dbg !37
}

; Function Attrs: nofree nosync nounwind memory(none)
declare i32 @llvm.bpf.passthrough.i32.i32(i32, i32) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { "btf_ama" }
attributes #1 = { nofree nounwind memory(readwrite, argmem: none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nofree nosync nounwind memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 4, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 17.0.0 (/home/eddy/work/llvm-project/clang 2f8c5c0afd1d79a771dd74c8fb1e5bbae6d04eb7)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/eddy/work/tmp", checksumkind: CSK_MD5, checksum: "ff78616039301f51cd56ee6ea1377b86")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !6)
!6 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !3, line: 3, size: 32, elements: !8)
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !7, file: !3, line: 3, baseType: !10, size: 32)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{i32 7, !"Dwarf Version", i32 5}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{i32 7, !"frame-pointer", i32 2}
!15 = !{!"clang version 17.0.0 (/home/eddy/work/llvm-project/clang 2f8c5c0afd1d79a771dd74c8fb1e5bbae6d04eb7)"}
!16 = distinct !DISubprogram(name: "root", scope: !3, file: !3, line: 5, type: !17, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !19)
!17 = !DISubroutineType(types: !18)
!18 = !{null}
!19 = !{!20}
!20 = !DILocalVariable(name: "bar", scope: !16, file: !3, line: 6, type: !21)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!22 = !DILocation(line: 0, scope: !16)
!23 = !DILocation(line: 7, column: 7, scope: !16)
!24 = !DILocation(line: 7, column: 5, scope: !16)
!25 = !{!26, !26, i64 0}
!26 = !{!"long", !27, i64 0}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C/C++ TBAA"}
!29 = !DILocation(line: 8, column: 7, scope: !16)
!30 = !DILocation(line: 8, column: 5, scope: !16)
!31 = !DILocation(line: 9, column: 7, scope: !16)
!32 = !DILocation(line: 9, column: 5, scope: !16)
!33 = !DILocation(line: 10, column: 7, scope: !16)
!34 = !DILocation(line: 10, column: 5, scope: !16)
!35 = !DILocation(line: 11, column: 7, scope: !16)
!36 = !DILocation(line: 11, column: 5, scope: !16)
!37 = !DILocation(line: 12, column: 1, scope: !16)
