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
;;  struct bar { } __pai;
;;  volatile unsigned long g;
;;  void root(void) {
;;    struct bar *bar = (void *)0;
;;    g = __builtin_preserve_type_info(*bar, 0);
;;    g = __builtin_preserve_type_info(*bar, 1);
;;    g = __builtin_preserve_type_info(*bar, 2);
;;  }
;;
;; Using the following command:
;;
;;  clang --target=bpf -g -O2 -emit-llvm -S t.c

; CHECK: CO-RE <type_exists> [[[#]]] struct bar
; CHECK: CO-RE <type_size> [[[#]]] struct bar
; CHECK: CO-RE <type_matches> [[[#]]] struct bar

@g = dso_local global i64 0, align 8, !dbg !0
@"llvm.bar:8:1$0" = external global i32, !llvm.preserve.access.index !7 #0
@"llvm.bar:9:0$0" = external global i32, !llvm.preserve.access.index !7 #0
@"llvm.bar:12:1$0" = external global i32, !llvm.preserve.access.index !7 #0

; Function Attrs: nofree nounwind memory(readwrite, argmem: none)
define dso_local void @root() local_unnamed_addr #1 !dbg !14 {
entry:
  call void @llvm.dbg.value(metadata ptr null, metadata !18, metadata !DIExpression()), !dbg !20
  %0 = load i32, ptr @"llvm.bar:8:1$0", align 4
  %1 = tail call i32 @llvm.bpf.passthrough.i32.i32(i32 0, i32 %0)
  %conv = zext i32 %1 to i64, !dbg !21
  store volatile i64 %conv, ptr @g, align 8, !dbg !22, !tbaa !23
  %2 = load i32, ptr @"llvm.bar:9:0$0", align 4
  %3 = tail call i32 @llvm.bpf.passthrough.i32.i32(i32 1, i32 %2)
  %conv1 = zext i32 %3 to i64, !dbg !27
  store volatile i64 %conv1, ptr @g, align 8, !dbg !28, !tbaa !23
  %4 = load i32, ptr @"llvm.bar:12:1$0", align 4
  %5 = tail call i32 @llvm.bpf.passthrough.i32.i32(i32 2, i32 %4)
  %conv2 = zext i32 %5 to i64, !dbg !29
  store volatile i64 %conv2, ptr @g, align 8, !dbg !30, !tbaa !23
  ret void, !dbg !31
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
!llvm.module.flags = !{!9, !10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 4, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 17.0.0 (/home/eddy/work/llvm-project/clang 2f8c5c0afd1d79a771dd74c8fb1e5bbae6d04eb7)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/eddy/work/tmp", checksumkind: CSK_MD5, checksum: "5bf218e82301e866fa302fd927913bcf")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !6)
!6 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !3, line: 3, elements: !8)
!8 = !{}
!9 = !{i32 7, !"Dwarf Version", i32 5}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{i32 7, !"frame-pointer", i32 2}
!13 = !{!"clang version 17.0.0 (/home/eddy/work/llvm-project/clang 2f8c5c0afd1d79a771dd74c8fb1e5bbae6d04eb7)"}
!14 = distinct !DISubprogram(name: "root", scope: !3, file: !3, line: 5, type: !15, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !{!18}
!18 = !DILocalVariable(name: "bar", scope: !14, file: !3, line: 6, type: !19)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!20 = !DILocation(line: 0, scope: !14)
!21 = !DILocation(line: 7, column: 7, scope: !14)
!22 = !DILocation(line: 7, column: 5, scope: !14)
!23 = !{!24, !24, i64 0}
!24 = !{!"long", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C/C++ TBAA"}
!27 = !DILocation(line: 8, column: 7, scope: !14)
!28 = !DILocation(line: 8, column: 5, scope: !14)
!29 = !DILocation(line: 9, column: 7, scope: !14)
!30 = !DILocation(line: 9, column: 5, scope: !14)
!31 = !DILocation(line: 10, column: 1, scope: !14)
