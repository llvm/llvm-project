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
;;  enum bar { U, V };
;;  volatile unsigned long g;
;;  void root(void) {
;;    g = __builtin_preserve_enum_value(*(enum bar *)U, 0);
;;    g = __builtin_preserve_enum_value(*(enum bar *)V, 1);
;;  }
;;
;; Using the following command:
;;
;;  clang --target=bpf -g -O2 -emit-llvm -S t.c

; CHECK: CO-RE <enumval_exists> [[[#]]] enum bar::U = 0
; CHECK: CO-RE <enumval_value> [[[#]]] enum bar::V = 1

@g = dso_local global i64 0, align 8, !dbg !0
@"llvm.bar:11:1$1" = external global i64, !llvm.preserve.access.index !5 #0
@"llvm.bar:10:1$0" = external global i64, !llvm.preserve.access.index !5 #0

; Function Attrs: nofree nounwind memory(readwrite, argmem: none)
define dso_local void @root() local_unnamed_addr #1 !dbg !18 {
entry:
  %0 = load i64, ptr @"llvm.bar:10:1$0", align 8
  %1 = tail call i64 @llvm.bpf.passthrough.i64.i64(i32 1, i64 %0)
  store volatile i64 %1, ptr @g, align 8, !dbg !22, !tbaa !23
  %2 = load i64, ptr @"llvm.bar:11:1$1", align 8
  %3 = tail call i64 @llvm.bpf.passthrough.i64.i64(i32 0, i64 %2)
  store volatile i64 %3, ptr @g, align 8, !dbg !27, !tbaa !23
  ret void, !dbg !28
}

; Function Attrs: nofree nosync nounwind memory(none)
declare i64 @llvm.bpf.passthrough.i64.i64(i32, i64) #2

attributes #0 = { "btf_ama" }
attributes #1 = { nofree nounwind memory(readwrite, argmem: none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nofree nosync nounwind memory(none) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13, !14, !15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 4, type: !11, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 17.0.0 (/home/eddy/work/llvm-project/clang 2f8c5c0afd1d79a771dd74c8fb1e5bbae6d04eb7)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !10, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/eddy/work/tmp", checksumkind: CSK_MD5, checksum: "5423aa9ef48cb61e948b5c2bd75fd1df")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "bar", file: !3, line: 3, baseType: !6, size: 32, elements: !7)
!6 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!7 = !{!8, !9}
!8 = !DIEnumerator(name: "U", value: 0)
!9 = !DIEnumerator(name: "V", value: 1)
!10 = !{!0}
!11 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !12)
!12 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!13 = !{i32 7, !"Dwarf Version", i32 5}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{i32 7, !"frame-pointer", i32 2}
!17 = !{!"clang version 17.0.0 (/home/eddy/work/llvm-project/clang 2f8c5c0afd1d79a771dd74c8fb1e5bbae6d04eb7)"}
!18 = distinct !DISubprogram(name: "root", scope: !3, file: !3, line: 5, type: !19, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !21)
!19 = !DISubroutineType(types: !20)
!20 = !{null}
!21 = !{}
!22 = !DILocation(line: 6, column: 5, scope: !18)
!23 = !{!24, !24, i64 0}
!24 = !{!"long", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C/C++ TBAA"}
!27 = !DILocation(line: 7, column: 5, scope: !18)
!28 = !DILocation(line: 8, column: 1, scope: !18)
