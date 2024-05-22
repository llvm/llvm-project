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
;;    g = __builtin_btf_type_id(*bar, 0);
;;    g = __builtin_btf_type_id(*bar, 1);
;;  }
;;
;; Using the following command:
;;
;;  clang --target=bpf -g -O2 -emit-llvm -S t.c

; CHECK: CO-RE <local_type_id> [[[#]]] struct bar
; CHECK: CO-RE <target_type_id> [[[#]]] struct bar

@g = dso_local global i64 0, align 8, !dbg !0
@"llvm.btf_type_id.0$6" = external global i64, !llvm.preserve.access.index !7 #0
@"llvm.btf_type_id.1$7" = external global i64, !llvm.preserve.access.index !7 #0

; Function Attrs: nofree nounwind memory(readwrite, argmem: none)
define dso_local void @root() local_unnamed_addr #1 !dbg !14 {
entry:
  call void @llvm.dbg.value(metadata ptr null, metadata !18, metadata !DIExpression()), !dbg !20
  %0 = load i64, ptr @"llvm.btf_type_id.0$6", align 8
  %1 = tail call i64 @llvm.bpf.passthrough.i64.i64(i32 0, i64 %0)
  store volatile i64 %1, ptr @g, align 8, !dbg !21, !tbaa !22
  %2 = load i64, ptr @"llvm.btf_type_id.1$7", align 8
  %3 = tail call i64 @llvm.bpf.passthrough.i64.i64(i32 1, i64 %2)
  store volatile i64 %3, ptr @g, align 8, !dbg !26, !tbaa !22
  ret void, !dbg !27
}

; Function Attrs: nofree nosync nounwind memory(none)
declare i64 @llvm.bpf.passthrough.i64.i64(i32, i64) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { "btf_type_id" }
attributes #1 = { nofree nounwind memory(readwrite, argmem: none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nofree nosync nounwind memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 4, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 17.0.0 (/home/eddy/work/llvm-project/clang 2f8c5c0afd1d79a771dd74c8fb1e5bbae6d04eb7)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/eddy/work/tmp", checksumkind: CSK_MD5, checksum: "29efc9dba44aaba9e4b0c389bb8694ea")
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
!21 = !DILocation(line: 7, column: 5, scope: !14)
!22 = !{!23, !23, i64 0}
!23 = !{!"long", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C/C++ TBAA"}
!26 = !DILocation(line: 8, column: 5, scope: !14)
!27 = !DILocation(line: 9, column: 1, scope: !14)
