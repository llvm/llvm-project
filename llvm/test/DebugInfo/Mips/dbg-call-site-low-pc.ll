;; Test mips32:
; RUN: llc -emit-call-site-info %s -mtriple=mips -filetype=obj -o -| llvm-dwarfdump -| FileCheck %s
;; Test mipsel:
; RUN: llc -emit-call-site-info %s -mtriple=mipsel -filetype=obj -o -| llvm-dwarfdump -| FileCheck %s
;; Test mips64:
; RUN: llc -emit-call-site-info %s -mtriple=mips64 -filetype=obj -o -| llvm-dwarfdump -| FileCheck %s
;; Test mips64el:
; RUN: llc -emit-call-site-info %s -mtriple=mips64el -filetype=obj -o -| llvm-dwarfdump -| FileCheck %s

;; Source:
;; __attribute__((noinline))
;; extern void f1(int a);
;; __attribute__((noinline))
;; int main(){
;;   int x = 10;
;;   f1(x);
;;   return ++x;
;; }
;; Command: clang -g -O2 -target mips-linux-gnu -S -emit-llvm m.c -c
;; Confirm that DW_AT_low_pc (call return address) points to instruction after call delay slot.

;; Test mips, mipsel, mips64, mips64el:
; CHECK:        DW_TAG_GNU_call_site
; CHECK-NEXT:     DW_AT_abstract_origin {{.*}} "f1"
; CHECK-NEXT:     DW_AT_low_pc (0x{{(00000000)?}}00000010)

; ModuleID = 'm.c'
source_filename = "m.c"
target datalayout = "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"
target triple = "mips-unknown-linux-gnu"

; Function Attrs: noinline nounwind
define dso_local i32 @main() local_unnamed_addr !dbg !12 {
entry:
  call void @llvm.dbg.value(metadata i32 10, metadata !16, metadata !DIExpression()), !dbg !17
  tail call void @f1(i32 signext 10), !dbg !17
  call void @llvm.dbg.value(metadata i32 11, metadata !16, metadata !DIExpression()), !dbg !17
  ret i32 11, !dbg !17
}

declare !dbg !4 dso_local void @f1(i32 signext) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "m.c", directory: "/dir")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "f1", scope: !1, file: !1, line: 2, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{!"clang version 11.0.0"}
!12 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 5, type: !13, scopeLine: 5, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{!7}
!15 = !{!16}
!16 = !DILocalVariable(name: "x", scope: !12, file: !1, line: 6, type: !7)
!17 = !DILocation(line: 0, scope: !12)
