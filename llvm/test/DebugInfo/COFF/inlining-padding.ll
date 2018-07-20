; RUN: llc < %s -filetype=obj -o - | llvm-readobj -codeview -codeview-subsection-bytes | FileCheck %s

; Check how we pad out the LF_FUNC_ID records. The 00F3F2F1 bytes in LeafData are
; what's interesting here.

; CHECK:  FuncId (0x1002) {
; CHECK:    TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:    Name: a
; CHECK:    LeafData (
; CHECK:      0000: {{.*}} 6100F2F1           |........a...|
; CHECK:    )
; CHECK:  }
; CHECK:  FuncId (0x1003) {
; CHECK:    TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:    Name: ab
; CHECK:    LeafData (
; CHECK:      0000: {{.*}} 616200F1           |........ab..|
; CHECK:    )
; CHECK:  }
; CHECK:  FuncId (0x1004) {
; CHECK:    TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:    Name: abc
; CHECK:    LeafData (
; CHECK:      0000: {{.*}} 61626300           |........abc.|
; CHECK:    )
; CHECK:  }
; CHECK:  FuncId (0x1005) {
; CHECK:    TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:    Name: abcd
; CHECK:    LeafData (
; CHECK:      0000: {{.*}} 61626364 00F3F2F1  |........abcd....|
; CHECK:    )
; CHECK:  }

; C++ source used to generate the IR:
;
; extern volatile int x;
; static void a() { x++; }
; static void ab() { x++; }
; static void abc() { x++; }
; static void abcd() { x++; }
; int main() {
;   a();
;   ab();
;   abc();
;   abcd();
; }

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@x = external global i32, align 4

; Function Attrs: norecurse nounwind
define i32 @main() #0 !dbg !6 {
entry:
  store volatile i32 0, i32* @x, align 4, !dbg !11, !tbaa !16
  store volatile i32 0, i32* @x, align 4, !dbg !20, !tbaa !16
  store volatile i32 0, i32* @x, align 4, !dbg !23, !tbaa !16
  store volatile i32 0, i32* @x, align 4, !dbg !26, !tbaa !16
  ret i32 0, !dbg !29
}

attributes #0 = { norecurse nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-features"="+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 270461) (llvm/trunk 270469)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.9.0 (trunk 270461) (llvm/trunk 270469)"}
!6 = distinct !DISubprogram(name: "main", scope: !7, file: !7, line: 6, type: !8, isLocal: false, isDefinition: true, scopeLine: 6, isOptimized: true, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 2, scope: !12, inlinedAt: !15)
!12 = distinct !DISubprogram(name: "a", scope: !7, file: !7, line: 2, type: !13, isLocal: true, isDefinition: true, scopeLine: 2, isOptimized: true, unit: !0, retainedNodes: !2)
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = distinct !DILocation(line: 7, scope: !6)
!16 = !{!17, !17, i64 0}
!17 = !{!"int", !18, i64 0}
!18 = !{!"omnipotent char", !19, i64 0}
!19 = !{!"Simple C/C++ TBAA"}
!20 = !DILocation(line: 3, scope: !21, inlinedAt: !22)
!21 = distinct !DISubprogram(name: "ab", scope: !7, file: !7, line: 3, type: !13, isLocal: true, isDefinition: true, scopeLine: 3, isOptimized: true, unit: !0, retainedNodes: !2)
!22 = distinct !DILocation(line: 8, scope: !6)
!23 = !DILocation(line: 4, scope: !24, inlinedAt: !25)
!24 = distinct !DISubprogram(name: "abc", scope: !7, file: !7, line: 4, type: !13, isLocal: true, isDefinition: true, scopeLine: 4, isOptimized: true, unit: !0, retainedNodes: !2)
!25 = distinct !DILocation(line: 9, scope: !6)
!26 = !DILocation(line: 5, scope: !27, inlinedAt: !28)
!27 = distinct !DISubprogram(name: "abcd", scope: !7, file: !7, line: 5, type: !13, isLocal: true, isDefinition: true, scopeLine: 5, isOptimized: true, unit: !0, retainedNodes: !2)
!28 = distinct !DILocation(line: 10, scope: !6)
!29 = !DILocation(line: 11, scope: !6)
