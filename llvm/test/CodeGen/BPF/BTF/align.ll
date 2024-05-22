; RUN: llc -mtriple=bpfel -filetype=obj < %s | llvm-readelf -S - | FileCheck %s
; RUN: llc -mtriple=bpfeb -filetype=obj < %s | llvm-readelf -S - | FileCheck %s
; Source:
;   int foo() { return 0; }
; Compilation flags:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define dso_local i32 @foo() local_unnamed_addr #0 !dbg !7 {
entry:
  ret i32 0, !dbg !12
}
; CHECK:   Name              Type            Address          Off           Size          ES Flg Lk Inf Al
; CHECK:   .BTF              PROGBITS        0000000000000000 {{[0-9a-f]+}} {{[0-9a-f]+}} 00      0   0  4
; CHECK:   .BTF.ext          PROGBITS        0000000000000000 {{[0-9a-f]+}} {{[0-9a-f]+}} 00      0   0  4

attributes #0 = { mustprogress nofree norecurse nosync nounwind readnone willreturn "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0 (https://github.com/llvm/llvm-project.git b1ab2a57b83e4b7224c38b534532500cc90e5b9a)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/tmp/home/yhs/work/tests/llvm/align")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!"clang version 14.0.0 (https://github.com/llvm/llvm-project.git b1ab2a57b83e4b7224c38b534532500cc90e5b9a)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{}
!12 = !DILocation(line: 1, column: 13, scope: !7)
