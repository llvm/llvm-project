; RUN: llc -O0 %s -filetype=obj -o %t.o
; RUN: llvm-dwarfdump -arch arm64   %t.o | FileCheck %s
; AArch64 does not support Metal. However in the absence of a suitable target
; it can still be used to test that DW_LANG_Metal/DW_LNAME_Metal can be
; encoded/decoded correctly.
; CHECK: DW_AT_language (DW_LANG_Metal)
source_filename = "test.cpp"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

; Function Attrs: mustprogress noinline norecurse nounwind optnone ssp uwtable(sync)
define noundef i32 @main() #0 !dbg !11 {
entry:
  ret i32 0, !dbg !14
}

attributes #0 = { mustprogress norecurse nounwind }

!llvm.module.flags = !{!3, !4, !5, !6, !7, !8, !9}
!llvm.dbg.cu = !{!0}
!llvm.linker.options = !{}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_Metal, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cpp", directory: "/tmp")
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = !{i32 2, !"SDK Version", [2 x i32] [i32 15, i32 0]}
!4 = !{i32 7, !"Dwarf Version", i32 5}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{i32 8, !"PIC Level", i32 2}
!8 = !{i32 7, !"uwtable", i32 1}
!9 = !{i32 7, !"frame-pointer", i32 1}
!10 = !{!"clang"}
!11 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 8, type: !12, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!12 = !DISubroutineType(types: !13)
!13 = !{!2}
!14 = !DILocation(line: 11, column: 1, scope: !11)
