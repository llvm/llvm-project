; RUN: llc -split-dwarf-file=foo.dwo -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-info -debug-line %t | FileCheck %s

; CHECK: .debug_info contents:
; CHECK: DW_AT_comp_dir ("/usr/local/google/home/blaikie/dev/scratch")

; CHECK: .debug_line contents:
; CHECK: file_names[ 1]:
; CHECK-NEXT:      name: "main.c"
; CHECK-NEXT: dir_index: 0

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() !dbg !6 {
entry:
  ret i32 0, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0 (trunk 349782) (llvm/trunk 349794)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: GNU)
!1 = !DIFile(filename: "main.c", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 8.0.0 (trunk 349782) (llvm/trunk 349794)"}
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 2, column: 1, scope: !6)
