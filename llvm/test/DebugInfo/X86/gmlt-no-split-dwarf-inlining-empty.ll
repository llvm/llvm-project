; RUN: llc -split-dwarf-file=foo.dwo  %s -filetype=obj -o - | llvm-objdump -h - | FileCheck %s

; Created from:
;   void f1() {
;   }
; $ clang-tot gmlt-no-split-dwarf-inlining-empty.c -fno-split-dwarf-inlining -gmlt -gsplit-dwarf -c -emit-llvm -S

; CHECK-NOT: .debug_{{.*}}.dwo

target triple = "x86_64-pc-linux"

define dso_local void @f1() !dbg !7 {
entry:
  ret void, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (trunk 353744) (llvm/trunk 353759)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, splitDebugInlining: false, nameTableKind: GNU)
!1 = !DIFile(filename: "gmlt-no-split-dwarf-inlining-empty.c", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 9.0.0 (trunk 353744) (llvm/trunk 353759)"}
!7 = distinct !DISubprogram(name: "f1", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 2, column: 1, scope: !7)
