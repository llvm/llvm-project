; RUN: llc -mtriple=amdgcn-amd-amdhsa -o - < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa --force-dwarf-frame-section -o - < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa --exception-model=dwarf -o - < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa --force-dwarf-frame-section --exception-model=dwarf -o - < %s | FileCheck %s

; CHECK: .cfi_sections .debug_frame

define void @f() nounwind !dbg !0 {
entry:
  ret void
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7}
!5 = !{!0}

!0 = distinct !DISubprogram(name: "f", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, scopeLine: 1, file: !6, scope: !1, type: !3)
!1 = !DIFile(filename: "/home/espindola/llvm/test.c", directory: "/home/espindola/llvm/build")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 ()", isOptimized: true, emissionKind: FullDebug, file: !6, enums: !{}, retainedTypes: !{})
!3 = !DISubroutineType(types: !4)
!4 = !{null}
!6 = !DIFile(filename: "/home/espindola/llvm/test.c", directory: "/home/espindola/llvm/build")
!7 = !{i32 1, !"Debug Info Version", i32 3}
