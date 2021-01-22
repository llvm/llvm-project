; RUN: llc -mtriple=amdgcn-amd-amdhsa -filetype=obj -o - < %s | llvm-readelf -S - | FileCheck --check-prefixes=NOEXCEPTIONS_NOFORCE-DEBUG %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -filetype=obj --force-dwarf-frame-section -o - < %s | llvm-readelf -S - | FileCheck --check-prefixes=NOEXCEPTIONS_FORCE-DEBUG %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -filetype=obj --exception-model=dwarf -o - < %s | llvm-readelf -S - | FileCheck --check-prefixes=EXCEPTIONS_NOFORCE-DEBUG %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -filetype=obj --force-dwarf-frame-section --exception-model=dwarf -o - < %s | llvm-readelf -S - | FileCheck --check-prefixes=EXCEPTIONS_FORCE-DEBUG %s

; Test that demonstrates we produce a .debug_frame only when exceptions are enabled even if --force-dwarf-frame-section is enabled

; NOEXCEPTIONS_NOFORCE-DEBUG-NOT: .debug_frame
; NOEXCEPTIONS_FORCE-DEBUG-NOT: .debug_frame
; EXCEPTIONS_NOFORCE-DEBUG: .debug_frame
; EXCEPTIONS_FORCE-DEBUG: .debug_frame

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
