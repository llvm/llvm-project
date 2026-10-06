; Verify that DIStringType rejects a `charType:` that is not a DIType.

; RUN: llvm-as -disable-output < %s 2>&1 | FileCheck %s

; CHECK: invalid character type

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1}

!0 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !2, producer: "flang", emissionKind: FullDebug, retainedTypes: !3)
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !DIFile(filename: "string.f90", directory: "/")
!3 = !{!4}
!4 = !DIStringType(name: "character(4)", charType: !2, size: 32)
