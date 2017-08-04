; RUN: not llvm-as -disable-output <%s 2>&1 | FileCheck %s
; CHECK:      assembly parsed, but does not verify
; CHECK-NEXT: DICompileUnit not listed in llvm.dbg.cu
; CHECK-NEXT: !0 = distinct !DICompileUnit(language: DW_LANG_Fortran77, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)

!named = !{!1}
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!llvm.dbg.cu = !{}
!1 = distinct !DICompileUnit(file: !2, language: DW_LANG_Fortran77)
!2 = !DIFile(filename: "test.f", directory: "")
