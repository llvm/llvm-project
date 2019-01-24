; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s

; CHECK: .target sm_{{[0-9]+$}}
; CHECK-NOT: }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 9.0.0 (trunk 351924) (llvm/trunk 351968)", isOptimized: false, runtimeVersion: 0, emissionKind: DebugDirectivesOnly, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "new.cc", directory: "/test")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!7 = !{i32 7, !"PIC Level", i32 2}
!8 = !{!"clang version 9.0.0 (trunk 351924) (llvm/trunk 351968)"}
