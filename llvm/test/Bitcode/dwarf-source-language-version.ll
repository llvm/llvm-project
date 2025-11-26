; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s --implicit-check-not "sourceLanguageVersion: 0"

; CHECK: sourceLanguageVersion: 120

source_filename = "cu.cpp"
target triple = "arm64-apple-macosx"

!llvm.dbg.cu = !{!0, !5}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(sourceLanguageName: DW_LNAME_ObjC_plus_plus, sourceLanguageVersion: 120, file: !1, producer: "handwritten", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !2, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!1 = !DIFile(filename: "cu.cpp", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DICompileUnit(sourceLanguageName: DW_LNAME_ObjC_plus_plus, sourceLanguageVersion: 0, file: !6, producer: "handwritten", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !2, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!6 = !DIFile(filename: "cu2.cpp", directory: "/tmp")
