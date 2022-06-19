; Make sure that absolute source dir is detected correctly regardless of the platform.

; On powerpc llvm-nm describes win_func as a global variable, not a function. It breaks the test.
; It is not essential to DWARF path handling code we're testing here.
; UNSUPPORTED: powerpc
; REQUIRES: object-emission
; RUN: %llc_dwarf -O0 -filetype=obj -o %t < %s
; RUN: llvm-nm --radix=o %t | grep posix_absolute_func > %t.posix_absolute_func
; RUN: llvm-nm --radix=o %t | grep posix_relative_func > %t.posix_relative_func
; RUN: llvm-nm --radix=o %t | grep win_func > %t.win_func
; RUN: llvm-symbolizer --functions=linkage --inlining --no-demangle --obj %t < %t.posix_absolute_func | FileCheck %s --check-prefix=POSIX_A
; RUN: llvm-symbolizer --functions=linkage --inlining --no-demangle --obj %t < %t.posix_relative_func | FileCheck %s --check-prefix=POSIX_R
; RUN: llvm-symbolizer --functions=linkage --inlining --no-demangle --obj %t < %t.win_func | FileCheck %s --check-prefix=WIN

;POSIX_A: posix_absolute_func
;POSIX_A: /absolute/posix/path{{[\/]}}posix.c

;POSIX_R: posix_relative_func
;POSIX_R: /ABSOLUTE/CU/PATH{{[\/]}}relative/posix/path{{[\/]}}posix2.c

;WIN: win_func
;WIN: E:\absolute\windows\path{{[\/]}}win.c

define i32 @win_func() #0 !dbg !54 {
  ret i32 5, !dbg !511
}

define i32 @posix_absolute_func() #0 !dbg !34 {
  ret i32 3, !dbg !311
}

define i32 @posix_relative_func() #0 !dbg !44 {
  ret i32 4, !dbg !411
}

!llvm.dbg.cu = !{!50, !30, !40}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang"}

!50 = distinct !DICompileUnit(language: DW_LANG_C99, file: !512, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !52)
!51 = !DIFile(filename: "win.c", directory: "E:\\absolute\\windows\\path")
!52 = !{}
!53 = !{!54}
!54 = distinct !DISubprogram(name: "win_func", scope: !51, file: !51, line: 55, type: !55, unit: !50, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, retainedNodes: !52)
!55 = !DISubroutineType(types: !56)
!56 = !{!57}
!57 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!511 = !DILocation(line: 55, column: 2, scope: !54)
!512 = !DIFile(filename: "a.c", directory: "/WIN_CU/PATH")

!30 = distinct !DICompileUnit(language: DW_LANG_C99, file: !312, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !32)
!31 = !DIFile(filename: "posix.c", directory: "/absolute/posix/path")
!32 = !{}
!33 = !{!34}
!34 = distinct !DISubprogram(name: "posix_absolute_func", scope: !31, file: !31, line: 33, type: !35, unit: !30, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, retainedNodes: !32)
!35 = !DISubroutineType(types: !36)
!36 = !{!37}
!37 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!311 = !DILocation(line: 33, column: 2, scope: !34)
!312 = !DIFile(filename: "b.c", directory: "/POSIX_CU/PATH")

!40 = distinct !DICompileUnit(language: DW_LANG_C99, file: !412, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !42)
!41 = !DIFile(filename: "posix2.c", directory: "relative/posix/path")
!42 = !{}
!43 = !{!44}
!44 = distinct !DISubprogram(name: "posix_relative_func", scope: !41, file: !41, line: 44, type: !45, unit: !40, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, retainedNodes: !42)
!45 = !DISubroutineType(types: !46)
!46 = !{!47}
!47 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!411 = !DILocation(line: 44, column: 2, scope: !44)
!412 = !DIFile(filename: "c.c", directory: "/ABSOLUTE/CU/PATH")
