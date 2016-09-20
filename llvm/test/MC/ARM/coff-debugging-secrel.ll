; RUN: llc -mtriple thumbv7--windows-itanium -filetype obj -o - %s \
; RUN:     | llvm-readobj -r - | FileCheck %s -check-prefix CHECK-ITANIUM

; RUN: sed -e 's/"Dwarf Version"/"CodeView"/' %s \
; RUN:    | llc -mtriple thumbv7--windows-msvc -filetype obj -o - \
; RUN:    | llvm-readobj -r - | FileCheck %s -check-prefix CHECK-MSVC

; ModuleID = '/Users/compnerd/work/llvm/test/MC/ARM/reduced.c'
target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7--windows-itanium"

define arm_aapcs_vfpcc void @function() !dbg !1 {
entry:
  ret void, !dbg !0
}

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!9, !10}

!0 = !DILocation(line: 1, scope: !1)
!1 = distinct !DISubprogram(name: "function", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !7, scopeLine: 1, file: !2, scope: !3, type: !4, variables: !6)
!2 = !DIFile(filename: "/Users/compnerd/work/llvm/test/MC/ARM/reduced.c", directory: "/Users/compnerd/work/llvm")
!3 = !DIFile(filename: "/Users/compnerd/work/llvm/test/MC/ARM/reduced.c", directory: "/Users/compnerd/work/llvm")
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = !{}
!7 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0", isOptimized: false, emissionKind: FullDebug, file: !2, enums: !6, retainedTypes: !6, globals: !6, imports: !6)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 1, !"Debug Info Version", i32 3}

; CHECK-ITANIUM: Relocations [
; CHECK-ITANIUM:   Section {{.*}} .debug_info {
; CHECK-ITANIUM:     0x6 IMAGE_REL_ARM_SECREL .debug_abbrev
; CHECK-ITANIUM:     0xC IMAGE_REL_ARM_SECREL .debug_str
; CHECK-ITANIUM:     0x12 IMAGE_REL_ARM_SECREL .debug_str
; CHECK-ITANIUM:     0x16 IMAGE_REL_ARM_SECREL .debug_line
; CHECK-ITANIUM:   }
; CHECK-ITANIUM:   Section {{.*}}.debug_pubnames {
; CHECK-ITANIUM:     0x6 IMAGE_REL_ARM_SECREL .debug_info
; CHECK-ITANIUM:   }
; CHECK-ITANIUM: ]

; CHECK-MSVC: Relocations [
; CHECK-MSVC:   Section {{.*}} .debug$S {
; CHECK-MSVC:     0x5A IMAGE_REL_ARM_SECREL function
; CHECK-MSVC:     0x5E IMAGE_REL_ARM_SECTION function
; CHECK-MSVC:     0x78 IMAGE_REL_ARM_SECREL function
; CHECK-MSVC:     0x7C IMAGE_REL_ARM_SECTION function
; CHECK-MSVC:   }
; CHECK-MSVC: ]

