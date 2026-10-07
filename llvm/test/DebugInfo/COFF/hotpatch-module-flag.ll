; RUN: split-file %s %t
; RUN: llc -mtriple=x86_64-pc-windows-msvc -filetype=obj %t/hotpatch.ll -o - \
; RUN:   | llvm-readobj --codeview - | FileCheck %s --check-prefix=HOTPATCH

;; The S_COMPILE3 HotPatch flag must also come from the "ms-hotpatch" module
;; flag, since LTO code generation doesn't set TargetOptions::Hotpatch. Merging
;; with a module compiled without /hotpatch resets the flag to 0.
; RUN: llvm-link %t/hotpatch.ll %t/no-hotpatch.ll -o %t/merged.bc
; RUN: llvm-dis %t/merged.bc -o - | FileCheck %s --check-prefix=MERGED
; RUN: llc -mtriple=x86_64-pc-windows-msvc -filetype=obj %t/merged.bc -o - \
; RUN:   | llvm-readobj --codeview - | FileCheck %s --check-prefix=NO-HOTPATCH

; HOTPATCH:         Compile3Sym {
; HOTPATCH:           Flags [ (0x4000)
; HOTPATCH-NEXT:        HotPatch (0x4000)
; HOTPATCH-NEXT:      ]

; MERGED: !{i32 8, !"ms-hotpatch", i32 0}

; NO-HOTPATCH:      Compile3Sym {
; NO-HOTPATCH:        Flags [ (0x0)
; NO-HOTPATCH-NEXT:   ]

;--- hotpatch.ll
define void @f() {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "clang", emissionKind: NoDebug)
!1 = !DIFile(filename: "hotpatch.c", directory: "path/to")
!2 = !{i32 2, !"CodeView", i32 1}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 8, !"ms-hotpatch", i32 1}

;--- no-hotpatch.ll
define void @g() {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "clang", emissionKind: NoDebug)
!1 = !DIFile(filename: "no-hotpatch.c", directory: "path/to")
!2 = !{i32 2, !"CodeView", i32 1}
!3 = !{i32 2, !"Debug Info Version", i32 3}
