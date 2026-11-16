; RUN: split-file %s %t
; RUN: llc -mtriple=x86_64-pc-windows-msvc -filetype=obj %t/hotpatch.ll -o - \
; RUN:   | llvm-readobj --codeview - | FileCheck %s --check-prefix=HOTPATCH

;; The S_COMPILE3 HotPatch flag comes from the "ms-hotpatch" module flag.
;; Merging hotpatchable modules preserves it, but merging with a module
;; compiled without /hotpatch resets the flag to 0.
; RUN: llvm-link %t/hotpatch.ll %t/hotpatch2.ll -o %t/hotpatch.bc
; RUN: llvm-dis %t/hotpatch.bc -o - | FileCheck %s --check-prefix=MERGED-HOTPATCH
; RUN: llc -mtriple=x86_64-pc-windows-msvc -filetype=obj %t/hotpatch.bc -o - \
; RUN:   | llvm-readobj --codeview - | FileCheck %s --check-prefix=HOTPATCH
; RUN: llc -mtriple=x86_64-pc-windows-msvc -filetype=obj %t/no-hotpatch.ll -o - \
; RUN:   | llvm-readobj --codeview - | FileCheck %s --check-prefix=NO-HOTPATCH
; RUN: llvm-link %t/hotpatch.ll %t/no-hotpatch.ll -o %t/merged.bc
; RUN: llvm-dis %t/merged.bc -o - | FileCheck %s --check-prefix=MERGED
; RUN: llc -mtriple=x86_64-pc-windows-msvc -filetype=obj %t/merged.bc -o - \
; RUN:   | llvm-readobj --codeview - | FileCheck %s --check-prefix=NO-HOTPATCH
; RUN: llvm-link %t/no-hotpatch.ll %t/hotpatch.ll -o %t/merged-reverse.bc
; RUN: llvm-dis %t/merged-reverse.bc -o - | FileCheck %s --check-prefix=MERGED

; HOTPATCH:         Compile3Sym {
; HOTPATCH:           Flags [ (0x4000)
; HOTPATCH-NEXT:        HotPatch (0x4000)
; HOTPATCH-NEXT:      ]

; MERGED: !{i32 8, !"ms-hotpatch", i32 0}
; MERGED-HOTPATCH: !{i32 8, !"ms-hotpatch", i32 1}

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

;--- hotpatch2.ll
define void @g() {
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 8, !"ms-hotpatch", i32 1}

;--- no-hotpatch.ll
define void @h() {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "clang", emissionKind: NoDebug)
!1 = !DIFile(filename: "no-hotpatch.c", directory: "path/to")
!2 = !{i32 2, !"CodeView", i32 1}
!3 = !{i32 2, !"Debug Info Version", i32 3}
