; RUN: sed -e 's/<LANG1>/DW_LANG_C/;s/<LANG2>/C/' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %t
;
; RUN: sed -e 's/<LANG1>/DW_LANG_C89/;s/<LANG2>/C/' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %t
;
; RUN: sed -e 's/<LANG1>/DW_LANG_C99/;s/<LANG2>/C/' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %t
;
; RUN: sed -e 's/<LANG1>/DW_LANG_C11/;s/<LANG2>/C/' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %t
;
; RUN: sed -e 's/<LANG1>/DW_LANG_C_plus_plus/;s/<LANG2>/Cpp/' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %t
;
; RUN: sed -e 's/<LANG1>/DW_LANG_C_plus_plus_03/;s/<LANG2>/Cpp/' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %t
;
; RUN: sed -e 's/<LANG1>/DW_LANG_C_plus_plus_11/;s/<LANG2>/Cpp/' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %t
;
; RUN: sed -e 's/<LANG1>/DW_LANG_C_plus_plus_14/;s/<LANG2>/Cpp/' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %t
;
; RUN: sed -e 's/<LANG1>/DW_LANG_Fortran77/;s/<LANG2>/Fortran/' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %t
;
; RUN: sed -e 's/<LANG1>/DW_LANG_Fortran90/;s/<LANG2>/Fortran/' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %t
;
; RUN: sed -e 's/<LANG1>/DW_LANG_Fortran95/;s/<LANG2>/Fortran/' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %t
;
; RUN: sed -e 's/<LANG1>/DW_LANG_Fortran03/;s/<LANG2>/Fortran/' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %t
;
; RUN: sed -e 's/<LANG1>/DW_LANG_Fortran08/;s/<LANG2>/Fortran/' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %t
;
; RUN: sed -e 's/<LANG1>/DW_LANG_Rust/;s/<LANG2>/Rust/' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %t

; CHECK:      CodeViewDebugInfo [
; CHECK:        Subsection [
; CHECK:          SubSectionType: Symbols (0xF1)
; CHECK:          Compile3Sym {
; CHECK-NEXT:        Kind: S_COMPILE3 (0x113C)
; CHECK-NEXT:        Language: <LANG2>
; CHECK-NEXT:        Flags [ (0x0)
; CHECK-NEXT:        ]
; CHECK-NEXT:        Machine: X64 (0xD0)
; CHECK-NEXT:        FrontendVersion: {{[0-9.]*}}
; CHECK-NEXT:        BackendVersion: {{[0-9.]*}}
; CHECK-NEXT:        VersionName: clang
; CHECK-NEXT:     }
; CHECK:        ]
; CHECK:      ]

; ModuleID = 'empty'
source_filename = "empty"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: <LANG1>, file: !1, producer: "clang")
!1 = !DIFile(filename: "empty", directory: "path/to")
!2 = !{i32 2, !"CodeView", i32 1}
!3 = !{i32 2, !"Debug Info Version", i32 3}
