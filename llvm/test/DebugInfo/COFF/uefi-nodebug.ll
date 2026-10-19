; RUN: llc -filetype=obj -o - < %s | llvm-readobj --codeview - | FileCheck %s
; Check that only basic compiler info is emitted for UEFI targets with
; emissionKind NoDebug, matching the behavior of Windows targets.

; CHECK-NOT:  CodeViewTypes
; CHECK:      CodeViewDebugInfo [
; CHECK-NEXT:   Section: .debug$S (4)
; CHECK-NEXT:   Magic: 0x4
; CHECK-NEXT:   Subsection [
; CHECK-NEXT:     SubSectionType: Symbols (0xF1)
; CHECK-NEXT:     SubSectionSize: 0x2C
; CHECK-NEXT:     ObjNameSym {
; CHECK-NEXT:       Kind: S_OBJNAME (0x1101)
; CHECK-NEXT:       Signature: 0x0
; CHECK-NEXT:       ObjectName:
; CHECK-NEXT:     }
; CHECK-NEXT:     Compile3Sym {
; CHECK-NEXT:       Kind: S_COMPILE3 (0x113C)
; CHECK-NEXT:       Language: C (0x0)
; CHECK-NEXT:       Flags [ (0x0)
; CHECK-NEXT:       ]
; CHECK-NEXT:       Machine: X64 (0xD0)
; CHECK-NEXT:       FrontendVersion:
; CHECK-NEXT:       BackendVersion:
; CHECK-NEXT:       VersionName: clang
; CHECK-NEXT:     }
; CHECK-NEXT:   ]
; CHECK-NEXT: ]

source_filename = "empty"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-uefi"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "clang", emissionKind: NoDebug)
!1 = !DIFile(filename: "empty", directory: "path/to")
!2 = !{i32 2, !"Debug Info Version", i32 3}
