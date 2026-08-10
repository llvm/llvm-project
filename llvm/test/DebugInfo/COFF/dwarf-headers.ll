; RUN: llc -dwarf-version=4 \
; RUN:     -filetype=obj -O0 -mtriple=x86_64-unknown-windows-msvc < %s \
; RUN:     | llvm-dwarfdump -v - | FileCheck %s --check-prefix=SINGLE-4

; RUN: llc -split-dwarf-file=foo.dwo -split-dwarf-output=%t.dwo \
; RUN:     -dwarf-version=4 \
; RUN:     -filetype=obj -O0 -mtriple=x86_64-unknown-windows-msvc < %s \
; RUN:     | llvm-dwarfdump -v - | FileCheck %s --check-prefix=O-4
; RUN: llvm-dwarfdump -v %t.dwo | FileCheck %s --check-prefix=DWO-4

; This test is derived from test/CodeGen/X86/dwarf-headers.ll

; Looking for DWARF headers to be generated correctly.
; There are 8 variants with 5 formats: v4 CU, v4 TU, v5 normal/partial CU,
; v5 skeleton/split CU, v5 normal/split TU.  Some v5 variants differ only
; in the unit_type code, and the skeleton/split CU differs from normal/partial
; by having one extra field (dwo_id).
; (v2 thru v4 CUs are all the same, and TUs were invented in v4,
; so we don't bother checking older versions.)

; Test case built from:
;struct S {
;  int s1;
;};
;
;S s;

; Verify the v4 non-split headers.
; Note that we check the exact offset of the DIEs because that tells us
; the length of the header.
;
; SINGLE-4: .debug_info contents:
; SINGLE-4: 0x00000000: Compile Unit: {{.*}} version = 0x0004, abbr_offset
; SINGLE-4: 0x0000000b: DW_TAG_compile_unit

; Verify the v4 split headers.
;
; O-4: .debug_info contents:
; O-4: 0x00000000: Compile Unit: {{.*}} version = 0x0004, abbr_offset
; O-4: 0x0000000b: DW_TAG_compile_unit
;
; DWO-4: .debug_info.dwo contents:
; DWO-4: 0x00000000: Compile Unit: {{.*}} version = 0x0004, abbr_offset
; DWO-4: 0x0000000b: DW_TAG_compile_unit

; Check that basic CodeView compiler info is emitted even when the DWARF debug format is used.
; RUN: llc -dwarf-version=4 \
; RUN:     -filetype=obj -O0 -mtriple=x86_64-unknown-windows-msvc < %s \
; RUN:     | llvm-readobj --codeview - | FileCheck %s --check-prefix=CODEVIEW
; CODEVIEW:      CodeViewDebugInfo [
; CODEVIEW-NEXT:   Section: .debug$S (4)
; CODEVIEW-NEXT:   Magic: 0x4
; CODEVIEW-NEXT:   Subsection [
; CODEVIEW-NEXT:     SubSectionType: Symbols (0xF1)
; CODEVIEW-NEXT:     SubSectionSize: 0x90
; CODEVIEW-NEXT:     ObjNameSym {
; CODEVIEW-NEXT:       Kind: S_OBJNAME (0x1101)
; CODEVIEW-NEXT:       Signature: 0x0
; CODEVIEW-NEXT:       ObjectName:
; CODEVIEW-NEXT:     }
; CODEVIEW-NEXT:     Compile3Sym {
; CODEVIEW-NEXT:       Kind: S_COMPILE3 (0x113C)
; CODEVIEW-NEXT:       Language: Cpp (0x1)
; CODEVIEW-NEXT:       Flags [ (0x0)
; CODEVIEW-NEXT:       ]
; CODEVIEW-NEXT:       Machine: X64 (0xD0)
; CODEVIEW-NEXT:       FrontendVersion: 17.0.0.0
; CODEVIEW-NEXT:       BackendVersion:
; CODEVIEW-NEXT:       VersionName: clang version 17.0.0
; CODEVIEW-NEXT:     }
; CODEVIEW-NEXT:   ]
; CODEVIEW-NEXT: ]

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc"

%struct.S = type { i32 }

@"?s@@3US@@A" = dso_local global %struct.S zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11, !12, !13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "s", linkageName: "?s@@3US@@A", scope: !2, file: !3, line: 5, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 17.0.0 (https://github.com/llvm/llvm-project.git f1106ef6c9d14d5b516ec352279aeee8f9d12818)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.cpp", directory: "e:\\llvm-project\\foo")
!4 = !{!0}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 1, size: 32, flags: DIFlagTypePassByValue, elements: !6, identifier: ".?AUS@@")
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "s1", scope: !5, file: !3, line: 2, baseType: !8, size: 32)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 2}
!12 = !{i32 8, !"PIC Level", i32 2}
!13 = !{i32 7, !"uwtable", i32 2}
!14 = !{i32 1, !"MaxTLSAlign", i32 65536}
!15 = !{!"clang version 17.0.0 (https://github.com/llvm/llvm-project.git f1106ef6c9d14d5b516ec352279aeee8f9d12818)"}
