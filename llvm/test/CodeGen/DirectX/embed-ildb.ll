; RUN: opt %s -dxil-embed -dxil-globals -S -o - | FileCheck %s
; RUN: llc %s --filetype=obj -o %t.bc
; RUN: obj2yaml %t.bc | FileCheck %s --check-prefix=YAML
; RUN: llvm-objcopy --dump-section=ILDB=%t.ildb %t.bc
; RUN: llvm-objcopy --dump-section=DXIL=%t.dxil %t.bc
; RUN: llvm-dis %t.ildb -o - | FileCheck %s --check-prefix=ILDB-DIS
; RUN: llvm-dis %t.dxil -o - | FileCheck %s --check-prefix=DXIL-DIS

target triple = "dxil-unknown-shadermodel6.5-library"
; CHECK: target triple = "dxil-unknown-shadermodel6.5-library"

define i32 @add(i32 %a, i32 %b) {
  %sum = add i32 %a, %b
  ret i32 %sum
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!dx.source.contents = !{!5}
!dx.source.defines = !{!6}
!dx.source.mainFileName = !{!7}
!dx.source.args = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Some Compiler", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "hlsl.hlsl", directory: "/some-path")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"hlsl.hlsl", !"int add(int a, int b) { return a + b; }"}
!6 = !{}
!7 = !{!"hlsl.hlsl"}
!8 = !{!"-T", !"lib_6_5", !"-g", !"hlsl.hlsl"}

; Check that DXIL, ILDB and SRCI parts are emitted as a GV and used by the compiler.

; CHECK: @dx.ildb = private constant [[BC_TYPE:\[[0-9]+ x i8\]]] c"BC\C0\DE{{[^"]+}}", section "ILDB", align 4
; CHECK: @dx.dxil = private constant [[BC_TYPE:\[[0-9]+ x i8\]]] c"BC\C0\DE{{[^"]+}}", section "DXIL", align 4
; CHECK: @dx.srci = private constant {{\[[0-9]+ x i8\]}}
; CHECK: @llvm.compiler.used = appending global {{\[[0-9]+ x ptr\]}} [ptr @dx.ildb, ptr @dx.dxil

; This is using regex matches on some sizes, offsets and fields. These are all
; going to change as the DirectX backend continues to evolve and implement more
; features. Rather than extending this test to cover those future features, this
; test's matches are extremely fuzzy so that it won't break.

; YAML: --- !dxcontainer
; YAML-NEXT: Header:
; YAML-NEXT:   Hash:            [ 0x0, 0x0, 0x0,
; YAML:   Version:
; YAML-NEXT:     Major:           1
; YAML-NEXT:     Minor:           0
; YAML-NEXT:   FileSize:        [[#]]
; YAML-NEXT:   PartCount:       [[#]]
; YAML-NEXT:   PartOffsets:     [ {{[0-9, ]+}}
; YAML:   Parts:

; In verifying the DXIL and ILDB parts, this test captures the size of the part,
; and derives the program header and dxil size fields from the part's size.

; YAML:   - Name:            ILDB
; YAML-NEXT:     Size:            [[#ILDBSIZE:]]
; YAML-NEXT:     Program:
; YAML-NEXT:       MajorVersion:    6
; YAML-NEXT:       MinorVersion:    5
; YAML-NEXT:       ShaderKind:      6
; YAML-NEXT:       Size:            [[#div(ILDBSIZE,4)]]
; YAML-NEXT:       DXILMajorVersion: 1
; YAML-NEXT:       DXILMinorVersion: 5
; YAML-NEXT:       DXILSize:        [[#ILDBSIZE - 24]]
; YAML-NEXT:       DXIL:            [ 0x42, 0x43, 0xC0, 0xDE,
; YAML:   - Name:            DXIL
; YAML-NEXT:     Size:            [[#DXILSIZE:]]
; YAML-NEXT:     Program:
; YAML-NEXT:       MajorVersion:    6
; YAML-NEXT:       MinorVersion:    5
; YAML-NEXT:       ShaderKind:      6
; YAML-NEXT:       Size:            [[#div(DXILSIZE,4)]]
; YAML-NEXT:       DXILMajorVersion: 1
; YAML-NEXT:       DXILMinorVersion: 5
; YAML-NEXT:       DXILSize:        [[#DXILSIZE - 24]]
; YAML-NEXT:       DXIL:            [ 0x42, 0x43, 0xC0, 0xDE,

; Check that despite dx.source is stripped from DXIL, SRCI is still emitted:
; YAML:   - Name:            SRCI

; Check that ILDB has the debug info, and DXIL does not:

; ILDB-DIS: define i32 @add(i32 %a, i32 %b)
; ILDB-DIS: !llvm.dbg.cu
; ILDB-DIS: !DICompileUnit
; ILDB-DIS: !DIFile
; ILDB-DIS: !"Dwarf Version"
; ILDB-DIS: !"Debug Info Version"

; DXIL-DIS: define i32 @add(i32 %a, i32 %b)
; DXIL-DIS-NOT: !llvm.dbg.cu
; DXIL-DIS-NOT: !dx.source
; DXIL-DIS-NOT: !DICompileUnit
; DXIL-DIS-NOT: !DIFile
; DXIL-DIS-NOT: !"Dwarf Version"
; DXIL-DIS-NOT: !"Debug Info Version"
; DXIL-DIS-NOT: "hlsl.hlsl"
