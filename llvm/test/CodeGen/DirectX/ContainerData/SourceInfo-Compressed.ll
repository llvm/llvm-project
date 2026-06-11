; RUN: opt %S/Inputs/SourceInfo.ll -dxil-embed -dxil-globals -S -o - | FileCheck %s
; RUN: llc %S/Inputs/SourceInfo.ll --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC
; REQUIRES: zlib

; CHECK: @dx.srci = private constant [348 x i8] c"{{.*}}", section "SRCI", align 4

; DXC:      - Name:            SRCI
; DXC-NEXT:   Size:            348
; DXC-NEXT:   SourceInfo:
; DXC-NEXT:     Header:
; DXC-NEXT:       AlignedSizeInBytes: 348
; DXC-NEXT:       Flags:           0
; DXC-NEXT:       SectionCount:    3
; DXC-NEXT:     Names:
; DXC-NEXT:       SectionHeader:
; DXC-NEXT:         AlignedSizeInBytes: 120
; DXC-NEXT:         Flags:           0
; DXC-NEXT:         Type:            SourceNames
; DXC-NEXT:       Header:
; DXC-NEXT:         Flags:           0
; DXC-NEXT:         Count:           3
; DXC-NEXT:         EntriesSizeInBytes: 100
; DXC-NEXT:       Entries:
; DXC-NEXT:         - AlignedSizeInBytes: 44
; DXC-NEXT:           Flags:           0
; DXC-NEXT:           NameSizeInBytes: 27
; DXC-NEXT:           ContentSizeInBytes: 86
; DXC-NEXT:           FileName:        'C:\dx-source-metadata.hlsl'
; DXC-NEXT:         - AlignedSizeInBytes: 28
; DXC-NEXT:           Flags:           0
; DXC-NEXT:           NameSizeInBytes: 10
; DXC-NEXT:           ContentSizeInBytes: 19
; DXC-NEXT:           FileName:        'C:\a.hlsl'
; DXC-NEXT:         - AlignedSizeInBytes: 28
; DXC-NEXT:           Flags:           0
; DXC-NEXT:           NameSizeInBytes: 10
; DXC-NEXT:           ContentSizeInBytes: 19
; DXC-NEXT:           FileName:        'C:\b.hlsl'
; DXC-NEXT:     Contents:
; DXC-NEXT:       SectionHeader:
; DXC-NEXT:         AlignedSizeInBytes: 124
; DXC-NEXT:         Flags:           0
; DXC-NEXT:         Type:            SourceContents
; DXC-NEXT:       Header:
; DXC-NEXT:         AlignedSizeInBytes: 116
; DXC-NEXT:         Flags:           0
; DXC-NEXT:         Type:            Zlib
; DXC-NEXT:         EntriesSizeInBytes: 96
; DXC-NEXT:         UncompressedEntriesSizeInBytes: 164
; DXC-NEXT:         Count:           3
; DXC-NEXT:       Entries:
; DXC-NEXT:         - AlignedSizeInBytes: 100
; DXC-NEXT:           Flags:           0
; DXC-NEXT:           ContentSizeInBytes: 86
; DXC-NEXT:           FileContent:     "#include \"a.hlsl\"\n#include \"b.hlsl\"\n\nfloat foo(float a, float b) {\n return a + b;\n}\n"
; DXC-NEXT:         - AlignedSizeInBytes: 32
; DXC-NEXT:           Flags:           0
; DXC-NEXT:           ContentSizeInBytes: 19
; DXC-NEXT:           FileContent:     "#include \"b.hlsl\"\n"
; DXC-NEXT:         - AlignedSizeInBytes: 32
; DXC-NEXT:           Flags:           0
; DXC-NEXT:           ContentSizeInBytes: 19
; DXC-NEXT:           FileContent:     "#include <c.hlsl>\n"
; DXC-NEXT:     Args:
; DXC:            SectionHeader:
; DXC:            Header:
; DXC:            Args:
