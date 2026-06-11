; RUN: llc %S/Inputs/SourceInfo.ll --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

; DXC:      - Name:            SRCI
; DXC:        SourceInfo:
; DXC:          Header:
; DXC:            Flags:           0
; DXC:            SectionCount:    3
; DXC:          Names:
; DXC:            SectionHeader:
; DXC:            Header:
; DXC:            Entries:
; DXC:          Contents:
; DXC:            SectionHeader:
; DXC:            Header:
; DXC:            Entries:
; DXC:          Args:
; DXC-NEXT:       SectionHeader:
; DXC-NEXT:         AlignedSizeInBytes: 96
; DXC-NEXT:         Flags:           0
; DXC-NEXT:         Type:            Args
; DXC-NEXT:       Header:
; DXC-NEXT:         Flags:           0
; DXC-NEXT:         SizeInBytes:     76
; DXC-NEXT:         Count:           5
; DXC-NEXT:       Args:
; DXC-NEXT:         - Arg:             '-g'
; DXC-NEXT:           Value:           ''
; DXC-NEXT:         - Arg:             '-Tlib_6_3'
; DXC-NEXT:           Value:           ''
; DXC-NEXT:         - Arg:             '-DUSER_DEF0=42'
; DXC-NEXT:           Value:           ''
; DXC-NEXT:         - Arg:             '-DUSER_DEF1=43'
; DXC-NEXT:           Value:           ''
; DXC-NEXT:         - Arg:             'C:\\dx-source-metadata.hlsl'
; DXC-NEXT:           Value:           ''
