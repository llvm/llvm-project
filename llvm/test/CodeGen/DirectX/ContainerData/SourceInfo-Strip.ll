; Compare source info emission with and without --dx-source-in-debug-module flag.

; RUN: llc %s --filetype=obj -o %t.dxbc --dx-pdb-path=%t.pdb
; RUN: llvm-pdbutil export --stream=5 --out=%t.pdb.dxbc %t.pdb
; RUN: obj2yaml %t.pdb.dxbc | FileCheck %s --check-prefix=DXC
; RUN: llvm-objcopy --dump-section=DXIL=%t.dxil.bc %t.dxbc
; RUN: llvm-dis %t.dxil.bc -o - | FileCheck %s --check-prefix=DXIL-DIS
; RUN: llvm-objcopy --dump-section=ILDB=%t.ildb.bc %t.pdb.dxbc
; RUN: llvm-dis %t.ildb.bc -o - | FileCheck %s --check-prefix=ILDB-DIS

; RUN: llc %s --filetype=obj -o %t.dxbc --dx-source-in-debug-module --dx-pdb-path=%t.pdb
; RUN: llvm-pdbutil export --stream=5 --out=%t.pdb.dxbc %t.pdb
; RUN: obj2yaml %t.pdb.dxbc | FileCheck %s --check-prefix=DXC-SOURCE
; RUN: llvm-objcopy --dump-section=DXIL=%t.dxil.bc %t.dxbc
; RUN: llvm-dis %t.dxil.bc -o - | FileCheck %s --check-prefix=DXIL-SOURCE-DIS
; RUN: llvm-objcopy --dump-section=ILDB=%t.ildb.bc %t.pdb.dxbc
; RUN: llvm-dis %t.ildb.bc -o - | FileCheck %s --check-prefix=ILDB-SOURCE-DIS

; Without the flag, dx.source should be stripped away from DXIL, and replaced
; with dummy metadata in ILDB.
; DXIL-DIS-NOT: dx.source
; ILDB-DIS: !dx.source.contents = !{![[CONTENTS:[0-9]+]]}
; ILDB-DIS: !dx.source.defines = !{![[EMPTY_ARR:[0-9]+]]}
; ILDB-DIS: !dx.source.mainFileName = !{![[MAIN:[0-9]+]]}
; ILDB-DIS: !dx.source.args = !{![[EMPTY_ARR]]}
; ILDB-DIS: ![[CONTENTS]] = !{!"", !""}
; ILDB-DIS: ![[EMPTY_ARR]] = !{}
; ILDB-DIS: ![[MAIN]] = !{!""}

; Without the flag, SRCI should be emitted.
; DXC:      - Name:            SRCI
; DXC:        SourceInfo:
; DXC-NEXT:     Header:
; DXC:            SectionCount:    3
; DXC-NEXT:     Names:
; DXC-NEXT:       SectionHeader:
; DXC:              Type:            SourceNames
; DXC-NEXT:       Header:
; DXC:              Count:           3
; DXC:            Entries:
; DXC:                FileName:        'C:\dx-source-metadata.hlsl'
; DXC:                FileName:        'C:\a.hlsl'
; DXC:                FileName:        'C:\b.hlsl'
; DXC-NEXT:     Contents:
; DXC-NEXT:       SectionHeader:
; DXC:              Type:            SourceContents
; DXC-NEXT:       Header:
; DXC:              Count:           3
; DXC:            Entries:
; DXC:                FileContent:     "#include \"a.hlsl\"\n#include \"b.hlsl\"\n\nfloat foo(float a, float b) {\n return a + b;\n}\n"
; DXC:                FileContent:     "#include \"b.hlsl\"\n"
; DXC:                FileContent:     "#include <c.hlsl>\n"
; DXC:          Args:
; DXC:            SectionHeader:
; DXC:            Header:
; DXC:              Count:           5
; DXC:            Args:
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

; With the flag, dx.source should be stripped away from DXIL, and kept untouched in ILDB.
; DXIL-SOURCE-DIS-NOT: dx.source
; ILDB-SOURCE-DIS: !dx.source.args = !{![[ARGS:[0-9]+]]}
; ILDB-SOURCE-DIS: !dx.source.contents = !{![[FILE1:[0-9]+]], ![[FILE2:[0-9]+]], ![[FILE3:[0-9]+]]}
; ILDB-SOURCE-DIS: !dx.source.mainFileName = !{![[MAIN:[0-9]+]]}
; ILDB-SOURCE-DIS: !dx.source.defines = !{![[DEFINES:[0-9]+]]}
; ILDB-SOURCE-DIS: ![[FILE1]] = !{!"C:\\dx-source-metadata.hlsl",
; ILDB-SOURCE-DIS: ![[FILE2]] = !{!"C:\\a.hlsl"
; ILDB-SOURCE-DIS: ![[FILE3]] = !{!"C:\\b.hlsl"
; ILDB-SOURCE-DIS: ![[MAIN]] = !{!"C:\\dx-source-metadata.hlsl"}
; ILDB-SOURCE-DIS: ![[DEFINES]] = !{!"USER_DEF0=42", !"USER_DEF1=43"}

; With the flag, SRCI should not be emitted.
; DXC-SOURCE-NOT: - Name: SRCI

target triple = "dxilv1.3-pc-shadermodel6.3-library"

define float @_Z3fooff(float %a, float %b) {
entry:
  %add = fadd float %a, %b
  ret float %add
}

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!6, !7}

!dx.source.args = !{!0}
!dx.source.contents = !{!1, !2, !3}
!dx.source.mainFileName = !{!8}
!dx.source.defines = !{!9}

!0 = !{!"-g", !"-Tlib_6_3", !"-DUSER_DEF0=42", !"-DUSER_DEF1=43", !"C:\\\\dx-source-metadata.hlsl"}
!1 = !{!"C:\\dx-source-metadata.hlsl", !"#include \22a.hlsl\22\0A#include \22b.hlsl\22\0A\0Afloat foo(float a, float b) {\0A  return a + b;\0A}\0A"}
!2 = !{!"C:\\a.hlsl", !"#include \22b.hlsl\22\0A"}
!3 = !{!"C:\\b.hlsl", !"#include <c.hlsl>\0A"}
!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !5, emissionKind: FullDebug)
!5 = !DIFile(filename: "dx-source-metadata.hlsl", directory: "C:\\")
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"C:\\dx-source-metadata.hlsl"}
!9 = !{!"USER_DEF0=42", !"USER_DEF1=43"}
