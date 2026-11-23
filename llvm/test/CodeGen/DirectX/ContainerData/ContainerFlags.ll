;; Check that --dx-embed-debug is enabled by default when debug info is present
; RUN: llc %S/Inputs/SourceInfo.ll --filetype=obj -o %t.cso
; RUN: obj2yaml %t.cso | FileCheck %s --check-prefix=DEFAULT-EMBED
; DEFAULT-EMBED: Parts:
; DEFAULT-EMBED:   - Name:            ILDB
; DEFAULT-EMBED:   - Name:            ILDN

;; Check that debug info is not embedded if only the PDB ouput is specified
; RUN: llc %S/Inputs/SourceInfo.ll --filetype=obj --dx-pdb-path=%t.pdb -o %t.cso
; RUN: obj2yaml %t.cso | FileCheck %s --check-prefix=NO-EMBED
; NO-EMBED: Parts:
; NO-EMBED-NOT:   - Name:            ILDB
; NO-EMBED:       - Name:            ILDN

;; Check that debug info is both embedded and output to the PDB
;; if both options are specified
; RUN: llc %S/Inputs/SourceInfo.ll --filetype=obj --dx-embed-debug --dx-pdb-path=%t.pdb -o %t.cso
; RUN: obj2yaml %t.cso | FileCheck %s --check-prefix=EMBED
; EMBED: Parts:
; EMBED:   - Name:            ILDB
; EMBED:   - Name:            ILDN
; RUN: llvm-pdbutil pdb2yaml --dxcontainer %t.pdb | FileCheck %s --check-prefix=PDB
; Check that PDB file contains only debug-info relevant parts.
; PDB:     Parts:
; PDB-DAG:   - Name:            ILDB
; PDB-DAG:   - Name:            ILDN

;; Check that --dx-strip-debug strips ILDB from DXContainer, but still keeps ILDN part
; RUN: llc %S/Inputs/SourceInfo.ll --filetype=obj --dx-strip-debug -o %t.cso
; RUN: obj2yaml %t.cso | FileCheck %s --check-prefix=STRIP --implicit-check-not ILDB
; STRIP:     Parts:
; STRIP-DAG:   - Name:            ILDN

;; Check that --dx-strip-debug is ignored when provided along with --dx-embed-debug
; RUN: llc %S/Inputs/SourceInfo.ll --filetype=obj --dx-strip-debug --dx-embed-debug -o %t.cso
; RUN: obj2yaml %t.cso | FileCheck %s --check-prefix=STRIP-EMBED
; STRIP-EMBED:     Parts:
; STRIP-EMBED-DAG:   - Name:            ILDB
; STRIP-EMBED-DAG:   - Name:            ILDN

;; Check errors when trying to output debug info with no debug info present
; RUN: not llc %s --filetype=obj --dx-embed-debug -o %t.cso 2>&1 | FileCheck %s --check-prefix=ERROR-NODBG
; ERROR-NODBG: Missing debug info for embedding into the container
; RUN: not llc %s --filetype=obj --dx-pdb-path=%t.pdb -o %t.cso 2>&1 | FileCheck %s --check-prefix=ERROR-NODBG-PDB
; ERROR-NODBG-PDB: Missing debug info for writing to the PDB file

;; Check that slim debug (-dx-slim-debug) omits ILDB from container and companion PDB
; RUN: llc %S/Inputs/SourceInfo.ll --filetype=obj -dx-slim-debug --dx-pdb-path=%t.pdb -o %t.cso
; RUN: obj2yaml %t.cso | FileCheck %s --check-prefix=SLIM --implicit-check-not=ILDB
; RUN: llvm-pdbutil pdb2yaml --dxcontainer %t.pdb | FileCheck %s --check-prefix=SLIM --implicit-check-not=ILDB

; SLIM: Parts:
; SLIM:   - Name:             HASH
; SLIM:   - Name:             ILDN

;; Check that /Zss can still hash from ILDB when /Zs omits it from output
; RUN: llc %S/Inputs/SourceInfo.ll --filetype=obj -dx-slim-debug -dx-Zss -o %t.slim.cso
; RUN: obj2yaml %t.slim.cso | FileCheck %s --check-prefix=SLIM-ZSS --implicit-check-not=ILDB
; SLIM-ZSS:   - Name:            HASH
; SLIM-ZSS:     Hash:
; SLIM-ZSS:       IncludesSource:  true

;; Check that slim debug and embed debug are mutually exclusive
; RUN: not llc %S/Inputs/SourceInfo.ll --filetype=obj -dx-slim-debug --dx-embed-debug -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR-ZS-EMBED
; ERROR-ZS-EMBED: /Qembed_debug is not compatible with /Zs

target triple = "dxil-unknown-shadermodel6.5-library"

define i32 @foo(i32 %a) {
  ret i32 %a
}
