; xfail this test on hexagon because at O2, instructions are bundled in packets
; and DW_OP_lit13 is correctly omitted.
; XFAIL: target=hexagon-{{.*}}

; This file contains the dwarf-version=2 tests extracted from incorrect-variable-debugloc1.ll.
; DWARF v2 is incompatible with 64-bit XCOFF/AIX (requires DWARF64 format which needs DWARF v3+).

; UNSUPPORTED: target=powerpc64{{.*}}-aix{{.*}}

; RUN: %llc_dwarf -O2  -dwarf-version 2 -filetype=obj < %S/incorrect-variable-debugloc1.ll | llvm-dwarfdump - | FileCheck %S/incorrect-variable-debugloc1.ll  --check-prefix=DWARF23
