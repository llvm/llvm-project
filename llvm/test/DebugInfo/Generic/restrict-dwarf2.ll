; This file contains the dwarf-version=2 tests extracted from restrict.ll.
; DWARF v2 is incompatible with 64-bit XCOFF/AIX (requires DWARF64 format which needs DWARF v3+).

; UNSUPPORTED: target=powerpc64{{.*}}-aix{{.*}}

; RUN: %llc_dwarf -dwarf-version=2 -O0 -filetype=obj < %S/restrict.ll | llvm-dwarfdump -debug-info - | FileCheck --check-prefix=CHECK --check-prefix=V2 %S/restrict.ll
