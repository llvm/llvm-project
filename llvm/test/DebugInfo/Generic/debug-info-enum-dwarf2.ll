; Test enumeration representation in DWARF debug info for DWARF v2:
; * test value representation for each possible underlying integer type
; * test the integer type is as expected
; * test the DW_AT_enum_class attribute is present (resp. absent) as expected.
; * test that DW_AT_type is present for v2 when strict DWARF is not enabled.

; This file contains the dwarf-version=2 tests extracted from debug-info-enum.ll.
; DWARF v2 is incompatible with 64-bit XCOFF/AIX (requires DWARF64 format which needs DWARF v3+).

; UNSUPPORTED: target=powerpc64{{.*}}-aix{{.*}}

; RUN: llc -debugger-tune=gdb -dwarf-version=2 -filetype=obj -o %t.o < %S/debug-info-enum.ll
; RUN: llvm-dwarfdump -debug-info %t.o | FileCheck %S/debug-info-enum.ll --check-prefixes=CHECK,CHECK-TYPE
; RUN: llc -debugger-tune=gdb -dwarf-version=2 -strict-dwarf=true -filetype=obj -o %t.o < %S/debug-info-enum.ll
; RUN: llvm-dwarfdump -debug-info %t.o | FileCheck %S/debug-info-enum.ll --check-prefixes=CHECK,CHECK-DW2-STRICT
