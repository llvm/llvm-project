; Check that use of the AIX/XCOFF related classes with ppc64le would
; cause llc to die with an appropriate message instead of proceeding
; with an invalid state.

; RUN: not --crash llc < %s -mtriple powerpc64le--aix-xcoff 2>&1 \
; RUN:   | FileCheck --check-prefix=AIXXCOFF %s
; AIXXCOFF: ERROR: XCOFF is not supported for little-endian

; RUN: not --crash llc < %s -mtriple powerpc64le--aix-macho 2>&1 \
; RUN:   | FileCheck --check-prefix=AIXMACHO %s
; AIXMACHO: ERROR: cannot create AIX PPC Assembly Printer for a little-endian target

define i32 @a() { ret i32 0 }
