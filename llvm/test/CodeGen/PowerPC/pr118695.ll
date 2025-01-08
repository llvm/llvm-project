; RUN: llc < %s -verify-machineinstrs -mtriple=powerpc-aix- -mcpu=pwr10 | FileCheck %s

; CHECK:      # %bb.0:                                # %bb
; CHECK-NEXT:   lwz 3, L..C0(2)                         # @dvar
; CHECK-NEXT:   plxv 0, -152758(3), 0
; CHECK-NEXT:   stxv 0, 0(3)
; CHECK-NEXT:   blr

%0 = type <{ double }>
@dvar = external global [2352637 x %0]

define void @Test() {
bb:
	%i9 = load <2 x double>, ptr getelementptr inbounds (i8, ptr @dvar, i64 -152758), align 8
	store <2 x double> %i9, ptr @dvar, align 8
	ret void
}
