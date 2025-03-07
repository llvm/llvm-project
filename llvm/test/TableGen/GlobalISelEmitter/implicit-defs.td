// RUN: llvm-tblgen -gen-global-isel -warn-on-skipped-patterns -I %p/../../../include -I %p/../Common %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not="Skipped pattern"

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// CHECK: Skipped pattern: Pattern defines a physical register
let Uses = [B0], Defs = [B0] in
def tst1 : I<(outs), (ins), [(set B0, (add B0, 1))]>;

// CHECK: Skipped pattern: Src pattern result has 1 def(s) without the HasNoUse predicate set to true but Dst MI has no def
let Uses = [B0] in
def tst2 : I<(outs), (ins), [(set B0, (add B0, 1))]>;
