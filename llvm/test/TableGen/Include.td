// RUN: llvm-tblgen -I %p %s
// XFAIL: vg_leak
def BeforeInclude;

include "Include.inc"

def AfterInclude;

