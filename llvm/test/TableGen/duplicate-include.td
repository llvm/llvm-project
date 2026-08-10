// RUN: llvm-tblgen -I %p %s | FileCheck %s

// CHECK: def InInclude

include "duplicate-include.inc"
include "duplicate-include.inc"

