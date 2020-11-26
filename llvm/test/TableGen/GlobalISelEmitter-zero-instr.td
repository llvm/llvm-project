// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false -I %p/../../include -I %p/Common %s -o /dev/null --warn-on-skipped-patterns 2>&1 < %s 2>&1 | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// CHECK: warning: Skipped pattern: Dst pattern root isn't a known leaf
def : Pat<(zext (i16 (trunc i32:$src))),
          (i32 $src)>;
