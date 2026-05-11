// RUN: llvm-tblgen -warn-on-skipped-patterns -gen-global-isel -I %p/../../../include %s -I %p/../Common -o /dev/null 2>&1 | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// A REG_SEQUENCE whose super register class has no subclass containing the
// requested subreg index used to crash the GlobalISel emitter on an empty
// std::optional dereference. It now skips the pattern with failedImport.

def bogus_sub : SubRegIndex<32>;

// CHECK: warning: Skipped pattern: REG_SEQUENCE subreg index is incompatible with inferred reg class
def : Pat<(i32 (add GPR32:$a, GPR32:$b)),
          (REG_SEQUENCE GPR32, GPR32:$a, bogus_sub, GPR32:$b, bogus_sub)>;
