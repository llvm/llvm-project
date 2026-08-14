// RUN: rm -rf %t && split-file %s %t
// RUN: not llvm-tblgen -gen-subtarget -I %p/../../include %t/non-existent.td 2>&1 \
// RUN:   | FileCheck %t/non-existent.td -DFILE=%t/non-existent.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-subtarget -I %p/../../include %t/dup-processor.td 2>&1 \
// RUN:   | FileCheck %t/dup-processor.td -DFILE=%t/dup-processor.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-subtarget -I %p/../../include %t/dup-alias.td 2>&1 \
// RUN:   | FileCheck %t/dup-alias.td -DFILE=%t/dup-alias.td --implicit-check-not="error:"

// Verify the ProcessorAlias validation performed by the subtarget emitter.

//--- non-existent.td
include "llvm/Target/Target.td"
def MyTarget : Target;
def ProcA : ProcessorModel<"cpu-a", NoSchedModel, []>;
// An alias must resolve to an existing processor.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: Alias 'bad-alias' references a non-existent Processor 'cpu-missing'
def : ProcessorAlias<"bad-alias", "cpu-missing">;

//--- dup-processor.td
include "llvm/Target/Target.td"
def MyTarget : Target;
def ProcA : ProcessorModel<"cpu-a", NoSchedModel, []>;
// An alias name must not collide with a real processor.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: Alias 'cpu-a' duplicates an existing Processor
def : ProcessorAlias<"cpu-a", "cpu-a">;

//--- dup-alias.td
include "llvm/Target/Target.td"
def MyTarget : Target;
def ProcA : ProcessorModel<"cpu-a", NoSchedModel, []>;
def DupA : ProcessorAlias<"dup", "cpu-a">;
// Two aliases must not share the same name.
// CHECK: [[FILE]]:[[#@LINE+1]]:5: error: Alias 'dup' duplicates an existing alias
def DupB : ProcessorAlias<"dup", "cpu-a">;
