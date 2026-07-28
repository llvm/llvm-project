// RUN: rm -rf %t && split-file %s %t
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %p/../../include %t/bad-alias.td 2>&1 \
// RUN:   | FileCheck %t/bad-alias.td -DFILE=%t/bad-alias.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %p/../../include %t/dup-processor.td 2>&1 \
// RUN:   | FileCheck %t/dup-processor.td -DFILE=%t/dup-processor.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %p/../../include %t/alias-shadows-processor.td 2>&1 \
// RUN:   | FileCheck %t/alias-shadows-processor.td -DFILE=%t/alias-shadows-processor.td --implicit-check-not="error:"

// Verify the validation performed by the -gen-amdgpu-target-def backend.

//--- bad-alias.td
include "llvm/Target/Target.td"
def MyTarget : Target;
class AMDGPUArchFeature<string spelling> { string Spelling = spelling; }
class AMDGPUGPUInfo { list<AMDGPUArchFeature> ArchFeatures = []; }

def : ProcessorModel<"gfx900", NoSchedModel, []>, AMDGPUGPUInfo;
// An alias must resolve to a canonical AMDGPU GPU.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: ProcessorAlias 'foo' aliases 'gfx-missing' which is not a canonical AMDGPU GPU
def : ProcessorAlias<"foo", "gfx-missing">;

//--- dup-processor.td
include "llvm/Target/Target.td"
def MyTarget : Target;
class AMDGPUArchFeature<string spelling> { string Spelling = spelling; }
class AMDGPUGPUInfo { list<AMDGPUArchFeature> ArchFeatures = []; }

def DupA : ProcessorModel<"gfx900", NoSchedModel, []>, AMDGPUGPUInfo;
// A canonical GPU name must be unique.
// CHECK: [[FILE]]:[[#@LINE+1]]:5: error: duplicate AMDGPU processor name 'gfx900'
def DupB : ProcessorModel<"gfx900", NoSchedModel, []>, AMDGPUGPUInfo;

//--- alias-shadows-processor.td
include "llvm/Target/Target.td"
def MyTarget : Target;
class AMDGPUArchFeature<string spelling> { string Spelling = spelling; }
class AMDGPUGPUInfo { list<AMDGPUArchFeature> ArchFeatures = []; }

def : ProcessorModel<"gfx900", NoSchedModel, []>, AMDGPUGPUInfo;
// An alias name must not collide with a canonical GPU name.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: duplicate AMDGPU processor name 'gfx900'
def : ProcessorAlias<"gfx900", "gfx900">;
