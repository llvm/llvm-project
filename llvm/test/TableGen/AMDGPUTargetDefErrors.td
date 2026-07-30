// RUN: rm -rf %t && split-file %s %t
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %p/../../include %t/bad-alias.td 2>&1 \
// RUN:   | FileCheck %t/bad-alias.td -DFILE=%t/bad-alias.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %p/../../include %t/dup-processor.td 2>&1 \
// RUN:   | FileCheck %t/dup-processor.td -DFILE=%t/dup-processor.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %p/../../include %t/alias-shadows-processor.td 2>&1 \
// RUN:   | FileCheck %t/alias-shadows-processor.td -DFILE=%t/alias-shadows-processor.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %p/../../include %t/bad-isa-version.td 2>&1 \
// RUN:   | FileCheck %t/bad-isa-version.td -DFILE=%t/bad-isa-version.td --implicit-check-not="error:"

// Verify the validation performed by the -gen-amdgpu-target-def backend.

//--- bad-alias.td
include "llvm/Target/Target.td"
def MyTarget : Target;
class AMDGPUArchFeature<string spelling> { string Spelling = spelling; }
class AMDGPUGPUInfo<list<int> isa = []> {
  list<AMDGPUArchFeature> ArchFeatures = [];
  list<int> IsaVersion = isa;
  list<Processor> CoveredGPUs = [];
  bit IsPseudoTarget = false;
}

def : ProcessorModel<"gfx900", NoSchedModel, []>, AMDGPUGPUInfo<[9, 0, 0]>;
// An alias must resolve to a canonical AMDGPU GPU.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: ProcessorAlias 'foo' aliases 'gfx-missing' which is not a canonical AMDGPU GPU
def : ProcessorAlias<"foo", "gfx-missing">;

//--- dup-processor.td
include "llvm/Target/Target.td"
def MyTarget : Target;
class AMDGPUArchFeature<string spelling> { string Spelling = spelling; }
class AMDGPUGPUInfo<list<int> isa = []> {
  list<AMDGPUArchFeature> ArchFeatures = [];
  list<int> IsaVersion = isa;
  list<Processor> CoveredGPUs = [];
  bit IsPseudoTarget = false;
}

def DupA : ProcessorModel<"gfx900", NoSchedModel, []>, AMDGPUGPUInfo<[9, 0, 0]>;
// A canonical GPU name must be unique.
// CHECK: [[FILE]]:[[#@LINE+1]]:5: error: duplicate AMDGPU processor name 'gfx900'
def DupB : ProcessorModel<"gfx900", NoSchedModel, []>, AMDGPUGPUInfo<[9, 0, 0]>;

//--- alias-shadows-processor.td
include "llvm/Target/Target.td"
def MyTarget : Target;
class AMDGPUArchFeature<string spelling> { string Spelling = spelling; }
class AMDGPUGPUInfo<list<int> isa = []> {
  list<AMDGPUArchFeature> ArchFeatures = [];
  list<int> IsaVersion = isa;
  list<Processor> CoveredGPUs = [];
  bit IsPseudoTarget = false;
}

def : ProcessorModel<"gfx900", NoSchedModel, []>, AMDGPUGPUInfo<[9, 0, 0]>;
// An alias name must not collide with a canonical GPU name.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: duplicate AMDGPU processor name 'gfx900'
def : ProcessorAlias<"gfx900", "gfx900">;

//--- bad-isa-version.td
include "llvm/Target/Target.td"
def MyTarget : Target;
class AMDGPUArchFeature<string spelling> { string Spelling = spelling; }
class AMDGPUGPUInfo<list<int> isa = []> {
  list<AMDGPUArchFeature> ArchFeatures = [];
  list<int> IsaVersion = isa;
  list<Processor> CoveredGPUs = [];
  bit IsPseudoTarget = false;
}
// A malformed IsaVersion is reported (not asserted), so this stays a clean
// diagnostic in release builds.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: GPU 'gfx900' must have a 3-element [major, minor, stepping] IsaVersion
def : ProcessorModel<"gfx900", NoSchedModel, []>, AMDGPUGPUInfo<[9, 0]>;
