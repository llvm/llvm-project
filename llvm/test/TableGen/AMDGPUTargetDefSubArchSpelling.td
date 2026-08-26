// RUN: llvm-tblgen -gen-amdgpu-target-def -I %p/../../include %s 2>&1 | FileCheck %s

// Verify the -gen-amdgpu-target-def backend's handling of SubArchSpelling.

include "llvm/Target/Target.td"
def MyTarget : Target;
class AMDGPUArchFeature<string spelling> { string Spelling = spelling; }
class AMDGPUGPUInfo<list<int> isa = []> {
  list<AMDGPUArchFeature> ArchFeatures = [];
  list<int> IsaVersion = isa;
  list<Processor> CoveredGPUs = [];
  bit IsPseudoTarget = false;
  string SubArchSpelling;
}

def GFX888 : ProcessorModel<"gfx888", NoSchedModel, []>, AMDGPUGPUInfo<[8, 8, 8]>;

// A variant CPU: its subarch, triple name, family, and base name come from the
// spelling / ISA version, not its own name.
def GFX888_AMAZING : ProcessorModel<"gfx888-amazing", NoSchedModel, []>,
    AMDGPUGPUInfo<[8, 8, 8]> {
  let SubArchSpelling = "8.88a";
}

// A variant whose stepping is wider than a hex digit: only the low nibble is
// named.
def GFX888_WIDE : ProcessorModel<"gfx888-wide", NoSchedModel, []>,
    AMDGPUGPUInfo<[8, 8, 0xfff8]> {
  let SubArchSpelling = "8.88w";
}

// The variant contributes a distinct GPUKind.
// CHECK: GK_GFX888
// CHECK: GK_GFX888_AMAZING

// The variant is its own major subarch, so it produces no override entry.
// CHECK: AMDGPUMajorSubArchEntry, 0>{{ }}AMDGPUMajorSubArch

// The name pool holds the variant's name, its gfx8 family, and its base name.
// CHECK: "gfx888\0"
// CHECK: "gfx8\0"
// CHECK: "gfx888-amazing\0"
// CHECK: "gfx888-wide\0"
// CHECK: "amdgpu8.88a\0"
// CHECK: "amdgpu8.88w\0"

// The GPU table: the base GPU has no base name (offset 0); the variant uses
// AMDGPUSubArch888A and records "gfx888" as its base name.
// CHECK: {[[#]], Triple::AMDGPUSubArch888, {{.*}}, {8, 8, 8}, [[FAM:[0-9]+]], 0},
// CHECK: {[[#]], Triple::AMDGPUSubArch888A, {{.*}}, {8, 8, 8}, [[FAM]], [[#]]},
// CHECK: {[[#]], Triple::AMDGPUSubArch888W, {{.*}}, {8, 8, 65528}, [[FAM]], [[#]]},

// The subarch-name table maps the variant's own subarch to its triple name.
// CHECK: {Triple::AMDGPUSubArch888A, [[#]], [[#]]},
