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

// A variant CPU: its subarch, triple name, and family come from the
// spelling / ISA version, not its own name.
def GFX888_AMAZING : ProcessorModel<"gfx888-amazing", NoSchedModel, []>,
    AMDGPUGPUInfo<[8, 8, 8]> {
  let SubArchSpelling = "8.88a";
}

// A concrete GPU whose stepping is a hex digit above 9: it is spelled as a
// single lowercase hex character in both the device and subarch names.
def GFX88F : ProcessorModel<"gfx88f", NoSchedModel, []>,
    AMDGPUGPUInfo<[8, 8, 0xF]>;

// The variant contributes a distinct GPUKind.
// CHECK: GK_GFX888
// CHECK: GK_GFX888_AMAZING
// CHECK: GK_GFX88F

// The variant is its own major subarch, so it produces no override entry.
// CHECK: AMDGPUMajorSubArchEntry, 0>{{ }}AMDGPUMajorSubArch

// The name pool holds the variant's name and its gfx8 family.
// CHECK: "gfx888\0"
// CHECK: "gfx8\0"
// CHECK: "gfx888-amazing\0"
// CHECK: "amdgpu8.88a\0"

// The stepping is spelled as a single lowercase hex digit in the triple name.
// CHECK: "amdgpu8.8f\0"

// The GPU table: both the base GPU and the variant share the gfx8 family; the
// variant uses its own AMDGPUSubArch888A from the spelling.
// CHECK: {[[#]], Triple::AMDGPUSubArch888, {{.*}}, {8, 8, 8}, [[FAM:[0-9]+]], [[#]], [[#]], [[#]]},
// CHECK: {[[#]], Triple::AMDGPUSubArch888A, {{.*}}, {8, 8, 8}, [[FAM]], [[#]], [[#]], [[#]]},
// CHECK: {[[#]], Triple::AMDGPUSubArch88F, {{.*}}, {8, 8, 15}, [[#]], [[#]], [[#]], [[#]]},

// The subarch-name table maps the variant's own subarch to its triple name.
// CHECK: {Triple::AMDGPUSubArch888A, [[#]], [[#]]},
