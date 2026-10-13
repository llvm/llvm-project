// RUN: split-file %s %t
// RUN: llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../lib/Target/AMDGPU %t/valid.td \
// RUN:   | FileCheck %t/valid.td
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../lib/Target/AMDGPU %t/missing-any.td 2>&1 \
// RUN:   | FileCheck %t/missing-any.td -DFILE=%t/missing-any.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../lib/Target/AMDGPU %t/missing-all.td 2>&1 \
// RUN:   | FileCheck %t/missing-all.td -DFILE=%t/missing-all.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../lib/Target/AMDGPU %t/implied-all.td 2>&1 \
// RUN:   | FileCheck %t/implied-all.td -DFILE=%t/implied-all.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../lib/Target/AMDGPU %t/missing-private.td 2>&1 \
// RUN:   | FileCheck %t/missing-private.td -DFILE=%t/missing-private.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../lib/Target/AMDGPU %t/implied-private.td 2>&1 \
// RUN:   | FileCheck %t/implied-private.td -DFILE=%t/implied-private.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../lib/Target/AMDGPU %t/numeric-mismatch.td 2>&1 \
// RUN:   | FileCheck %t/numeric-mismatch.td -DFILE=%t/numeric-mismatch.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../lib/Target/AMDGPU %t/empty-members.td 2>&1 \
// RUN:   | FileCheck %t/empty-members.td -DFILE=%t/empty-members.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../lib/Target/AMDGPU %t/self-member.td 2>&1 \
// RUN:   | FileCheck %t/self-member.td -DFILE=%t/self-member.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../lib/Target/AMDGPU %t/generic-member.td 2>&1 \
// RUN:   | FileCheck %t/generic-member.td -DFILE=%t/generic-member.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../lib/Target/AMDGPU %t/pseudo-member.td 2>&1 \
// RUN:   | FileCheck %t/pseudo-member.td -DFILE=%t/pseudo-member.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../lib/Target/AMDGPU %t/unregistered-member.td 2>&1 \
// RUN:   | FileCheck %t/unregistered-member.td -DFILE=%t/unregistered-member.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../lib/Target/AMDGPU %t/r600-member.td 2>&1 \
// RUN:   | FileCheck %t/r600-member.td -DFILE=%t/r600-member.td --implicit-check-not="error:"

//--- common.td
// Exercise the validation rules with small synthetic targets. The real targets
// are validated when AMDGPUTargetParserDef.inc is generated during the build.
// Stand-ins for the Target.td classes used by AMDGPUTargetParser.td, limited to
// the fields the backend reads.
class InlineBehavior;
def InlineIgnore : InlineBehavior;
def InlineDefault : InlineBehavior;
class SubtargetFeature<string n, string f, string v, string d,
                       list<SubtargetFeature> i = [],
                       InlineBehavior ib = InlineDefault> {
  string Name = n;
  string FieldName = f;
  string Value = v;
  string Desc = d;
  list<SubtargetFeature> Implies = i;
  InlineBehavior InlineBehavior = ib;
}

class SchedMachineModel;
def NoSchedModel : SchedMachineModel;
class ProcessorItineraries;
def NoItineraries : ProcessorItineraries;

class Processor<string n, ProcessorItineraries pi, list<SubtargetFeature> f> {
  string Name = n;
  ProcessorItineraries ProcItin = pi;
  list<SubtargetFeature> Features = f;
}
class ProcessorModel<string n, SchedMachineModel m, list<SubtargetFeature> f>
    : Processor<n, NoItineraries, f> {
  SchedMachineModel SchedModel = m;
}

include "AMDGPUTargetParser.td"

def FeatureAll : SubtargetFeature<"all", "HasAll", "true", "">;
def FeatureAnyA : AMDGPUGenericAnyFeature<"any-a", "HasAnyA", "true", "">;
// Check that subclasses of the marker also opt in, including numeric features.
class NumericFeature<int value> : AMDGPUGenericAnyFeature<
    "any-b-"#value, "TestProperty", !cast<string>(value), "">;
def FeatureAnySmall : NumericFeature<4>;
def FeatureAnyB : NumericFeature<8>;
def FeatureAnyLarge : NumericFeature<16>;
// Classification inherited through a subclass also applies to backend-only
// features, which must participate in validation without being emitted.
class PrivateFeature : AMDGPUGenericAnyFeature<
    "private", "HasPrivate", "true", "">;
def FeaturePrivate : PrivateFeature;
def FeatureImpliesPrivate : SubtargetFeature<
    "implies-private", "HasImpliesPrivate", "true", "", [FeaturePrivate]>;
def FeatureImpliesA : SubtargetFeature<
    "implies-a", "HasImpliesA", "true", "", [FeatureAnyA]>;
def FeatureImpliesB : SubtargetFeature<
    "implies-b", "HasImpliesB", "true", "", [FeatureAnyB]>;
def FeatureAnyParent : AMDGPUGenericAnyFeature<
    "any-parent", "HasAnyParent", "true", "", [FeatureAll], InlineIgnore>;

def AMDGPUFrontendVisibleFeatures {
  list<SubtargetFeature> Features = [FeatureAll, FeatureAnyA, FeatureAnyB,
                                   FeatureAnyParent];
}

//--- valid.td
include "common.td"
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel,
    [FeatureAll, FeatureAnyA, FeatureAnySmall], [9, 0, 0]>;
def GFX901 : AMDGPUProcessorModel<"gfx901", NoSchedModel,
    [FeatureAll, FeatureImpliesB, FeatureImpliesPrivate], [9, 0, 1]>;
// Different members supply different marked features. Follow Implies on both
// the generic and the members, and check marked backend-only features too.
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel,
    [FeatureAll, FeatureImpliesA, FeatureAnyB, FeaturePrivate], [9, 0, 0]> {
  let CoveredGPUs = [GFX900, GFX901];
}
// Validation does not change which features are emitted in the frontend bitset.
// CHECK: Triple::AMDGPUSubArch9, AMDGPUFeatureBitset({FEAT_ALL, FEAT_ANY_A, FEAT_ANY_B_8})

//--- missing-any.td
include "common.td"
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel, [], [9, 0, 0]>;
def GFX901 : AMDGPUProcessorModel<"gfx901", NoSchedModel, [], [9, 0, 1]>;
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' exposes feature 'any-a' not supported by any covered GPU
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureAnyA], [9, 0, 0]> {
  let CoveredGPUs = [GFX900, GFX901];
}

//--- missing-all.td
include "common.td"
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel, [FeatureAll], [9, 0, 0]>;
def GFX901 : AMDGPUProcessorModel<"gfx901", NoSchedModel, [], [9, 0, 1]>;
// Ordinary features still require support from every member.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' exposes feature 'all' not supported by covered GPU 'gfx901'
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureAll], [9, 0, 0]> {
  let CoveredGPUs = [GFX900, GFX901];
}

//--- implied-all.td
include "common.td"
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel, [FeatureAnyParent], [9, 0, 0]>;
def GFX901 : AMDGPUProcessorModel<"gfx901", NoSchedModel, [], [9, 0, 1]>;
// The parent's relaxed rule does not apply to its ordinary implied feature.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' exposes feature 'all' not supported by covered GPU 'gfx901'
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureAnyParent], [9, 0, 0]> {
  let CoveredGPUs = [GFX900, GFX901];
}

//--- missing-private.td
include "common.td"
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel, [], [9, 0, 0]>;
// Explicitly marked features are checked even outside the frontend-visible set.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' exposes feature 'private' not supported by any covered GPU
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeaturePrivate], [9, 0, 0]> {
  let CoveredGPUs = [GFX900];
}

//--- implied-private.td
include "common.td"
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel, [], [9, 0, 0]>;
// An unclassified backend-only parent must not hide a classified dependency.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' exposes feature 'private' not supported by any covered GPU
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureImpliesPrivate], [9, 0, 0]> {
  let CoveredGPUs = [GFX900];
}

//--- numeric-mismatch.td
include "common.td"
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel, [FeatureAnySmall], [9, 0, 0]>;
def GFX901 : AMDGPUProcessorModel<"gfx901", NoSchedModel, [FeatureAnyLarge], [9, 0, 1]>;
// Members provide 4 and 16, but neither provides the generic's exact feature 8.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' exposes feature 'any-b-8' not supported by any covered GPU
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureAnyB], [9, 0, 0]> {
  let CoveredGPUs = [GFX900, GFX901];
}

//--- empty-members.td
include "common.td"
// A named generic target must not bypass validation by leaving its list empty.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' must cover at least one GPU
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureAnyA], [9, 0, 0]>;

//--- self-member.td
include "common.td"
// A generic target cannot supply its own feature support.
// CHECK: [[FILE]]:[[#@LINE+1]]:5: error: generic target 'gfx9-generic' covers 'gfx9-generic', which is not a concrete AMDGPU GPU
def Self : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureAnyA], [9, 0, 0]> {
  let CoveredGPUs = [Self];
}

//--- generic-member.td
include "common.td"
def GFX910 : AMDGPUProcessorModel<"gfx910", NoSchedModel, [FeatureAnyA], [9, 1, 0]>;
def GFX9_1_GENERIC : AMDGPUProcessorModel<"gfx9-1-generic", NoSchedModel, [FeatureAnyA], [9, 1, 0]> {
  let CoveredGPUs = [GFX910];
}
// Covered GPUs must be concrete even when another generic has valid members.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' covers 'gfx9-1-generic', which is not a concrete AMDGPU GPU
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureAnyA], [9, 0, 0]> {
  let CoveredGPUs = [GFX9_1_GENERIC];
}

//--- pseudo-member.td
include "common.td"
def Pseudo : AMDGPUProcessorModel<"generic", NoSchedModel, [FeatureAnyA], [9, 0, 0]> {
  let IsPseudoTarget = true;
}
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' covers 'generic', which is not a concrete AMDGPU GPU
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureAnyA], [9, 0, 0]> {
  let CoveredGPUs = [Pseudo];
}

//--- unregistered-member.td
include "common.td"
def Unregistered : ProcessorModel<"gfx900", NoSchedModel, [FeatureAnyA]>;
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' covers 'gfx900', which is not a concrete AMDGPU GPU
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureAnyA], [9, 0, 0]> {
  let CoveredGPUs = [Unregistered];
}

//--- r600-member.td
include "common.td"
def Cayman : R600ProcessorModel<"cayman", NoItineraries, [FeatureAnyA]>;
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' covers 'cayman', which is not a concrete AMDGPU GPU
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureAnyA], [9, 0, 0]> {
  let CoveredGPUs = [Cayman];
}
