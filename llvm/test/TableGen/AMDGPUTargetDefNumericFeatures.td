// RUN: split-file %s %t
// RUN: llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/valid.td \
// RUN:   | FileCheck %t/valid.td
// RUN: llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/metadata-default.td \
// RUN:   | FileCheck %t/metadata-default.td
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/at-least-invalid.td 2>&1 \
// RUN:   | FileCheck %t/at-least-invalid.td -DFILE=%t/at-least-invalid.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/at-most-invalid.td 2>&1 \
// RUN:   | FileCheck %t/at-most-invalid.td -DFILE=%t/at-most-invalid.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/equal-invalid.td 2>&1 \
// RUN:   | FileCheck %t/equal-invalid.td -DFILE=%t/equal-invalid.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/missing-generic.td 2>&1 \
// RUN:   | FileCheck %t/missing-generic.td -DFILE=%t/missing-generic.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/missing-member.td 2>&1 \
// RUN:   | FileCheck %t/missing-member.td -DFILE=%t/missing-member.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/shared-implied.td 2>&1 \
// RUN:   | FileCheck %t/shared-implied.td -DFILE=%t/shared-implied.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/implied-value.td 2>&1 \
// RUN:   | FileCheck %t/implied-value.td -DFILE=%t/implied-value.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/generic-conflict.td 2>&1 \
// RUN:   | FileCheck %t/generic-conflict.td -DFILE=%t/generic-conflict.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/table-field-conflict.td 2>&1 \
// RUN:   | FileCheck %t/table-field-conflict.td -DFILE=%t/table-field-conflict.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/default-is-fallback.td 2>&1 \
// RUN:   | FileCheck %t/default-is-fallback.td -DFILE=%t/default-is-fallback.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/unannotated.td 2>&1 \
// RUN:   | FileCheck %t/unannotated.td -DFILE=%t/unannotated.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/invalid-value.td 2>&1 \
// RUN:   | FileCheck %t/invalid-value.td -DFILE=%t/invalid-value.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/duplicate-property.td 2>&1 \
// RUN:   | FileCheck %t/duplicate-property.td -DFILE=%t/duplicate-property.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/empty-field.td 2>&1 \
// RUN:   | FileCheck %t/empty-field.td -DFILE=%t/empty-field.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/unknown-comparison.td 2>&1 \
// RUN:   | FileCheck %t/unknown-comparison.td -DFILE=%t/unknown-comparison.td --implicit-check-not="error:"

//--- common.td
include "llvm/Target/Target.td"
include "AMDGPUTargetParser.td"
def MyTarget : Target;

class NumericFeature<string name, string fieldName, int value>
    : SubtargetFeature<name, fieldName, !cast<string>(value), "">;

def FeatureLow : NumericFeature<"low", "TestProperty", 4>;
def FeatureHigh : NumericFeature<"high", "TestProperty", 8>;
def FeatureHigher : NumericFeature<"higher", "TestProperty", 16>;
def FeatureImpliesHigher : SubtargetFeature<
    "implies-higher", "HasImpliesHigher", "true", "", [FeatureHigher]>;
def FeatureCapacitySmall : NumericFeature<"capacity-small", "AddressableLocalMemorySize", 65536>;
def FeatureCapacityBig : NumericFeature<"capacity-big", "AddressableLocalMemorySize", 131072>;
def FeatureEncodingA : NumericFeature<"encoding-a", "Encoding", 7>;
def FeatureEncodingB : NumericFeature<"encoding-b", "Encoding", 7>;

// The higher value and the feature implying it are deliberately not visible.
def AMDGPUFrontendVisibleFeatures {
  list<SubtargetFeature> Features = [FeatureLow, FeatureHigh,
      FeatureCapacitySmall, FeatureCapacityBig, FeatureEncodingA, FeatureEncodingB];
}

//--- valid.td
include "common.td"
def TestProperty : AMDGPUGenericNumericProperty<
    "TestProperty", 4, AMDGPUGenericAtLeast>;
def CapacityProperty : AMDGPUGenericNumericProperty<
    "AddressableLocalMemorySize", 32768, AMDGPUGenericAtMost>;
def EncodingProperty : AMDGPUGenericNumericProperty<
    "Encoding", 0, AMDGPUGenericEqual>;

def FeatureWaves8 : NumericFeature<"waves-8", "MaxWavesPerEU", 8>;
def FeatureWaves8Alias : NumericFeature<"waves-8-alias", "MaxWavesPerEU", 8>;
def FeatureWaves16 : NumericFeature<"waves-16", "MaxWavesPerEU", 16>;

// Multiple features setting the same value are still accepted.
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel,
    [FeatureLow, FeatureCapacitySmall, FeatureEncodingA, FeatureWaves8,
     FeatureWaves8Alias], [9, 0, 0]>;
def GFX901 : AMDGPUProcessorModel<"gfx901", NoSchedModel,
    [FeatureHigh, FeatureCapacityBig, FeatureEncodingB, FeatureWaves16], [9, 0, 1]>;
def GFX902 : AMDGPUProcessorModel<"gfx902", NoSchedModel, [], [9, 0, 2]>;
def GFX903 : AMDGPUProcessorModel<"gfx903", NoSchedModel,
    [FeatureImpliesHigher], [9, 0, 3]>;

// All three comparisons replace exact-record matching for visible numeric
// features. In particular, the generic's encoding-a is absent from GFX901,
// whose encoding-b sets the same value, and its capacity-small is below GFX901's
// capacity-big. Neither prevents this generic from covering GFX901.
// CHECK: GK_GFX9_GENERIC
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel,
    [FeatureHigh, FeatureCapacitySmall, FeatureEncodingA], [9, 0, 0]> {
  let CoveredGPUs = [GFX900, GFX901];
}

// Missing features use the declared defaults on either side.
// CHECK: GK_GFX8_GENERIC
def : AMDGPUProcessorModel<"gfx8-generic", NoSchedModel, [], [8, 0, 0]> {
  let CoveredGPUs = [GFX902];
}

// Follow Implies edges, including features outside the frontend-visible set.
// CHECK: GK_GFX7_GENERIC
def : AMDGPUProcessorModel<"gfx7-generic", NoSchedModel,
    [FeatureHigher], [7, 0, 0]> {
  let CoveredGPUs = [GFX903];
}

// The emitted table uses the unambiguous feature value, or its fallback when no
// feature sets the field. A fallback (MaxWavesPerEU = 10) is not a minimum.
// CHECK: {[[#]], Triple::AMDGPUSubArch900, {{.*}}, {9, 0, 0}, [[#]], 8, 65536, 32, 0},
// CHECK: {[[#]], Triple::AMDGPUSubArch901, {{.*}}, {9, 0, 1}, [[#]], 16, 131072, 32, 0},
// CHECK: {[[#]], Triple::AMDGPUSubArch902, {{.*}}, {9, 0, 2}, [[#]], 10, 32768, 32, 0},

//--- metadata-default.td
include "common.td"
// Deliberately differ from the legacy 32768 fallback. The metadata must be used
// both when checking equality and when emitting concrete and generic GPU rows.
def CapacityProperty : AMDGPUGenericNumericProperty<
    "AddressableLocalMemorySize", 65536, AMDGPUGenericEqual>;
def FeatureCapacity32K : NumericFeature<"capacity-32k", "AddressableLocalMemorySize", 32768>;
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel,
    [FeatureCapacitySmall], [9, 0, 0]>;
def GFX901 : AMDGPUProcessorModel<"gfx901", NoSchedModel, [], [9, 0, 1]>;
// An explicit value below the metadata fallback must not be clamped to it.
def GFX902 : AMDGPUProcessorModel<"gfx902", NoSchedModel,
    [FeatureCapacity32K], [9, 0, 2]>;
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [], [9, 0, 0]> {
  let CoveredGPUs = [GFX900, GFX901];
}
// CHECK: {[[#]], Triple::AMDGPUSubArch900, {{.*}}, {9, 0, 0}, [[#]], 10, 65536, 32, 0},
// CHECK: {[[#]], Triple::AMDGPUSubArch901, {{.*}}, {9, 0, 1}, [[#]], 10, 65536, 32, 0},
// CHECK: {[[#]], Triple::AMDGPUSubArch902, {{.*}}, {9, 0, 2}, [[#]], 10, 32768, 32, 0},
// CHECK: {[[#]], Triple::AMDGPUSubArch9, {{.*}}, {9, 0, 0}, [[#]], 10, 65536, 32, 0},

//--- at-least-invalid.td
include "common.td"
def TestProperty : AMDGPUGenericNumericProperty<
    "TestProperty", 4, AMDGPUGenericAtLeast>;
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel, [FeatureHigh], [9, 0, 0]>;
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' has 'TestProperty' value 4, which must be at least the value 8 of covered GPU 'gfx900'
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureLow], [9, 0, 0]> {
  let CoveredGPUs = [GFX900];
}

//--- at-most-invalid.td
include "common.td"
def TestProperty : AMDGPUGenericNumericProperty<
    "TestProperty", 4, AMDGPUGenericAtMost>;
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel, [FeatureLow], [9, 0, 0]>;
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' has 'TestProperty' value 8, which must be at most the value 4 of covered GPU 'gfx900'
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureHigh], [9, 0, 0]> {
  let CoveredGPUs = [GFX900];
}

//--- equal-invalid.td
include "common.td"
def TestProperty : AMDGPUGenericNumericProperty<
    "TestProperty", 4, AMDGPUGenericEqual>;
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel, [FeatureHigh], [9, 0, 0]>;
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' has 'TestProperty' value 4, which must be equal to the value 8 of covered GPU 'gfx900'
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureLow], [9, 0, 0]> {
  let CoveredGPUs = [GFX900];
}

//--- missing-generic.td
include "common.td"
def TestProperty : AMDGPUGenericNumericProperty<
    "TestProperty", 4, AMDGPUGenericAtLeast>;
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel, [FeatureHigh], [9, 0, 0]>;
// A missing generic feature still requires checking the property's default.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' has 'TestProperty' value 4, which must be at least the value 8 of covered GPU 'gfx900'
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [], [9, 0, 0]> {
  let CoveredGPUs = [GFX900];
}

//--- missing-member.td
include "common.td"
def TestProperty : AMDGPUGenericNumericProperty<
    "TestProperty", 4, AMDGPUGenericAtMost>;
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel, [], [9, 0, 0]>;
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' has 'TestProperty' value 8, which must be at most the value 4 of covered GPU 'gfx900'
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureHigh], [9, 0, 0]> {
  let CoveredGPUs = [GFX900];
}

//--- shared-implied.td
include "common.td"
def TestProperty : AMDGPUGenericNumericProperty<
    "TestProperty", 4, AMDGPUGenericAtLeast>;
// A shared visible feature cannot hide a conflicting implied, non-visible one.
// CHECK: [[FILE]]:[[#@LINE+1]]:5: error: GPU 'gfx900' gets conflicting 'TestProperty' values from 'higher' and 'low'
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel,
    [FeatureLow, FeatureImpliesHigher], [9, 0, 0]>;
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureLow], [9, 0, 0]> {
  let CoveredGPUs = [GFX900];
}

//--- implied-value.td
include "common.td"
def TestProperty : AMDGPUGenericNumericProperty<
    "TestProperty", 4, AMDGPUGenericAtLeast>;
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel,
    [FeatureImpliesHigher], [9, 0, 0]>;
// The numeric comparison also considers implied features that are not visible.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' has 'TestProperty' value 4, which must be at least the value 16 of covered GPU 'gfx900'
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureLow], [9, 0, 0]> {
  let CoveredGPUs = [GFX900];
}

//--- generic-conflict.td
include "common.td"
def TestProperty : AMDGPUGenericNumericProperty<
    "TestProperty", 4, AMDGPUGenericAtLeast>;
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel, [FeatureHigh], [9, 0, 0]>;
// Reject conflicting values in the generic target as well as in its members.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: GPU 'gfx9-generic' gets conflicting 'TestProperty' values from 'low' and 'high'
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel,
    [FeatureHigh, FeatureLow], [9, 0, 0]> {
  let CoveredGPUs = [GFX900];
}

//--- table-field-conflict.td
include "common.td"
def FeatureWaves8 : NumericFeature<"waves-8", "MaxWavesPerEU", 8>;
def FeatureWaves16 : NumericFeature<"waves-16", "MaxWavesPerEU", 16>;
// Preserve conflict checks for existing table fields, even without property
// metadata or any generic target covering this GPU.
// CHECK: [[FILE]]:[[#@LINE+1]]:5: error: GPU 'gfx900' gets conflicting 'MaxWavesPerEU' values from 'waves-16' and 'waves-8'
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel,
    [FeatureWaves8, FeatureWaves16], [9, 0, 0]>;

//--- default-is-fallback.td
include "common.td"
def TestProperty : AMDGPUGenericNumericProperty<
    "TestProperty", 10, AMDGPUGenericAtMost>;
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel, [FeatureLow], [9, 0, 0]>;
// The default does not clamp explicitly specified values up to 10.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' has 'TestProperty' value 8, which must be at most the value 4 of covered GPU 'gfx900'
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureHigh], [9, 0, 0]> {
  let CoveredGPUs = [GFX900];
}

//--- unannotated.td
include "common.td"
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel, [FeatureHigh], [9, 0, 0]>;
// Numeric-looking values alone do not opt a field out of exact matching.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' exposes feature 'low' not supported by covered GPU 'gfx900'
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureLow], [9, 0, 0]> {
  let CoveredGPUs = [GFX900];
}

//--- invalid-value.td
include "common.td"
def TestProperty : AMDGPUGenericNumericProperty<
    "TestProperty", 4, AMDGPUGenericAtLeast>;
// Numeric validation requires integer literals, not C++ enum expressions.
// CHECK: [[FILE]]:[[#@LINE+1]]:5: error: feature 'invalid' must have an integer value
def FeatureInvalid : SubtargetFeature<"invalid", "TestProperty", "SomeEnumValue", "">;
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel, [FeatureInvalid], [9, 0, 0]>;
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel, [FeatureLow], [9, 0, 0]> {
  let CoveredGPUs = [GFX900];
}

//--- duplicate-property.td
include "common.td"
def A : AMDGPUGenericNumericProperty<"TestProperty", 4, AMDGPUGenericAtLeast>;
// CHECK: [[FILE]]:[[#@LINE+1]]:5: error: duplicate numeric property for field 'TestProperty'
def B : AMDGPUGenericNumericProperty<"TestProperty", 4, AMDGPUGenericAtMost>;

//--- empty-field.td
include "common.td"
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: numeric property must have a field name
def : AMDGPUGenericNumericProperty<"", 4, AMDGPUGenericAtLeast>;

//--- unknown-comparison.td
include "common.td"
def UnknownComparison : AMDGPUGenericNumericComparison;
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: unknown generic numeric comparison 'UnknownComparison'
def : AMDGPUGenericNumericProperty<"TestProperty", 4, UnknownComparison>;
