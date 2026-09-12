//===-- AMDGPUTargetParser.td - TargetParser metadata ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TargetParser metadata for the -gen-amdgpu-target-def backend.
//
//===----------------------------------------------------------------------===//

// How a generic target's numeric property must compare with the same property
// on each covered GPU. This only compares values between GPUs; it does not
// relax the checks for conflicting feature values within a GPU.
class AMDGPUGenericNumericComparison;
def AMDGPUGenericAtLeast : AMDGPUGenericNumericComparison;
def AMDGPUGenericAtMost : AMDGPUGenericNumericComparison;
def AMDGPUGenericEqual : AMDGPUGenericNumericComparison;

// Opt a numeric SubtargetFeature field into generic-target validation. All
// features setting FieldName participate, including implied features and ones
// that are not frontend-visible, and must agree on an integer-literal value.
// DefaultValue is the fallback used by both validation and table emission when
// no feature sets the field, and must match the backend's fallback. Declare at
// most one property per FieldName.
//
// Fields without this metadata retain the exact feature-record matching used
// for frontend-visible capabilities.
class AMDGPUGenericNumericProperty<
    string fieldName, int defaultValue,
    AMDGPUGenericNumericComparison comparison> {
  string FieldName = fieldName;
  int DefaultValue = defaultValue;
  AMDGPUGenericNumericComparison GenericComparison = comparison;
}

// Marks a Processor/ProcessorModel record as a canonical GPU.
//
// \p isa is the ISA version [major, minor, stepping]. Empty for R600 (no AMDGCN
// ISA version); required for AMDGCN GPUs. Not derivable from the name.
class AMDGPUGPUInfo<list<int> isa = []> {
  list<int> IsaVersion = isa;

  // List of targets which are compatible with this target. This
  // should be used for a "gfxN-generic" targets only, and empty for
  // individual GPUs.
  list<Processor> CoveredGPUs = [];

  // A pseudo target ("generic"/"generic-hsa") that represents no
  // hardware.
  bit IsPseudoTarget = false;

  // An explicit triple subarch spelling (e.g. "12.50"). If unset,
  // this will be guessed from the device name.
  string SubArchSpelling;
}

// An R600 processor that is also a canonical TargetParser GPU.
class R600ProcessorModel<string n, ProcessorItineraries pi,
                         list<SubtargetFeature> f>
    : Processor<n, pi, f>, AMDGPUGPUInfo;

// A gfx6+ processor that is also a canonical TargetParser GPU. \p isa
// is the ISA version [major, minor, stepping].
class AMDGPUProcessorModel<string n, SchedMachineModel m,
                           list<SubtargetFeature> f, list<int> isa>
    : ProcessorModel<n, m, f>, AMDGPUGPUInfo<isa>;

// A major-subarch family that has no "gfxN-generic" target of its own
// (gfx6, gfx7, gfx8). \p major is the bare subarch suffix (e.g. "6");
// \p members are the GPUs mapped to Triple::AMDGPUSubArch<major>
class AMDGPUFamily<string major, list<Processor> members> {
  string MajorSubArch = major;
  list<Processor> Members = members;
}
