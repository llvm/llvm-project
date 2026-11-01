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

// A generic target may use this feature if at least one covered GPU has it.
// These features participate in validation even when not frontend-visible.
// Other frontend-visible features must be present on every covered GPU.
// Implied features are checked according to their own classes.
// This checks feature membership, not numeric ordering of feature values.
class AMDGPUGenericAnyFeature<string name, string fieldName, string value,
                              string description,
                              list<SubtargetFeature> implies = [],
                              InlineBehavior ib = InlineDefault>
    : SubtargetFeature<name, fieldName, value, description, implies, ib>;

// Marks a Processor/ProcessorModel record as a canonical GPU.
//
// \p isa is the ISA version [major, minor, stepping]. Empty for R600 (no AMDGCN
// ISA version); required for AMDGCN GPUs. Not derivable from the name.
class AMDGPUGPUInfo<list<int> isa = []> {
  list<int> IsaVersion = isa;

  // For a "gfxN-generic" target, this must contain at least one compatible
  // concrete AMDGPU GPU. For individual GPUs and pseudo targets, it must be empty.
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
