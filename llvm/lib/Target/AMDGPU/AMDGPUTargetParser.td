//===-- AMDGPUTargetParser.td - TargetParser metadata ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TargetParser metadata attached to processor records for the
// -gen-amdgpu-target-def backend.
//
//===----------------------------------------------------------------------===//

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
