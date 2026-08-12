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

// FIXME: These duplicate real SubtargetFeatures and should be unified
// (e.g. FEATURE_XNACK mirrors FeatureXNACK).
class AMDGPUArchFeature<string spelling> {
  string Spelling = spelling;
}

// R600-only.
def R600_FEATURE_FMA : AMDGPUArchFeature<"R600_FEATURE_FMA">;

// AMDGCN.
def FEATURE_FAST_FMA_F32       : AMDGPUArchFeature<"FEATURE_FAST_FMA_F32">;
def FEATURE_FAST_DENORMAL_F32  : AMDGPUArchFeature<"FEATURE_FAST_DENORMAL_F32">;
def FEATURE_WAVE32             : AMDGPUArchFeature<"FEATURE_WAVE32">;
def FEATURE_XNACK              : AMDGPUArchFeature<"FEATURE_XNACK">;
def FEATURE_SRAMECC            : AMDGPUArchFeature<"FEATURE_SRAMECC">;
def FEATURE_WGP                : AMDGPUArchFeature<"FEATURE_WGP">;
def FEATURE_XNACK_ON_OFF_MODES : AMDGPUArchFeature<"FEATURE_XNACK_ON_OFF_MODES">;
def FEATURE_SGPR_INIT_BUG      : AMDGPUArchFeature<"FEATURE_SGPR_INIT_BUG">;

// Marks a Processor/ProcessorModel record as a canonical GPU.
//
// \p isa is the ISA version [major, minor, stepping]. Empty for R600 (no AMDGCN
// ISA version); required for AMDGCN GPUs. Not derivable from the name.
class AMDGPUGPUInfo<list<int> isa = []> {
  list<AMDGPUArchFeature> ArchFeatures = [];

  list<int> IsaVersion = isa;

  // List of targets which are compatible with this target. This
  // should be used for a "gfxN-generic" targets only, and empty for
  // individual GPUs.
  list<Processor> CoveredGPUs = [];

  // A pseudo target ("generic"/"generic-hsa") that represents no
  // hardware.
  bit IsPseudoTarget = false;
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
