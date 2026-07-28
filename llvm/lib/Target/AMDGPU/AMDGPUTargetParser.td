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
def FEATURE_FMA : AMDGPUArchFeature<"FEATURE_FMA">;

// Marks a Processor/ProcessorModel record as a canonical TargetParser GPU.
class AMDGPUGPUInfo {
  list<AMDGPUArchFeature> ArchFeatures = [];
}

// An R600 processor that is also a canonical TargetParser GPU.
class R600ProcessorModel<string n, ProcessorItineraries pi,
                         list<SubtargetFeature> f>
    : Processor<n, pi, f>, AMDGPUGPUInfo;
