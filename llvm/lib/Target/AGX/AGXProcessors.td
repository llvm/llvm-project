//===-- AGXProcessors.td - AGX processor definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Processor names follow the GPU names used by the Asahi Linux driver
// (agx-<generation><variant>). The variant letter is G (base), S (Pro),
// C (Max), D (Ultra) and P (A-series phone/tablet parts).
//
//===----------------------------------------------------------------------===//

class AGXProc<string Name, SchedMachineModel Model,
              list<SubtargetFeature> Features>
    : ProcessorModel<Name, Model, Features>;

// Generic target: the G13 base feature set.
def : AGXProc<"generic", AGXG13Model, [FeatureG13]>;

// G13: Apple M1 family.
def : AGXProc<"agx-g13g", AGXG13Model, [FeatureG13]>; // M1
def : AGXProc<"agx-g13s", AGXG13Model,                // M1 Pro
              [FeatureG13, FeatureMultiDieCoherency]>;
def : AGXProc<"agx-g13c", AGXG13Model,                // M1 Max
              [FeatureG13, FeatureMultiDieCoherency]>;
def : AGXProc<"agx-g13d", AGXG13Model,                // M1 Ultra
              [FeatureG13, FeatureMultiDieCoherency]>;

// G14: Apple M2 family. The scheduling model is measured on G14C.
def : AGXProc<"agx-g14g", AGXG14Model, [FeatureG14]>; // M2
def : AGXProc<"agx-g14s", AGXG14Model, [FeatureG14]>; // M2 Pro
def : AGXProc<"agx-g14c", AGXG14Model, [FeatureG14]>; // M2 Max
def : AGXProc<"agx-g14d", AGXG14Model, [FeatureG14]>; // M2 Ultra

// G15: Apple M3 family.
def : AGXProc<"agx-g15g", AGXG14Model, [FeatureG15]>; // M3
def : AGXProc<"agx-g15s", AGXG14Model, [FeatureG15]>; // M3 Pro
def : AGXProc<"agx-g15c", AGXG14Model, [FeatureG15]>; // M3 Max
def : AGXProc<"agx-g15d", AGXG14Model, [FeatureG15]>; // M3 Ultra

// G16: Apple M4 family.
def : AGXProc<"agx-g16g", AGXApple9Model, [FeatureG16]>; // M4
def : AGXProc<"agx-g16s", AGXApple9Model, [FeatureG16]>; // M4 Pro
def : AGXProc<"agx-g16c", AGXApple9Model, [FeatureG16]>; // M4 Max

// G17: Apple A18 Pro and M5.
def : AGXProc<"agx-g17p", AGXApple9Model, [FeatureG17]>; // A18 Pro
def : AGXProc<"agx-g17g", AGXApple9Model, [FeatureG17]>; // M5
