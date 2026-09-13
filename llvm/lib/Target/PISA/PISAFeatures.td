//===-- PISAFeatures.td - Describe the PISA Features -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Target architecture feature.
def Feature100 : SubtargetFeature<"100", "PISATarget100", "true",
                                  "Support PISA 100 instruction/feature set", []>;
