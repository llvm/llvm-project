//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def SystemZOptions : OptionsStruct<prefix = "systemz-">;

// SystemZHazardRecognizer.cpp
// This is the limit of processor resource usage at which the scheduler should
// try to look for other instructions (not using the critical resource).
defm : ValueField<"systemz-procres-cost-lim", "int", "8",
  "The OOO window for processor resources during scheduling">;

// SystemZISelLowering.cpp
// Temporarily let this be disabled by default until all known problems related
// to argument extensions are fixed.
defm : OptionalBoolField<"argext-abi-check",
  "Verify that narrow int args are properly extended per the SystemZ ABI">;

// SystemZSubtarget.cpp
defm : BoolField<"systemz-subreg-liveness", "false",
  "Enable subregister liveness tracking for SystemZ (experimental)">;

// SystemZTargetMachine.cpp
defm : BoolField<"systemz-generic-sched", "false",
  "Run the generic pre-ra scheduler instead of the SystemZ scheduler">;
defm : BoolField<"systemz-machine-combiner", "true",
  "Enable the machine combiner pass">;
