//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def SuperHOptions : OptionsStruct<prefix = "sh-">;

//===--------------------------------------------------------------------------===//
// SuperHISelDAGToDAG.cpp
//===--------------------------------------------------------------------------===//
defm : BoolField<"force-delay-slot", "false",
    "Force the usage of delay slots for conditional branches.">;




//===--------------------------------------------------------------------------===//
// SuperHFrameLowering.cpp
//===--------------------------------------------------------------------------===//
defm : BoolField<"accumulate-outgoing-args", "false",
    "Reserve space for outgoing arguments in the function prologue.">;

defm : ValueField<"max-sp-add", "int64_t", 
    "10", "How many add instructions are deemed low enough cost to generate for sp adjust.">;




//===--------------------------------------------------------------------------===//
// SuperHConstantIslandPass.cpp
//===--------------------------------------------------------------------------===//
defm : ValueField<"constant-island-max-iteration", "unsigned", 
    "30", "The max number of iteration for converge">;




//===--------------------------------------------------------------------------===//
// SuperHExpandPseudoInsts.cpp
//===--------------------------------------------------------------------------===//
defm : ValueField<"pseudo-fp-max-add", "uint64_t", 
    "16", 
    "Maximum amount of add instructions to generate for heap fp load/stores.">;

let Condition = AssertsOnly in
defm : BoolField<"pseudo-no-expand", "false",
    "Force the backend to not expand instructions.">;

let Condition = AssertsOnly in
defm : BoolField<"pseudo-keep", "false",
    "Force the backend to keep pseudos around.">;