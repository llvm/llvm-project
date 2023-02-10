//===- ARMRandezvousOptions.cpp - ARM Randezvous Command Line Options -----===//
//
// Copyright (c) 2021-2022, University of Rochester
//
// Part of the Randezvous Project, under the Apache License v2.0 with
// LLVM Exceptions.  See LICENSE.txt in the llvm directory for license
// information.
//
//===----------------------------------------------------------------------===//
//
// This file defines the command line options for ARM Randezvous passes.
//
//===----------------------------------------------------------------------===//

#include "ARMRandezvousOptions.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// Randezvous pass enablers
//===----------------------------------------------------------------------===//

bool EnableRandezvousCLR;
static cl::opt<bool, true>
CLR("arm-randezvous-clr",
    cl::Hidden,
    cl::desc("Enable ARM Randezvous Code Layout Randomization"),
    cl::location(EnableRandezvousCLR),
    cl::init(false));

bool EnableRandezvousBBLR;
static cl::opt<bool, true>
BBLR("arm-randezvous-bblr",
     cl::Hidden,
     cl::desc("Enable Basic Block Layout Randomization for ARM Randezvous CLR"),
     cl::location(EnableRandezvousBBLR),
     cl::init(false));

bool EnableRandezvousBBCLR;
static cl::opt<bool, true>
BBCLR("arm-randezvous-bbclr",
      cl::Hidden,
      cl::desc("Enable Basic Block Cluster Layout Randomization for ARM Randezvous CLR"),
      cl::location(EnableRandezvousBBCLR),
      cl::init(false));

bool EnableRandezvousPicoXOM;
static cl::opt<bool, true>
PicoXOM("arm-randezvous-picoxom",
        cl::Hidden,
        cl::desc("Enable ARM Randezvous Execute-Only Memory"),
        cl::location(EnableRandezvousPicoXOM),
        cl::init(false));

bool EnableRandezvousGDLR;
static cl::opt<bool, true>
GDLR("arm-randezvous-gdlr",
     cl::Hidden,
     cl::desc("Enable ARM Randezvous Global Data Layout Randomization"),
     cl::location(EnableRandezvousGDLR),
     cl::init(false));

bool EnableRandezvousDecoyPointers;
static cl::opt<bool, true>
DecoyPointers("arm-randezvous-dp",
              cl::Hidden,
              cl::desc("Enable ARM Randezvous Decoy Pointers"),
              cl::location(EnableRandezvousDecoyPointers),
              cl::init(false));

bool EnableRandezvousGlobalGuard;
static cl::opt<bool, true>
GlobalGuard("arm-randezvous-global-guard",
            cl::Hidden,
            cl::desc("Enable ARM Randezvou Global Guard"),
            cl::location(EnableRandezvousGlobalGuard),
            cl::init(false));

bool EnableRandezvousShadowStack;
static cl::opt<bool, true>
ShadowStack("arm-randezvous-shadow-stack",
            cl::Hidden,
            cl::desc("Enable ARM Randezvous Shadow Stack"),
            cl::location(EnableRandezvousShadowStack),
            cl::init(false));

bool EnableRandezvousRAN;
static cl::opt<bool, true>
RAN("arm-randezvous-ran",
    cl::Hidden,
    cl::desc("Enable ARM Randezvous Return Address Nullification"),
    cl::location(EnableRandezvousRAN),
    cl::init(false));

bool EnableRandezvousLGPromote;
static cl::opt<bool, true>
LGPromote("arm-randezvous-lgp",
          cl::Hidden,
          cl::desc("Enable ARM Randezvous Local-to-Global Promotion"),
          cl::location(EnableRandezvousLGPromote),
          cl::init(false));

bool EnableRandezvousICallLimiter;
static cl::opt<bool, true>
ICallLimiter("arm-randezvous-icall-limiter",
             cl::Hidden,
             cl::desc("Enable ARM Randezvous Indirect Call Limiter"),
             cl::location(EnableRandezvousICallLimiter),
             cl::init(false));

//===----------------------------------------------------------------------===//
// Randezvous pass seeds
//===----------------------------------------------------------------------===//

uint64_t RandezvousCLRSeed;
static cl::opt<uint64_t, true>
CLRSeed("arm-randezvous-clr-seed",
        cl::Hidden,
        cl::desc("Seed for the RNG used in ARM Randezvous CLR"),
        cl::location(RandezvousCLRSeed),
        cl::init(0));

uint64_t RandezvousGDLRSeed;
static cl::opt<uint64_t, true>
GDLRSeed("arm-randezvous-gdlr-seed",
         cl::Hidden,
         cl::desc("Seed for the RNG used in ARM Randezvous GDLR"),
         cl::location(RandezvousGDLRSeed),
         cl::init(0));

uint64_t RandezvousShadowStackSeed;
static cl::opt<uint64_t, true>
ShadowStackSeed("arm-randezvous-shadow-stack-seed",
                cl::Hidden,
                cl::desc("Seed for the RNG used in ARM Randezvous Shadow Stack"),
                cl::location(RandezvousShadowStackSeed),
                cl::init(0));

//===----------------------------------------------------------------------===//
// Size options used by Randezvous passes
//===----------------------------------------------------------------------===//

size_t RandezvousMaxTextSize;
static cl::opt<size_t, true>
MaxTextSize("arm-randezvous-max-text-size",
            cl::Hidden,
            cl::desc("Maximum text section size in bytes"),
            cl::location(RandezvousMaxTextSize),
            cl::init(0x1e0000));   // 2 MB - 128 KB

size_t RandezvousMaxRodataSize;
static cl::opt<size_t, true>
MaxRodataSize("arm-randezvous-max-rodata-size",
              cl::Hidden,
              cl::desc("Maximum rodata section size in bytes"),
              cl::location(RandezvousMaxRodataSize),
              cl::init(0x10000));  // 64 KB

size_t RandezvousMaxDataSize;
static cl::opt<size_t, true>
MaxDataSize("arm-randezvous-max-data-size",
            cl::Hidden,
            cl::desc("Maximum data section size in bytes"),
            cl::location(RandezvousMaxDataSize),
            cl::init(0x10000));    // 64 KB

size_t RandezvousMaxBssSize;
static cl::opt<size_t, true>
MaxBssSize("arm-randezvous-max-bss-size",
           cl::Hidden,
           cl::desc("Maximum bss section size in bytes"),
           cl::location(RandezvousMaxBssSize),
           cl::init(0x10000));     // 64 KB

size_t RandezvousShadowStackSize;
static cl::opt<size_t, true>
ShadowStackSize("arm-randezvous-shadow-stack-size",
                cl::Hidden,
                cl::desc("ARM Randezvous Shadow Stack size in bytes"),
                cl::location(RandezvousShadowStackSize),
                cl::init(0x8000)); // 32 KB

//===----------------------------------------------------------------------===//
// Miscellaneous options used by Randezvous passes
//===----------------------------------------------------------------------===//

unsigned RandezvousShadowStackStrideLength;
static cl::opt<unsigned, true>
ShadowStackStrideLength("arm-randezvous-shadow-stack-stride-length",
                        cl::Hidden,
                        cl::desc("Number of bits for ARM Randezvous Shadow Stack stride"),
                        cl::location(RandezvousShadowStackStrideLength),
                        cl::init(8));

unsigned RandezvousNumGlobalGuardCandidates;
static cl::opt<unsigned, true>
NumGlobalGuardCandidates("arm-randezvous-num-global-guard-candidates",
                         cl::Hidden,
                         cl::desc("Number of global guard candidates to generate"),
                         cl::location(RandezvousNumGlobalGuardCandidates),
                         cl::init(64));

uintptr_t RandezvousRNGAddress;
static cl::opt<uintptr_t, true>
RNGAddress("arm-randezvous-rng-addr",
           cl::Hidden,
           cl::desc("Address of a dynamic RNG"),
           cl::location(RandezvousRNGAddress),
           cl::init(0));