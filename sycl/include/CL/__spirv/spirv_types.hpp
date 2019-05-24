//===----------- spirv_types.hpp --- SPIRV types -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>

// TODO: include the header file with SPIR-V declarations from SPIRV-Headers
// project.

enum Scope {
  CrossDevice = 0,
  Device = 1,
  Workgroup = 2,
  Subgroup = 3,
  Invocation = 4,
};


enum MemorySemantics {
  None = 0x0,
  Acquire = 0x2,
  Release = 0x4,
  AcquireRelease = 0x8,
  SequentiallyConsistent = 0x10,
  UniformMemory = 0x40,
  SubgroupMemory = 0x80,
  WorkgroupMemory = 0x100,
  CrossWorkgroupMemory = 0x200,
  AtomicCounterMemory = 0x400,
  ImageMemory = 0x800,
};

// This class does not have definition, it is only predeclared here.
// The pointers to this class objects can be passed to or returned from
// SPIRV built-in functions.
// Only in such cases the class is recognized as SPIRV type __ocl_event_t.
#ifndef __SYCL_DEVICE_ONLY__
typedef void* __ocl_event_t;
typedef void* __ocl_sampler_t;
#endif

enum GroupOperation { Reduce = 0, InclusiveScan = 1, ExclusiveScan = 2 };
