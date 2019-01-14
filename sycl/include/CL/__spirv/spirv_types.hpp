//===----------- spirv_types.hpp --- SPIRV types -------------------------===//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>

namespace cl {
namespace __spirv {

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
// Only in such cases the class is recognized as SPIRV type OpTypeEvent.
class OpTypeEvent;

} // namespace __spirv
} // namespace cl
