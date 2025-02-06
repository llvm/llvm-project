//===- Utility.h - Collection of geneirc offloading utilities -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_OFFLOADING_UTILITY_H
#define LLVM_FRONTEND_OFFLOADING_UTILITY_H

#include <cstdint>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"

namespace llvm {
namespace offloading {

/// This is the record of an object that just be registered with the offloading
/// runtime.
struct EntryTy {
  /// Reserved bytes used to detect an older version of the struct, always zero.
  uint64_t Reserved = 0x0;
  /// The current version of the struct for runtime forward compatibility.
  uint16_t Version = 0x1;
  /// The expected consumer of this entry, e.g. CUDA or OpenMP.
  uint16_t Kind;
  /// Flags associated with the global.
  uint32_t Flags;
  /// The address of the global to be registered by the runtime.
  void *Address;
  /// The name of the symbol in the device image.
  char *SymbolName;
  /// The number of bytes the symbol takes.
  uint64_t Size;
  /// Extra generic data used to register this entry.
  uint64_t Data;
  /// An extra pointer, usually null.
  void *AuxAddr;
};

/// Offloading entry flags for CUDA / HIP. The first three bits indicate the
/// type of entry while the others are a bit field for additional information.
enum OffloadEntryKindFlag : uint32_t {
  /// Mark the entry as a global entry. This indicates the presense of a
  /// kernel if the size size field is zero and a variable otherwise.
  OffloadGlobalEntry = 0x0,
  /// Mark the entry as a managed global variable.
  OffloadGlobalManagedEntry = 0x1,
  /// Mark the entry as a surface variable.
  OffloadGlobalSurfaceEntry = 0x2,
  /// Mark the entry as a texture variable.
  OffloadGlobalTextureEntry = 0x3,
  /// Mark the entry as being extern.
  OffloadGlobalExtern = 0x1 << 3,
  /// Mark the entry as being constant.
  OffloadGlobalConstant = 0x1 << 4,
  /// Mark the entry as being a normalized surface.
  OffloadGlobalNormalized = 0x1 << 5,
};

/// Returns the type of the offloading entry we use to store kernels and
/// globals that will be registered with the offloading runtime.
StructType *getEntryTy(Module &M);

/// Create an offloading section struct used to register this global at
/// runtime.
///
/// \param M The module to be used
/// \param Addr The pointer to the global being registered.
/// \param Kind The offloading language expected to consume this.
/// \param Name The symbol name associated with the global.
/// \param Size The size in bytes of the global (0 for functions).
/// \param Flags Flags associated with the entry.
/// \param Data Extra data storage associated with the entry.
/// \param SectionName The section this entry will be placed at.
/// \param AuxAddr An extra pointer if needed.
void emitOffloadingEntry(Module &M, object::OffloadKind Kind, Constant *Addr,
                         StringRef Name, uint64_t Size, uint32_t Flags,
                         uint64_t Data, Constant *AuxAddr = nullptr,
                         StringRef SectionName = "llvm_offload_entries");

/// Create a constant struct initializer used to register this global at
/// runtime.
/// \return the constant struct and the global variable holding the symbol name.
std::pair<Constant *, GlobalVariable *>
getOffloadingEntryInitializer(Module &M, object::OffloadKind Kind,
                              Constant *Addr, StringRef Name, uint64_t Size,
                              uint32_t Flags, uint64_t Data, Constant *AuxAddr);

/// Creates a pair of globals used to iterate the array of offloading entries by
/// accessing the section variables provided by the linker.
std::pair<GlobalVariable *, GlobalVariable *>
getOffloadEntryArray(Module &M, StringRef SectionName = "llvm_offload_entries");

namespace amdgpu {
/// Check if an image is compatible with current system's environment. The
/// system environment is given as a 'target-id' which has the form:
///
/// <target-id> := <processor> ( ":" <target-feature> ( "+" | "-" ) )*
///
/// If a feature is not specific as '+' or '-' it is assumed to be in an 'any'
/// and is compatible with either '+' or '-'. The HSA runtime returns this
/// information using the target-id, while we use the ELF header to determine
/// these features.
bool isImageCompatibleWithEnv(StringRef ImageArch, uint32_t ImageFlags,
                              StringRef EnvTargetID);

/// Struct for holding metadata related to AMDGPU kernels, for more information
/// about the metadata and its meaning see:
/// https://llvm.org/docs/AMDGPUUsage.html#code-object-v3
struct AMDGPUKernelMetaData {
  /// Constant indicating that a value is invalid.
  static constexpr uint32_t KInvalidValue =
      std::numeric_limits<uint32_t>::max();
  /// The amount of group segment memory required by a work-group in bytes.
  uint32_t GroupSegmentList = KInvalidValue;
  /// The amount of fixed private address space memory required for a work-item
  /// in bytes.
  uint32_t PrivateSegmentSize = KInvalidValue;
  /// Number of scalar registers required by a wavefront.
  uint32_t SGPRCount = KInvalidValue;
  /// Number of vector registers required by each work-item.
  uint32_t VGPRCount = KInvalidValue;
  /// Number of stores from a scalar register to a register allocator created
  /// spill location.
  uint32_t SGPRSpillCount = KInvalidValue;
  /// Number of stores from a vector register to a register allocator created
  /// spill location.
  uint32_t VGPRSpillCount = KInvalidValue;
  /// Number of accumulator registers required by each work-item.
  uint32_t AGPRCount = KInvalidValue;
  /// Corresponds to the OpenCL reqd_work_group_size attribute.
  uint32_t RequestedWorkgroupSize[3] = {KInvalidValue, KInvalidValue,
                                        KInvalidValue};
  /// Corresponds to the OpenCL work_group_size_hint attribute.
  uint32_t WorkgroupSizeHint[3] = {KInvalidValue, KInvalidValue, KInvalidValue};
  /// Wavefront size.
  uint32_t WavefrontSize = KInvalidValue;
  /// Maximum flat work-group size supported by the kernel in work-items.
  uint32_t MaxFlatWorkgroupSize = KInvalidValue;
};

/// Reads AMDGPU specific metadata from the ELF file and propagates the
/// KernelInfoMap.
Error getAMDGPUMetaDataFromImage(MemoryBufferRef MemBuffer,
                                 StringMap<AMDGPUKernelMetaData> &KernelInfoMap,
                                 uint16_t &ELFABIVersion);
} // namespace amdgpu
} // namespace offloading
} // namespace llvm

#endif // LLVM_FRONTEND_OFFLOADING_UTILITY_H
