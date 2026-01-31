//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Level Zero RTL Options support.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0OPTIONS_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0OPTIONS_H

#include <level_zero/ze_api.h>

#include "Shared/EnvironmentVar.h"

#include "L0Defs.h"

namespace llvm::omp::target::plugin {
/// Command submission mode.
enum class CommandModeTy { Sync = 0, Async, AsyncOrdered };

/// Specialization constants used for a module compilation.
class SpecConstantsTy {
  std::vector<uint32_t> ConstantIds;
  std::vector<const void *> ConstantValues;
  BumpPtrAllocator &Allocator;

public:
  SpecConstantsTy(BumpPtrAllocator &Allocator) : Allocator(Allocator) {}
  SpecConstantsTy(const SpecConstantsTy &) = delete;
  SpecConstantsTy(SpecConstantsTy &&) = delete;
  SpecConstantsTy &operator=(const SpecConstantsTy &) = delete;
  SpecConstantsTy &operator=(const SpecConstantsTy &&) = delete;
  SpecConstantsTy(const SpecConstantsTy &&Other)
      : ConstantIds(std::move(Other.ConstantIds)),
        ConstantValues(std::move(Other.ConstantValues)),
        Allocator(Other.Allocator) {}
  ~SpecConstantsTy() = default;

  template <typename T> void addConstant(uint32_t Id, T Val) {
    T *ValuePtr =
        reinterpret_cast<T *>(Allocator.Allocate(sizeof(T), alignof(T)));
    *ValuePtr = Val;

    ConstantIds.push_back(Id);
    ConstantValues.push_back(reinterpret_cast<void *>(ValuePtr));
  }

  ze_module_constants_t getModuleConstants() const {
    ze_module_constants_t Tmp{static_cast<uint32_t>(ConstantValues.size()),
                              ConstantIds.data(),
                              // Unfortunately we have to const_cast it.
                              // L0 data type should probably be fixed.
                              const_cast<const void **>(ConstantValues.data())};
    return Tmp;
  }
};

/// L0 Plugin flags.
struct L0OptionFlagsTy {
  uint64_t UseMemoryPool : 1;
  uint64_t Reserved : 63;
  L0OptionFlagsTy() : UseMemoryPool(1), Reserved(0) {}
};

struct L0OptionsTy {
  /// Binary flags.
  L0OptionFlagsTy Flags;

  /// Staging buffer size.
  size_t StagingBufferSize = L0StagingBufferSize;

  /// Staging buffer count.
  size_t StagingBufferCount = L0StagingBufferCount;

  struct MemPoolConfigTy {
    bool Use;
    int32_t AllocMax;
    int32_t Capacity;
    int32_t PoolSize;
  };
  /// Memory pool default parameters for each allocation kind:
  /// {UseByDefault, AllocMax(MB), Capacity, PoolSize(MB)}
  std::array<MemPoolConfigTy, 3> MemPoolConfig{
      MemPoolConfigTy{true, 1, 4, 256},  // TARGET_ALLOC_DEVICE
      MemPoolConfigTy{true, 1, 4, 256},  // TARGET_ALLOC_HOST
      MemPoolConfigTy{true, 8, 4, 256}}; // TARGET_ALLOC_SHARED

  /// Parameters for memory pools dedicated to reduction scratch space.
  std::array<int32_t, 3> ReductionPoolInfo{256, 8, 8192};

  /// Oversubscription rate for normal kernels.
  uint32_t SubscriptionRate = 4;

  /// Loop kernels with known ND-range may be known to have
  /// few iterations and they may not exploit the offload device
  /// to the fullest extent.
  /// Let's assume a device has N total HW threads available,
  /// and the kernel requires M hardware threads with LWS set to L.
  /// If (M < N * ThinThreadsThreshold), then we will try
  /// to iteratively divide L by 2 to increase the number of HW
  /// threads used for executing the kernel. Effectively, we will
  /// end up with L less than the kernel's SIMD width, so the HW
  /// threads will not use all their SIMD lanes. This (presumably) should
  /// allow more parallelism, because the stalls in the SIMD lanes
  /// will be distributed across more HW threads, and the probability
  /// of having a stall (or a sequence of stalls) on a critical path
  /// in the kernel should decrease.
  /// Anyway, this is just a heuristics that seems to work well for some
  /// kernels (which poorly expose parallelism in the first place).
  double ThinThreadsThreshold = 0.1;

  // Compilation options for IGC.
  // OpenCL 2.0 builtins (like atomic_load_explicit and etc.) are used by
  // runtime, so we have to explicitly specify the "-cl-std=CL2.0" compilation
  // option. With it, the SPIR-V will be converted to LLVM IR with OpenCL 2.0
  // builtins. Otherwise, SPIR-V will be converted to LLVM IR with OpenCL 1.2
  // builtins.
  static constexpr std::string_view CompilationOptions = "-cl-std=CL2.0 ";
  static constexpr std::string_view InternalCompilationOptions =
      "-cl-take-global-address";
  std::string UserCompilationOptions;

  /// Spec constants used for all modules.
  SpecConstantsTy CommonSpecConstants;

  /// Command execution mode.
  /// Whether the runtime uses asynchronous mode or not depends on the type of
  /// devices and whether immediate command list is fully enabled.
  CommandModeTy CommandMode = CommandModeTy::Async;

  /// Controls if we need to reduce available HW threads. We need this
  /// adjustment on XeHPG when Level Zero debug is enabled
  /// (ZET_ENABLE_PROGRAM_DEBUGGING=1).
  bool ZeDebugEnabled = false;

  bool Init = false; // Have the options already been processed.

  // Allocator for long-lived allocations (e.g. spec constants).
  BumpPtrAllocator Allocator;

  L0OptionsTy() : CommonSpecConstants(Allocator) {}

  /// Read environment variables.
  void processEnvironmentVars();

  void init() {
    if (!Init) {
      processEnvironmentVars();
      Init = true;
    }
  }

  bool match(const StringEnvar &Var, const llvm::StringRef Matched) {
    return Matched.equals_insensitive(Var.get());
  }

}; // L0OptionsTy

} // namespace llvm::omp::target::plugin

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0OPTIONS_H
