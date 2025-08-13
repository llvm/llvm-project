//===- Interface.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Interface.h"
#include "Debug.h"
#include "DeviceManager.h"
#include "Logger.h"
#include "OpenMP/Mapping.h"
#include "PluginManager.h"
#include "Private.h"
#include "QueueManager.h"
#include "Shared/APITypes.h"
#include "Shared/Debug.h"
#include "Shared/SourceInfo.h"
#include "device.h"
#include "omptarget.h"
#include "openacc.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <string.h>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

using namespace llvm::acc::target;
using namespace llvm::acc::target::debug;

using llvm::SmallVector;

namespace {
constexpr int32_t AccAsyncSync = acc_async_sync;
constexpr int32_t AccAsyncDefault = acc_async_default;
constexpr int32_t AccAsyncNoval = acc_async_noval;
constexpr int32_t AccAsyncDefaultQueue = -5;
} // namespace

namespace llvm::acc::target::icv {
// acc-default-async-var
thread_local int32_t AccDefaultAsyncVar = AccAsyncDefaultQueue;
} // namespace llvm::acc::target::icv

namespace {
// TODO hook up to some env var
bool Pedantic = true;

#define STR_AND_VAL(X) #X << " " << X
#define STR_AND_VALI(X) #X << " " << (int64_t)X
#define OPT_STR_AND_VAL(X) #X << " " << (X ? std::to_string(*X) : "(nil)")
#define SOPT_STR_AND_VAL(X) #X << " " << (X ? X : "(nil)")

struct DescMappingInfoTy {
  // The size of the descriptor
  size_t DescriptorSize = 0;
  // The offset in the host descriptor where the pointer to the raw memory is
  // stored.
  size_t RawMemoryPtrOffset = 0;
};

struct MemMappingInfoTy {
  void *RawMemoryPtr = nullptr;
  void *RawMemoryBasePtr = nullptr;
  std::optional<uint64_t> RawMemorySize = std::nullopt;
  std::optional<NonContigDescTy> CopyDesc = std::nullopt;

  ptrdiff_t getBaseDelta() {
    ptrdiff_t Delta = reinterpret_cast<intptr_t>(RawMemoryPtr) -
                      reinterpret_cast<intptr_t>(RawMemoryBasePtr);
    return Delta;
  }
  void verify() { assert(RawMemoryPtr); }
  void dump(llvm::raw_ostream &OS) {
    OS << "MemMappingInfoTy:\n";
    // clang-format off
    OS
        << " " << STR_AND_VAL(RawMemoryPtr)
        << " " << STR_AND_VAL(RawMemoryBasePtr)
        << " " << OPT_STR_AND_VAL(RawMemorySize)
        << " " << !!CopyDesc
        << "\n";
    // clang-format on
  }
};

struct AccArrayDim {
  long Offset;
  long Stride;
  long Size;
  long Extent;
};

struct ArrayInfo {
  std::vector<AccArrayDim> Dims;
  // The size of the raw memory allocation.
  std::optional<uint64_t> RawMemorySize = {};
  // The address of the host memory to be copied.
  void *RawMemoryAddr = nullptr;
  // Size of the array element.
  int64_t ElementSize = 0;

  void setPtr(void *Ptr) { RawMemoryAddr = Ptr; }

  std::optional<size_t> getSizeInDim(ident_t *Loc, unsigned I) {
    if (Dims[I].Stride < 0) {
      REPORT_FATAL() << Loc << "Unsupported negative stride";
    }

    auto TrySize = [&](int64_t Size) -> std::optional<size_t> {
      if (Size == -1) {
        return std::nullopt;
      }
      return Size * Dims[I].Stride * ElementSize;
    };

    // Prefer the `size` instead of `extent`. This is due to cases like this:
    //
    // real a0(100)
    // call acc_copyin(a0(1:99))
    // !$acc present(a0(1:99))
    //
    // Where the `acc_copyin` will allocate space for 99 elements because we
    // parse the flang descriptor which only contains information on the size
    // (99), but then if we use the `extent` from the `acc present`, we would
    // require 100 elements, which is larger than the previously allocated
    // memory. Thus, we use the `size`.

    if (auto Size = TrySize(Dims[I].Size)) {
      return Size;
    }
    if (auto Size = TrySize(Dims[I].Extent)) {
      return Size;
    }

    return std::nullopt;
  }

  void computeSizeFromDims(ident_t *Loc) {
    if (Dims.size() == 0) {
      RawMemorySize = ElementSize;
      return;
    }

    std::optional<size_t> LargestSize = getSizeInDim(Loc, Dims.size() - 1);
    RawMemorySize = LargestSize;
    ODBG(ADT_Descriptor) << "Computed " << OPT_STR_AND_VAL(RawMemorySize);

#ifndef NDEBUG
    if (!LargestSize) {
      return;
    }
    for (unsigned I = 0; I < Dims.size() - 1; I++) {
      auto Size = getSizeInDim(Loc, I);
      assert(!Size || *Size <= *LargestSize);
    }
#endif
  }

  void normalize() {
    normalizeStrides();
    normalizeOffsets();
  }

  bool hasNormalizedStrides() {
    for (std::size_t i = 0; i < Dims.size(); i++) {
      if (Dims[i].Stride < 0) {
        return false;
      }
    }
    return true;
  }

  void normalizeStrides() {
    FUNC_LOGGER();
    ODBG_IF([&]() { dump(llvm::dbgs()); });

    if (hasNormalizedStrides()) {
      ODBG(ADT_Descriptor) << "No normalization needed.";
      return;
    }
    ODBG(ADT_Descriptor) << "Descriptor needs normalization.";

    // The runtime cannot map negative stride arrays.  So we must find the base
    // address of the host pointer and then invert the descriptor so that the
    // strides in all dimensions are positive. The base pointer delta will be
    // used to attach the adjusted device pointer to the array descriptor - that
    // is, the F18 descriptor will contain the end address of the array because
    // that is what the compiler assumes.
    int64_t baseHostPtrDeltaInBytes = 0;
    for (std::size_t i = 0; i < Dims.size(); i++) {
      if (Dims[i].Stride < 0) {
        Dims[i].Stride = -Dims[i].Stride;
        Dims[i].Offset = Dims[i].Extent - Dims[i].Size - Dims[i].Offset;

        // For each negative stride, skip to previously accounted array.
        baseHostPtrDeltaInBytes += Dims[i].Stride * (Dims[i].Extent - 1);
      }
    }

    baseHostPtrDeltaInBytes *= ElementSize;

    RawMemoryAddr =
        reinterpret_cast<char *>(RawMemoryAddr) - baseHostPtrDeltaInBytes;

    ODBG(ADT_Descriptor) << "Normalized:";
    ODBG_IF([&]() { dump(llvm::dbgs()); });
  }

  bool hasNormalizedOffsets() {
    for (std::size_t i = 0; i < Dims.size(); i++) {
      if (Dims[i].Offset != 0) {
        return false;
      }
    }
    return true;
  }

  void normalizeOffsets() {
    FUNC_LOGGER();
    ODBG_IF([&]() { dump(llvm::dbgs()); });
    assert(hasNormalizedStrides());

    if (hasNormalizedOffsets()) {
      ODBG(ADT_Descriptor) << "No normalization needed.";
      return;
    }
    ODBG(ADT_Descriptor) << "Descriptor needs normalization.";

    int64_t baseHostPtrDeltaInBytes = 0;
    for (auto &Dim : Dims) {
      if (Dim.Offset != 0) {
        baseHostPtrDeltaInBytes += Dim.Offset * Dim.Stride;
        Dim.Offset = 0;
      }
    }

    baseHostPtrDeltaInBytes *= ElementSize;

    RawMemoryAddr =
        reinterpret_cast<char *>(RawMemoryAddr) + baseHostPtrDeltaInBytes;

    ODBG(ADT_Descriptor) << "Normalized:";
    ODBG_IF([&]() { dump(llvm::dbgs()); });
  }

  void verify() {
    assert(ElementSize > 0);
    for (unsigned I = 0; I < Dims.size() - 1; I++) {
      assert(Dims[I].Stride < Dims[I + 1].Stride &&
             "Expected dimensions to be sorted");
    }
  }

  void dump(llvm::raw_ostream &OS) {
    OS << "ArrayInfo:\n";
    for (unsigned I = 0; I < Dims.size(); I++) {
      // clang-format off
      OS << "      Dim " << I
        << "\t" << STR_AND_VAL(Dims[I].Offset)
        << "\t" << STR_AND_VAL(Dims[I].Size)
        << "\t" << STR_AND_VAL(Dims[I].Stride)
        << "\t" << STR_AND_VAL(Dims[I].Extent)
        << "\n";
      // clang-format on
    }
    // clang-format off
    OS << "    "
        << " " << STR_AND_VAL(RawMemoryAddr)
        << " " << OPT_STR_AND_VAL(RawMemorySize)
        << " " << STR_AND_VAL(ElementSize)
        << "\n";
    // clang-format on
  };

  /// See the llvm-project/offload/test/offloading/non_contiguous_update.cpp
  /// test for examples.
  std::optional<NonContigDescTy> generateNonContigCopyDesc(ident_t *Loc) {
    NonContigDescTy CopyDesc;
    CopyDesc.Dims.reserve(Dims.size() + 1);

    for (int I = Dims.size() - 1; I >= 0; I--) {
      auto const &Dim = Dims[I];
      if (Dim.Size < 0) {
        ODBG(ADT_Descriptor)
            << "Dim size missing, cannot build copy descriptor";
        return std::nullopt;
      }
      CopyDesc.Dims.push_back({});
      auto &LastDim = CopyDesc.Dims.back();
      LastDim.Count = Dim.Size;
      LastDim.Stride = Dim.Stride * ElementSize;
      LastDim.Offset = Dim.Offset * LastDim.Stride;
    }

    CopyDesc.Dims.push_back({});
    auto &LastDim = CopyDesc.Dims.back();
    LastDim.Count = ElementSize;
    LastDim.Offset = 0;
    LastDim.Stride = 1;

    return CopyDesc;
  }
};

struct MaterializedMemRefDesc {
  void *allocatedPtr;
  void *alignedPtr;
  uint64_t offset;
  int64_t elementSize;
  unsigned char rank;
  const uint64_t *sizes;
  const uint64_t *strides;
};

void dump(const MaterializedMemRefDesc &Desc, llvm::raw_ostream &OS) {
  // clang-format off
  OS
      << " " << STR_AND_VAL(Desc.allocatedPtr)
      << " " << STR_AND_VAL(Desc.alignedPtr)
      << " " << STR_AND_VAL(Desc.offset)
      << " " << STR_AND_VAL(Desc.elementSize)
      << " " << STR_AND_VALI(Desc.rank)
      << "\n";
  for (unsigned I = 0; I < Desc.rank; I++) {
    OS << "Dim " << I
        << " " << STR_AND_VAL(Desc.sizes[I])
        << " " << STR_AND_VAL(Desc.strides[I])
        << "\n";
  }
  // clang-format on
}

void dump(const AccDataDescOpenACC &Desc, llvm::raw_ostream &OS) {
  // clang-format off
  OS
      << " " << STR_AND_VAL(Desc.Base.Version)
      << " " << STR_AND_VALI(Desc.Rank)
      << " " << STR_AND_VAL(Desc.ElementSize)
      << "\n";
  for (unsigned I = 0; I < Desc.Rank; I++) {
    OS << "Dim " << I
        << " " << STR_AND_VAL(Desc.LowerBounds[I])
        << " " << STR_AND_VAL(Desc.UpperBounds[I])
        << " " << STR_AND_VAL(Desc.Extents[I])
        << " " << STR_AND_VAL(Desc.StridesInBytes[I])
        << " " << STR_AND_VAL(Desc.StartIndices[I])
        << "\n";
  }
  // clang-format on
}

template <class... Ts> struct overloads : Ts... {
  using Ts::operator()...;
};
template <class... Ts> overloads(Ts...) -> overloads<Ts...>;

std::string asyncToString(int64_t Async) {
  if (Async >= 0) {
    return "STREAM(" + std::to_string(Async) + ")";
  } else if (Async == AccAsyncSync) {
    return "SYNC";
  } else if (Async == AccAsyncDefault) {
    return "DEFAULT";
  } else if (Async == AccAsyncNoval) {
    return "NOVAL";
  } else {
    return "UNKNOWN";
  }
}

std::string mapTypeToString(int64_t Type) {
  std::stringstream SS;

  if (Type & TGT_ACC_MAPTYPE_TO)
    SS << "TO ";
  if (Type & TGT_ACC_MAPTYPE_FROM)
    SS << "FROM ";
  if (Type & TGT_ACC_MAPTYPE_FINALIZE)
    SS << "DELETE ";
  if (Type & TGT_ACC_MAPTYPE_PTR_AND_OBJ)
    SS << "PTR_AND_OBJ ";
  if (Type & TGT_ACC_MAPTYPE_PRIVATE)
    SS << "PRIVATE ";
  if (Type & TGT_ACC_MAPTYPE_LITERAL)
    SS << "LITERAL ";
  if (Type & TGT_ACC_MAPTYPE_DEVPTR)
    SS << "DEVPTR ";
  if (Type & TGT_ACC_MAPTYPE_MANAGED_DEVPTR)
    SS << "MANAGED_DEVPTR ";
  if (Type & TGT_ACC_MAPTYPE_NO_CREATE)
    SS << "NO_CREATE ";
  if (Type & TGT_ACC_MAPTYPE_GANG_PRIVATE)
    SS << "GANG_PRIVATE ";
  if (Type & TGT_ACC_MAPTYPE_WORKER_PRIVATE)
    SS << "WORKER_PRIVATE ";
  if (Type & TGT_ACC_MAPTYPE_VECTOR_PRIVATE)
    SS << "VECTOR_PRIVATE ";
  if (Type & TGT_ACC_MAPTYPE_INIT_ZERO)
    SS << "INIT_ZERO ";
  if (Type & TGT_ACC_MAPTYPE_DEVICE_RESIDENT)
    SS << "DEVICE_RESIDENT ";
  if (Type & TGT_ACC_MAPTYPE_IF_PRESENT)
    SS << "IF_PRESENT ";

  std::string Str = SS.str();
  if (Str.empty())
    return "(none)";
  else
    // Remove trailing space.
    Str.resize(Str.size() - 1);

  return Str;
}

enum class AccCopyOutType { Always, OnDelete, Never };
enum class AccRefCountingType { Dynamic, Structured };

struct PostProcessingInfo {
  /// The target pointer information.
  TargetPointerResultTy TPR;
  int64_t DataSize;
  bool ShouldRestoreShadow;
  bool ShouldDelete;
};

struct KernelArgsMappingInfoTy {
  AccKernelArgsTy &KernelArgs;

  // Memory needed for launch
  void addLaunchAlloc(void *Alloc) { LaunchAllocs.push_back(Alloc); }
  SmallVector<void *> LaunchAllocs;

  // Arguments
  void addArg(void *Arg) { Args.push_back(Arg); }
  SmallVector<void *> Args;
  SmallVector<void *> Ptrs;

  KernelLaunchParamsTy getLaunchArgs() {
    assert(Ptrs.size() == 0);

    if (Args.size() == 0)
      return KernelLaunchParamsTy{};

    unsigned NumArgs = Args.size();
    Ptrs.resize(NumArgs);
    for (uint32_t I = 0; I < NumArgs; ++I)
      Ptrs[I] = &Args[I];
    return KernelLaunchParamsTy{sizeof(void *) * NumArgs, &Args[0], &Ptrs[0]};
  }
};

[[nodiscard]] int accPostProcessingTargetDataEnd(DeviceTy *Device,
                                                 PostProcessingInfo *Info) {
  // This will make sure we delete it when we exit the function.
  std::unique_ptr<PostProcessingInfo> InfoDeleter(Info);

  int Ret = OFFLOAD_SUCCESS;

  assert(!Info->TPR.isHostPointer());

  MappingInfoTy::HDTTMapAccessorTy HDTTMap =
      Device->getMappingInfo().HostDataToTargetMap.getExclusiveAccessor();

  // We cannot use a lock guard because we may end up delete the mutex.
  // We also explicitly unlocked the entry after it was put in the EntriesInfo
  // so it can be reused.
  Info->TPR.getEntry()->lock();
  auto *Entry = Info->TPR.getEntry();

  // TODO I do not understand why this is necessary - does the mapping
  // automatically queue up entry deletion?
  bool DelEntry = Info->ShouldDelete;
  const bool IsNotLastUser = Entry->decDataEndThreadCount() != 0;
  if (DelEntry && (Entry->getTotalRefCount() != 0 || IsNotLastUser)) {
    ODBG(ADT_Mapping) << "IsNotLastUser";
    // The thread is not in charge of deletion anymore. Give up access
    // to the HDTT map and unset the deletion flag.
    HDTTMap.destroy();
    DelEntry = false;
  }

  if (Info->ShouldRestoreShadow) {
    Entry->foreachShadowPointerInfo([&](const ShadowPtrInfoTy &ShadowPtr) {
      ODBG(ADT_Mapping) << "Restoring host shadow "
                        << (void *)ShadowPtr.HstPtrAddr
                        << " to its original content (" << ShadowPtr.PtrSize
                        << " bytes)";
      std::memcpy(ShadowPtr.HstPtrAddr, ShadowPtr.HstPtrContent.data(),
                  ShadowPtr.PtrSize);
      return OFFLOAD_SUCCESS;
    });
  }

  // Give up the lock as we either don't need it anymore (e.g., done with
  // TPR), or erase TPR.
  Info->TPR.setEntry(nullptr);

  if (!Info->ShouldDelete)
    return Ret;

  Ret = Device->getMappingInfo().eraseMapEntry(HDTTMap, Entry, Info->DataSize);
  // Entry is already remove from the map, we can unlock it now.
  HDTTMap.destroy();
  Ret |= Device->getMappingInfo().deallocTgtPtrAndEntry(Entry, Info->DataSize);
  if (Ret != OFFLOAD_SUCCESS)
    REPORT_FATAL() << "Deallocating data from device failed.";

  return OFFLOAD_SUCCESS;
}

template <typename SizeTy>
void handleSingleDataEnd(ident_t *Loc, void *ArgBasePtr, void *ArgPtr,
                         SizeTy ArgSize, bool ForceDelete, bool IsNoCreate,
                         AccCopyOutType CopyType, AccRefCountingType MapType,
                         AsyncInfoTy &AsyncInfo, DeviceTy &Device) {
  int64_t DataSize;
  if constexpr (std::is_same<SizeTy, int64_t>::value) {
    DataSize = ArgSize;
  } else if constexpr (std::is_same<SizeTy, NonContigDescTy &>::value) {
    DataSize = ArgSize.getAllocSize();
  } else {
    static_assert(false);
  }

  FUNC_LOGGER();
  TargetPointerResultTy TPR = Device.getMappingInfo().getTgtPtrBegin(
      ArgPtr, DataSize, /*UpdateRefCount=*/true,
      /*HasHoldModifier=*/MapType == AccRefCountingType::Structured, IsNoCreate,
      ForceDelete,
      /*FromDataEnd=*/true);
  if (!TPR.isPresent()) {
    ODBG(ADT_Mapping) << "Mapping does not exist: "
                      << (IsNoCreate ? "is no_create" : "error");
    if (Pedantic && !IsNoCreate)
      REPORT_FATAL() << "Device mapping does not exist at " << Loc;
    return;
  }

  void *HstPtrBegin = ArgPtr;
  void *TgtPtrBegin = TPR.TargetPointer;
  ODBG(ADT_Mapping) << "There are " << DataSize
                    << " bytes allocated at target address " << TgtPtrBegin
                    << " - is" << (TPR.Flags.IsLast ? "" : " not") << " last";

  bool ShouldDelete = ForceDelete || TPR.Flags.IsLast;
  bool ShouldCopyOut = CopyType == AccCopyOutType::Always ||
                       (CopyType == AccCopyOutType::OnDelete && ShouldDelete);
  if (ShouldCopyOut) {
    ODBG(ADT_Mapping) << "Moving " << DataSize << " bytes (tgt:" << TgtPtrBegin
                      << ") -> (hst:" << HstPtrBegin << ")";
    int Ret;
    if constexpr (std::is_same<SizeTy, int64_t>::value) {
      Ret = Device.retrieveData(HstPtrBegin, TgtPtrBegin, DataSize, AsyncInfo,
                                TPR.getEntry());
    } else if constexpr (std::is_same<SizeTy, NonContigDescTy &>::value) {
      Ret = Device.retrieveNonContigData(HstPtrBegin, TgtPtrBegin, ArgSize,
                                         AsyncInfo, TPR.getEntry());
    }
    if (Ret != OFFLOAD_SUCCESS)
      REPORT_FATAL() << "Failed to transfer data from device at " << Loc;
  }

  ODBG(ADT_Mapping) << "Queueing up post processing";

  // TODO We may want to have a more intricate system for queueing up post
  // processing. In OpenACC, we could potentially queue up a lot of stream
  // operations before syncing, and we only execute these post processing
  // functions after we sync. This would leave a lot of deallocation and
  // unmapping queued for post processing but never happening because the
  // operations in the stream are continuing to execute while we add more post
  // processing funcitons which we never execute.
  //
  // One option here is to use Device.enqueueHostCall, however, for example, for
  // CUDA, having cuFree's etc in a function executed in the stream is not
  // supported (the context in that thread is invalid for calling cuda
  // functions). Instead, we can have the host call "notify" that we can execute
  // specific post processing functions, and we execute them at some point
  // during execution on the normal threads.
  auto *PostProcessingPtr = new PostProcessingInfo{std::move(TPR), DataSize,
                                                   ShouldCopyOut, ShouldDelete};
  PostProcessingPtr->TPR.getEntry()->unlock();
  AsyncInfo.addPostProcessingFunction([=, Device = &Device]() -> int {
    return accPostProcessingTargetDataEnd(Device, PostProcessingPtr);
  });
}

struct DescAndMemMappingInfoTy {
  DescMappingInfoTy Desc;
  std::optional<MemMappingInfoTy> Memory;
};

struct ArgDescriptorsTy {

  const Fortran::runtime::Descriptor *Flang = nullptr;
  std::optional<MaterializedMemRefDesc> MemRef = std::nullopt;
  const AccDataDescOpenACC *Acc = nullptr;

  bool isNone() { return !Flang && !MemRef && !Acc; }

  void verify() {
    assert(!(Flang && MemRef));
    assert(!isNone());
  }

  void dump(llvm::raw_ostream &OS) {
    OS << "ArgDescriptorsTy:\n";
    OS << "Flang:\n";
    if (Flang) {
      // TODO can we use OS somehow?
      Flang->Dump(stderr);
    } else {
      OS << "(nil)\n";
    }
    OS << "MemRef:\n";
    if (MemRef) {
      ::dump(*MemRef, OS);
    } else {
      OS << "(nil)\n";
    }
    OS << "Acc:\n";
    if (Acc) {
      ::dump(*Acc, OS);
    } else {
      OS << "(nil)\n";
    }
  }

  void collectAccBounds(ident_t *Loc, ArrayInfo &AI) {
    assert(Acc);
    if (AI.ElementSize <= 0) {
      REPORT_FATAL() << Loc << "Invalid element size";
    }

    AI.Dims.reserve(Acc->Rank);
    for (std::size_t I = 0; I < Acc->Rank; I++) {
      AI.Dims.push_back({});
      auto &ThisDim = AI.Dims.back();
      long SizeFactor = 1;
      if (Acc->StridesInBytes[I] % AI.ElementSize != 0) {
        if (I == 0) {
          // `stride` in AccArrayDim is meant to be multiplied by elementsize.
          // But the stride of a sliced descriptor array might not be divisible
          // by the current element size. So, reduce elementsize.
          SizeFactor =
              AI.ElementSize / std::gcd(AI.ElementSize, Acc->StridesInBytes[I]);
          AI.ElementSize /= SizeFactor;
        } else {
          REPORT_FATAL() << Loc << "Invalid array stride";
        }
      }
      ThisDim.Offset = Acc->LowerBounds[I];
      ThisDim.Stride = Acc->StridesInBytes[I] / AI.ElementSize;
      ThisDim.Size =
          SizeFactor * (Acc->UpperBounds[I] - Acc->LowerBounds[I] + 1);
      ThisDim.Extent = Acc->Extents[I];
    }
  }

  void collectFlangBounds(ident_t *Loc, ArrayInfo &AI) {
    assert(Flang);
    if (AI.ElementSize <= 0) {
      REPORT_FATAL() << Loc << "Invalid element size";
    }

    AI.Dims.reserve(Flang->rank());
    AI.ElementSize = Flang->ElementBytes();
    for (int I = 0; I < Flang->rank(); I++) {
      AI.Dims.push_back({});
      auto &ThisDim = AI.Dims.back();
      auto &FlangDim = Flang->GetDimension(I);
      long SizeFactor = 1;
      if (FlangDim.ByteStride() % AI.ElementSize != 0) {
        if (I == 0) {
          SizeFactor =
              AI.ElementSize / std::gcd(AI.ElementSize, FlangDim.ByteStride());
          AI.ElementSize /= SizeFactor;
        } else {
          REPORT_FATAL() << Loc << "Invalid array stride";
        }
      }
      ThisDim.Offset = 0;
      ThisDim.Stride = FlangDim.ByteStride() / AI.ElementSize;
      ThisDim.Size = SizeFactor * FlangDim.Extent();
    }
  }

  using LiteralArg = void *;
  using ArgMappingInfoTy =
      std::variant<DescAndMemMappingInfoTy, MemMappingInfoTy, LiteralArg>;
  using ArgMappingInfosTy = std::vector<ArgMappingInfoTy>;

  ArgMappingInfosTy getMappingInfos(ident_t *Loc, void *Ptr) {
    if (Flang) {
      auto DMI = DescMappingInfoTy{Flang->SizeInBytes(),
                                   offsetof(CFI_cdesc_t, base_addr)};
      if (!Flang->IsAllocated()) {
        ODBG() << "Is not allocated - nothing to map.";
        ArgMappingInfosTy MIs;
        MIs.emplace_back(DescAndMemMappingInfoTy{DMI, std::nullopt});
        return MIs;
      }

      ArrayInfo AI;
      AI.ElementSize = Flang->ElementBytes();
      if (AI.ElementSize == 0 && Acc) {
        AI.ElementSize = Acc->ElementSize;
      }
      AI.setPtr(Flang->OffsetElement(0));

      if (Acc) {
        collectAccBounds(Loc, AI);
      } else {
        collectFlangBounds(Loc, AI);
      }
      AI.normalize();
      AI.computeSizeFromDims(Loc);

      auto MMI = MemMappingInfoTy{};
      MMI.RawMemoryPtr = AI.RawMemoryAddr;
      MMI.RawMemoryBasePtr = Flang->OffsetElement(0);
      MMI.RawMemorySize = AI.RawMemorySize;
      MMI.CopyDesc = AI.generateNonContigCopyDesc(Loc);

      ArgMappingInfosTy MIs;
      MIs.emplace_back(DescAndMemMappingInfoTy{DMI, std::move(MMI)});
      return MIs;
    } else if (MemRef) {
      if (Acc) {
        REPORT_FATAL() << Loc << "Unsupported: MemRef with OpenACC bounds";
      }

      ArgMappingInfosTy MIs;

      uint64_t Extent = 1LL;
      for (ssize_t I = (ssize_t)MemRef->rank - 1; I >= 0; I--) {
        Extent *= MemRef->sizes[I];
        if (Extent != MemRef->strides[I] * MemRef->sizes[I]) {
          REPORT_FATAL() << Loc << "Invalid memref descriptor";
        }
      }
      Extent *= MemRef->elementSize;

      {
        auto MMI = MemMappingInfoTy{};
        MMI.RawMemoryPtr = MemRef->allocatedPtr;
        MMI.RawMemoryBasePtr = MemRef->allocatedPtr;
        MMI.RawMemorySize = Extent;
        MIs.push_back(std::move(MMI));
      }
      {
        auto MMI = MemMappingInfoTy{};
        MMI.RawMemoryPtr = MemRef->alignedPtr;
        MMI.RawMemoryBasePtr = MemRef->allocatedPtr;
        MMI.RawMemorySize = Extent;
        MIs.push_back(std::move(MMI));
      }

      MIs.push_back(reinterpret_cast<void *>(MemRef->offset));

      for (size_t I = 0; I < MemRef->rank; I++) {
        MIs.push_back(reinterpret_cast<void *>(MemRef->sizes[I]));
        MIs.push_back(reinterpret_cast<void *>(MemRef->strides[I]));
      }

      return MIs;
    } else if (Acc) {
      ArrayInfo AI;
      AI.ElementSize = Acc->ElementSize;
      AI.setPtr(Ptr);
      collectAccBounds(Loc, AI);
      AI.normalize();
      AI.computeSizeFromDims(Loc);

      auto MMI = MemMappingInfoTy{};
      MMI.RawMemoryPtr = AI.RawMemoryAddr;
      MMI.RawMemoryBasePtr = Ptr;
      MMI.RawMemorySize = AI.RawMemorySize;
      MMI.CopyDesc = AI.generateNonContigCopyDesc(Loc);

      ArgMappingInfosTy MIs;
      MIs.push_back(std::move(MMI));
      return MIs;
    } else {
      REPORT_FATAL() << Loc << "Unknown case.";
      abort();
    }
  }

  void dataBeginPrivate(ident_t *Loc, void *ArgPtr, int64_t ArgSize,
                        bool HasFlagTo, DeviceTy &Device,
                        AsyncInfoTy &AsyncInfo,
                        MappingInfoTy::HDTTMapAccessorTy &HDTTMap,
                        KernelArgsMappingInfoTy &KI) {
    assert(!MemRef);

    if (Flang) {
      REPORT_FATAL() << "TODO Flang descriptor on private variable";
    } else if (Acc) {
      assert(ArgPtr);
      ArrayInfo AI;
      AI.ElementSize = Acc->ElementSize;
      AI.setPtr(ArgPtr);
      collectAccBounds(Loc, AI);
      AI.normalize();
      AI.computeSizeFromDims(Loc);

      assert(AI.RawMemorySize);
      size_t DataSize = *AI.RawMemorySize;
      void *HostData = AI.RawMemoryAddr;
      ptrdiff_t Offset = reinterpret_cast<intptr_t>(HostData) -
                         reinterpret_cast<intptr_t>(ArgPtr);

      ODBG(ADT_Interface) << "ACC firstprivate (partial): dataSize=" << DataSize
                          << " hostData=" << HostData << " (base=" << ArgPtr
                          << ")";

      void *PrivateMemory =
          Device.allocData(DataSize, nullptr, TARGET_ALLOC_DEVICE);
      Device.submitData(PrivateMemory, HostData, DataSize, AsyncInfo,
                        /*Entry=*/nullptr, &HDTTMap);
      KI.addArg(static_cast<char *>(PrivateMemory) - Offset);
      KI.addLaunchAlloc(PrivateMemory);
    } else {
      REPORT_FATAL() << Loc << "Unknown descriptor type for private variable";
    }
  }

  void dataBeginDevPtr(ident_t *Loc, DeviceTy &Device, AsyncInfoTy &AsyncInfo,
                       MappingInfoTy::HDTTMapAccessorTy &HDTTMap,
                       KernelArgsMappingInfoTy &KI) {
    assert(Flang && !MemRef);
    size_t DescSize = Flang->SizeInBytes();
    void *DevDesc = Device.allocData(DescSize, nullptr, TARGET_ALLOC_DEVICE);
    Device.submitData(DevDesc,
                      const_cast<void *>(static_cast<const void *>(Flang)),
                      DescSize, AsyncInfo, /*Entry=*/nullptr, &HDTTMap);
    KI.addArg(DevDesc);
    KI.addLaunchAlloc(DevDesc);
  }

  void dataBegin(ident_t *Loc, void *ArgPtr, void *DescriptorAddr,
                 void *&ParentAllocation, bool IsPtrAndObj, char *ArgName,
                 bool HasFlagTo, bool IsNoCreate, AccRefCountingType MapType,
                 AsyncInfoTy &AsyncInfo, DeviceTy &Device,
                 MappingInfoTy::HDTTMapAccessorTy &HDTTMap,
                 KernelArgsMappingInfoTy *KI) {
    auto AddArg = [&](TargetPointerResultTy &TPR, void *TgtArg, void *HstArg) {
      if (KI) {
        if (TPR.isPresent()) {
          KI->addArg(TgtArg);
        } else {
          assert(IsNoCreate);
          KI->addArg(HstArg);
        }
      }
    };
    auto MapWithDesc = [&](MemMappingInfoTy &MemInfo, void *BasePtr,
                           bool IsParam) -> void * {
      assert(MemInfo.CopyDesc);
      ODBG() << "Will use non-contig copy.";

      TargetPointerResultTy TPR = Device.getMappingInfo().getTargetPointer(
          HDTTMap, MemInfo.RawMemoryPtr, MemInfo.RawMemoryBasePtr, 0,
          &*MemInfo.CopyDesc, ArgName, HasFlagTo,
          /*HasFlagAlways=*/false, /*IsImplicit=*/false,
          /*UpdateRefCount=*/true, /*HasCloseModifier=*/false,
          /*HasPresentModifier=*/false,
          /*HasHoldModifier=*/MapType == AccRefCountingType::Structured,
          IsNoCreate, AsyncInfo, /*OwnedTPR=*/nullptr,
          /*ReleaseHDTTMap=*/false);
      if (IsParam)
        AddArg(TPR,
               reinterpret_cast<void *>(
                   (reinterpret_cast<intptr_t>(TPR.TargetPointer) -
                    MemInfo.getBaseDelta())),
               MemInfo.RawMemoryPtr);
      return TPR.TargetPointer;
    };

    auto MapInfos = getMappingInfos(Loc, ArgPtr);
    auto DescAndMemCase = [&](DescAndMemMappingInfoTy &MapInfo) {
      ODBG() << "Mapping desc and mem";
      auto &DescInfo = MapInfo.Desc;

      void *DescTgtPtr = nullptr;
      if (!ParentAllocation)
        ParentAllocation = DescriptorAddr;

      {
        // Always copy the descriptor to device. It is needed regardless of the
        // user-specified TO/FROM, and regardless of whether no_create is on or
        // not as the no_create can refer to the raw memory in the descriptor.
        TargetPointerResultTy TPR = Device.getMappingInfo().getTargetPointer(
            HDTTMap, DescriptorAddr, DescriptorAddr, 0, (int64_t)DescInfo.DescriptorSize,
            ArgName, /*HasFlagTo=*/true,
            /*HasFlagAlways=*/false, /*IsImplicit=*/false,
            /*UpdateRefCount=*/true, /*HasCloseModifier=*/false,
            /*HasPresentModifier=*/false,
            /*HasHoldModifier=*/MapType == AccRefCountingType::Structured,
            /*IsNoCreate=*/false, AsyncInfo, /*OwnedTPR=*/nullptr,
            /*ReleaseHDTTMap=*/false);
        DescTgtPtr = TPR.TargetPointer;
        AddArg(TPR, DescTgtPtr, DescriptorAddr);
      }

      void *MemTgtPtr = nullptr;
      if (MapInfo.Memory) {
        auto &MemInfo = *MapInfo.Memory;
        void *BasePtr =
            static_cast<char *>(DescriptorAddr) + DescInfo.RawMemoryPtrOffset;
        if (MemInfo.RawMemorySize) {
          if (MemInfo.CopyDesc) {
            MemTgtPtr = MapWithDesc(MemInfo, BasePtr, false);
          } else {
            TargetPointerResultTy TPR =
                Device.getMappingInfo().getTargetPointer(
                    HDTTMap, MemInfo.RawMemoryPtr, BasePtr, 0,
                    (int64_t)*MemInfo.RawMemorySize, ArgName, HasFlagTo,
                    /*HasFlagAlways=*/false, /*IsImplicit=*/false,
                    /*UpdateRefCount=*/true, /*HasCloseModifier=*/false,
                    /*HasPresentModifier=*/false,
                    /*HasHoldModifier=*/MapType ==
                        AccRefCountingType::Structured,
                    IsNoCreate, AsyncInfo, /*OwnedTPR=*/nullptr,
                    /*ReleaseHDTTMap=*/false);
            MemTgtPtr = TPR.TargetPointer;
          }
        } else {
          TargetPointerResultTy TPR = Device.getMappingInfo().getTargetPointer(
              HDTTMap, MemInfo.RawMemoryPtr, BasePtr, 0, (int64_t)*MemInfo.RawMemorySize,
              ArgName, HasFlagTo,
              /*HasFlagAlways=*/false, /*IsImplicit=*/false,
              /*UpdateRefCount=*/true, /*HasCloseModifier=*/false,
              /*HasPresentModifier=*/true,
              /*HasHoldModifier=*/MapType == AccRefCountingType::Structured,
              IsNoCreate, AsyncInfo, /*OwnedTPR=*/nullptr,
              /*ReleaseHDTTMap=*/false);
          MemTgtPtr = TPR.TargetPointer;
        }

        if (MemTgtPtr && DescTgtPtr) {
          LookupResult DescLR = Device.getMappingInfo().lookupMapping(
              HDTTMap, DescriptorAddr, DescInfo.DescriptorSize);
          auto *DescEntry = DescLR.TPR.getEntry();
          if (DescEntry) {
            uintptr_t TgtDescBase =
                DescEntry->TgtPtrBegin +
                (reinterpret_cast<uintptr_t>(DescriptorAddr) -
                 DescEntry->HstPtrBegin);
            void **TgtPtrAddr = reinterpret_cast<void **>(
                TgtDescBase + DescInfo.RawMemoryPtrOffset);
            void **HstPtrAddr = reinterpret_cast<void **>(
                reinterpret_cast<uintptr_t>(DescriptorAddr) +
                DescInfo.RawMemoryPtrOffset);

            void *HstBaseAddr = *HstPtrAddr;
            void *TgtPteeBase = reinterpret_cast<void *>(
                reinterpret_cast<uintptr_t>(MemTgtPtr) +
                (reinterpret_cast<uintptr_t>(HstBaseAddr) -
                 reinterpret_cast<uintptr_t>(MemInfo.RawMemoryPtr)));

            if (DescEntry->addShadowPointer(
                    ShadowPtrInfoTy{HstPtrAddr, TgtPtrAddr, TgtPteeBase,
                                    static_cast<int64_t>(sizeof(void *))})) {
              ODBG() << "DescAndMemCase attach: device field " << TgtPtrAddr
                     << " -> " << TgtPteeBase;
              void *&Buf = AsyncInfo.getVoidPtrLocation();
              Buf = TgtPteeBase;
              Device.submitData(TgtPtrAddr, &Buf, sizeof(void *), AsyncInfo,
                                DescEntry, &HDTTMap);
              DescEntry->addEventIfNecessary(Device, AsyncInfo);
            }
          }
        }
      }
    };
    auto MemCase = [&](MemMappingInfoTy &MemInfo) {
      if (MemInfo.RawMemorySize) {
        if (MemInfo.CopyDesc) {
          MapWithDesc(MemInfo, MemInfo.RawMemoryBasePtr, true);
        } else {
          TargetPointerResultTy TPR = Device.getMappingInfo().getTargetPointer(
              HDTTMap, MemInfo.RawMemoryPtr, MemInfo.RawMemoryBasePtr, 0,
              (int64_t)*MemInfo.RawMemorySize, ArgName, HasFlagTo,
              /*HasFlagAlways=*/false, /*IsImplicit=*/false,
              /*UpdateRefCount=*/true, /*HasCloseModifier=*/false,
              /*HasPresentModifier=*/false,
              /*HasHoldModifier=*/MapType == AccRefCountingType::Structured,
              IsNoCreate, AsyncInfo, /*OwnedTPR=*/nullptr,
              /*ReleaseHDTTMap=*/false);
          AddArg(TPR, TPR.TargetPointer, MemInfo.RawMemoryPtr);
        }
      } else {
        TargetPointerResultTy TPR = Device.getMappingInfo().getTargetPointer(
            HDTTMap, MemInfo.RawMemoryPtr, MemInfo.RawMemoryBasePtr, 0,
            (int64_t)*MemInfo.RawMemorySize, ArgName, HasFlagTo,
            /*HasFlagAlways=*/false, /*IsImplicit=*/false,
            /*UpdateRefCount=*/true, /*HasCloseModifier=*/false,
            /*HasPresentModifier=*/true,
            /*HasHoldModifier=*/MapType == AccRefCountingType::Structured,
            IsNoCreate, AsyncInfo, /*OwnedTPR=*/nullptr,
            /*ReleaseHDTTMap=*/false);
        AddArg(TPR, TPR.TargetPointer, MemInfo.RawMemoryPtr);
      }
    };
    auto LiteralCase = [&](void *Literal) {
      if (KI)
        KI->addArg(Literal);
    };
    for (auto &MapInfo : MapInfos) {
      std::visit(overloads{DescAndMemCase, MemCase, LiteralCase}, MapInfo);
    }
  }

  void dataEnd(ident_t *Loc, void *ArgPtr, void *DescriptorAddr,
               void *ParentAllocation, int64_t ArgType, bool ForceDelete,
               bool IsNoCreate, AccCopyOutType CopyType,
               AccRefCountingType MapType, AsyncInfoTy &AsyncInfo,
               DeviceTy &Device) {
    auto MapWithDesc = [&](MemMappingInfoTy &MemInfo, void *BasePtr) {
      if (MemInfo.CopyDesc) {
        ODBG(ADT_Mapping) << "Will use non-contig copy.";

        handleSingleDataEnd<NonContigDescTy &>(
            Loc, BasePtr, MemInfo.RawMemoryPtr, *MemInfo.CopyDesc, ForceDelete,
            IsNoCreate, CopyType, MapType, AsyncInfo, Device);
      }
    };

    auto MapInfos = getMappingInfos(Loc, ArgPtr);
    auto DescAndMemCase = [&](DescAndMemMappingInfoTy &MapInfo) {
      auto &DescInfo = MapInfo.Desc;

      if (!ParentAllocation) {
        ParentAllocation = DescriptorAddr;
      }

      handleSingleDataEnd<int64_t>(
          Loc, DescriptorAddr, DescriptorAddr, DescInfo.DescriptorSize,
          ForceDelete, IsNoCreate, CopyType, MapType, AsyncInfo, Device);

      if (MapInfo.Memory) {
        auto &MemInfo = *MapInfo.Memory;
        void *BasePtr =
            static_cast<char *>(DescriptorAddr) + DescInfo.RawMemoryPtrOffset;
        if (MemInfo.RawMemorySize) {
          if (MemInfo.CopyDesc) {
            MapWithDesc(MemInfo, BasePtr);
          } else {
            handleSingleDataEnd<int64_t>(
                Loc, BasePtr, MemInfo.RawMemoryPtr, *MemInfo.RawMemorySize,
                ForceDelete, IsNoCreate, CopyType, MapType, AsyncInfo, Device);
          }
        } else {
          handleSingleDataEnd<int64_t>(Loc, BasePtr, MemInfo.RawMemoryPtr, 0,
                                       ForceDelete, IsNoCreate, CopyType,
                                       MapType, AsyncInfo, Device);
        }
      }
    };
    auto MemCase = [&](MemMappingInfoTy &MemInfo) {
      if (MemInfo.RawMemorySize) {
        if (MemInfo.CopyDesc) {
          MapWithDesc(MemInfo, MemInfo.RawMemoryBasePtr);
        } else {
          handleSingleDataEnd<int64_t>(
              Loc, MemInfo.RawMemoryBasePtr, MemInfo.RawMemoryPtr,
              *MemInfo.RawMemorySize, ForceDelete, IsNoCreate, CopyType,
              MapType, AsyncInfo, Device);
        }
      } else {
        handleSingleDataEnd<int64_t>(
            Loc, MemInfo.RawMemoryBasePtr, MemInfo.RawMemoryPtr, 0, ForceDelete,
            IsNoCreate, CopyType, MapType, AsyncInfo, Device);
      }
    };
    auto LiteralCase = [&](void *Literal) {};
    for (auto &MapInfo : MapInfos) {
      std::visit(overloads{DescAndMemCase, MemCase, LiteralCase}, MapInfo);
    }
  }

  void dataUpdate(ident_t *Loc, void *ArgPtr, int64_t ArgType,
                  AsyncInfoTy &AsyncInfo, DeviceTy &Device) {
    const bool HasFlagTo = ArgType & TGT_ACC_MAPTYPE_TO;
    const bool HasFlagFrom = ArgType & TGT_ACC_MAPTYPE_FROM;

    auto LookupMapping = [&](void *HstPtr,
                             int64_t Size) -> TargetPointerResultTy {
      TargetPointerResultTy TPR = Device.getMappingInfo().getTgtPtrBegin(
          HstPtr, Size, /*UpdateRefCount=*/false, /*UseHoldRefCount=*/false,
          /*MustContain=*/true);
      if (!TPR.isPresent()) {
        if (ArgType & TGT_ACC_MAPTYPE_IF_PRESENT) {
          ODBG(ADT_Interface) << "Not present, if_present - skipping update.";
          return TPR;
        }
        REPORT_FATAL() << "Device mapping does not exist for update at " << Loc;
      }
      return TPR;
    };

    auto DoContiguousUpdate = [&](void *HstPtr, int64_t Size) {
      TargetPointerResultTy TPR = LookupMapping(HstPtr, Size);
      if (!TPR.isPresent())
        return;
      void *TgtPtr = TPR.TargetPointer;
      if (HasFlagTo) {
        ODBG(ADT_Interface) << "Update TO: " << Size << " bytes hst:" << HstPtr
                            << " -> tgt:" << TgtPtr;
        Device.submitData(TgtPtr, HstPtr, Size, AsyncInfo, TPR.getEntry());
      }
      if (HasFlagFrom) {
        ODBG(ADT_Interface) << "Update FROM: " << Size
                            << " bytes tgt:" << TgtPtr << " -> hst:" << HstPtr;
        Device.retrieveData(HstPtr, TgtPtr, Size, AsyncInfo, TPR.getEntry());
      }
    };

    auto DoNonContigUpdate = [&](MemMappingInfoTy &MemInfo) {
      ODBG(ADT_Interface) << "Will use non-contig update.";

      int64_t AllocSize = MemInfo.CopyDesc->getAllocSize();

      TargetPointerResultTy TPR =
          LookupMapping(MemInfo.RawMemoryPtr, AllocSize);
      if (!TPR.isPresent())
        return;
      void *TgtPtr = TPR.TargetPointer;
      if (HasFlagTo) {
        ODBG(ADT_Interface)
            << "Non-contig update TO: hst:" << MemInfo.RawMemoryPtr
            << " -> tgt:" << TgtPtr;
        Device.submitNonContigData(TgtPtr, MemInfo.RawMemoryPtr,
                                   *MemInfo.CopyDesc, AsyncInfo,
                                   TPR.getEntry());
      }
      if (HasFlagFrom) {
        ODBG(ADT_Interface) << "Non-contig update FROM: tgt:" << TgtPtr
                            << " -> hst:" << MemInfo.RawMemoryPtr;
        Device.retrieveNonContigData(MemInfo.RawMemoryPtr, TgtPtr,
                                     *MemInfo.CopyDesc, AsyncInfo,
                                     TPR.getEntry());
      }
    };

    auto MapInfos = getMappingInfos(Loc, ArgPtr);
    for (auto &MapInfo : MapInfos) {
      if (auto *DM = std::get_if<DescAndMemMappingInfoTy>(&MapInfo)) {
        if (DM->Memory && DM->Memory->RawMemorySize) {
          if (DM->Memory->CopyDesc) {
            DoNonContigUpdate(*DM->Memory);
          } else {
            DoContiguousUpdate(DM->Memory->RawMemoryPtr,
                               *DM->Memory->RawMemorySize);
          }
        }
      } else if (auto *MM = std::get_if<MemMappingInfoTy>(&MapInfo)) {
        if (MM->RawMemorySize) {
          if (MM->CopyDesc) {
            DoNonContigUpdate(*MM);
          } else {
            DoContiguousUpdate(MM->RawMemoryPtr, *MM->RawMemorySize);
          }
        }
      }
    }
  }
};

const uint64_t *getMemRefSizes(const MemRefDesc *Desc) {
  return &Desc->sizes[0];
}
const uint64_t *getMemRefStrides(const MemRefDesc *Desc, unsigned Rank) {
  return &Desc->sizes[0] + Rank;
}

ArgDescriptorsTy parseArgDescs(ident_t *Loc, const AccDataDesc *ArgDesc) {
  ArgDescriptorsTy Descs;
  if (!ArgDesc)
    return Descs;

  if (ArgDesc->Version & TGT_ACC_DESC_F18) {
    Descs.Flang = reinterpret_cast<decltype(Descs.Flang)>(
        ((const AccDataDescF18 *)ArgDesc)->FortranDescriptor);
  }
  if (ArgDesc->Version & TGT_ACC_DESC_MEMREF) {
    const MemRefDesc *DescMemRef =
        ((const AccDataDescMemRef *)ArgDesc)->MemRefDescriptor;
    Descs.MemRef = MaterializedMemRefDesc{};
    Descs.MemRef->allocatedPtr = DescMemRef->allocatedPtr;
    Descs.MemRef->alignedPtr = DescMemRef->alignedPtr;
    Descs.MemRef->offset = DescMemRef->offset;
    Descs.MemRef->rank = ((const AccDataDescMemRef *)ArgDesc)->Rank;
    Descs.MemRef->sizes = getMemRefSizes(DescMemRef);
    Descs.MemRef->strides = getMemRefStrides(DescMemRef, Descs.MemRef->rank);
    Descs.MemRef->elementSize =
        ((const AccDataDescMemRef *)ArgDesc)->ElementSize;
  }
  if (ArgDesc->Version & TGT_ACC_DESC_OPENACC) {
    int64_t DescPadding = 0;
    if (Descs.Flang) {
      DescPadding = sizeof(CFI_cdesc_t *);
    } else if (Descs.MemRef) {
      REPORT_FATAL() << Loc << "Unsupported: MemRef with OpenACC bounds";
    } else {
      DescPadding = 0;
    }
    Descs.Acc = reinterpret_cast<const AccDataDescOpenACC *>(
        reinterpret_cast<const char *>(ArgDesc) + DescPadding);
  }
  return Descs;
}

ArgDescriptorsTy parseAndVerifyArgDescs(ident_t *Loc,
                                        const AccDataDesc *ArgDesc) {
  ArgDescriptorsTy Descs = parseArgDescs(Loc, ArgDesc);
  ODBG_IF([&]() { Descs.dump(llvm::dbgs()); });
  Descs.verify();
  return Descs;
}

void accTargetDataBegin(ident_t *Loc, void *ArgBasePtr, void *ArgPtr,
                        int64_t ArgSize, int64_t ArgType, char *ArgName,
                        AccDataDesc *ArgDesc, AccRefCountingType MapType,
                        AsyncInfoTy &AsyncInfo, DeviceTy &Device,
                        MappingInfoTy::HDTTMapAccessorTy &HDTTMap,
                        KernelArgsMappingInfoTy *KI = nullptr) {
  // clang-format off
  ODBG(ADT_Interface)
      << "targetDataBegin "
      << "ArgName=" << getNameFromMapping(ArgName) << ", "
      << "ArgBasePtr=" << ArgBasePtr << ", "
      << "ArgPtr=" << ArgPtr << ", "
      << "ArgSize=" << ArgSize << ", "
      << "ArgType=" << mapTypeToString(ArgType)
      << " (" << llvm::format_hex(ArgType, 0) << "), "
      << "ArgDesc=" << ArgDesc;
  // clang-format on

  // OpenACC 3.4: `if_present` is only valid on `host_data` and `update`
  // directives.
  assert(!(ArgType & TGT_ACC_MAPTYPE_IF_PRESENT));
  assert(!!ArgBasePtr == !!(ArgType & TGT_ACC_MAPTYPE_PTR_AND_OBJ));

  bool IsNoCreate = ArgType & TGT_ACC_MAPTYPE_NO_CREATE;
  auto AddArg = [&](TargetPointerResultTy &TPR, void *TgtArg, void *HstArg) {
    if (KI) {
      if (TPR.isPresent()) {
        KI->addArg(TgtArg);
      } else {
        assert(IsNoCreate);
        KI->addArg(HstArg);
      }
    }
  };

  if (ArgType & TGT_ACC_MAPTYPE_DEVPTR) {
    if (!KI) {
      ODBG(ADT_Interface) << "DEVPTR arg in non-kernel context - ignoring.";
      return;
    }
    if (!ArgDesc) {
      void *LiteralValue = *reinterpret_cast<void ***>(ArgPtr);
      ODBG(ADT_Interface) << "Got literal device pointer: " << LiteralValue;
      KI->addArg(LiteralValue);
      return;
    }
    // DEVPTR with a descriptor. The kernel is compiled to receive a device-side
    // descriptor as a pointer.
    ODBG(ADT_Interface) << "DEVPTR with descriptor";
    ArgDescriptorsTy Descs = parseAndVerifyArgDescs(Loc, ArgDesc);
    Descs.dataBeginDevPtr(Loc, Device, AsyncInfo, HDTTMap, *KI);
    return;
  }

  const bool HasFlagTo = ArgType & TGT_ACC_MAPTYPE_TO;
  if (ArgType & TGT_ACC_MAPTYPE_PRIVATE) {
    assert(KI && "Private arg should only appear on kernels");

    int64_t BaseAllocSize = ArgSize;

    if (BaseAllocSize <= 0)
      REPORT_FATAL() << "Invalid private variable size";

    int64_t NumPrivate = 1;
    if ((ArgType & TGT_ACC_MAPTYPE_GANG_PRIVATE))
      NumPrivate *= KI->KernelArgs.NumGangs[0] * KI->KernelArgs.NumGangs[1] *
                    KI->KernelArgs.NumGangs[2];
    if ((ArgType & TGT_ACC_MAPTYPE_WORKER_PRIVATE))
      NumPrivate *= KI->KernelArgs.NumWorkers;
    if ((ArgType & TGT_ACC_MAPTYPE_VECTOR_PRIVATE))
      NumPrivate *= KI->KernelArgs.VectorLength;

    if (ArgDesc) {
      if (NumPrivate != 1)
        REPORT_FATAL() << Loc << " Multi-dim private array variable is invalid";
      ODBG() << "Arg desc on private variable";
      ArgDescriptorsTy Descs = parseAndVerifyArgDescs(Loc, ArgDesc);
      Descs.dataBeginPrivate(Loc, ArgPtr, ArgSize, HasFlagTo, Device, AsyncInfo,
                             HDTTMap, *KI);
    } else {
      void *PrivateMemory = Device.allocData(BaseAllocSize * NumPrivate,
                                             nullptr, TARGET_ALLOC_DEFAULT);
      ODBG(ADT_Interface) << "Allocated private memory with size "
                          << BaseAllocSize << " (" << NumPrivate
                          << " instances) at " << PrivateMemory;

      if (HasFlagTo) {
        if (NumPrivate != 1)
          REPORT_FATAL() << Loc
                         << " Multi-dim private variable with copy is invalid";

        assert(ArgPtr);
        Device.submitData(PrivateMemory, ArgPtr, ArgSize, AsyncInfo,
                          /*Entry=*/nullptr, &HDTTMap);
      }
      KI->addArg(PrivateMemory);
      KI->addLaunchAlloc(PrivateMemory);
    }

    return;
  }

  if (ArgType & TGT_ACC_MAPTYPE_LITERAL) {
    assert(KI && "Literal arg should only appear on kernels");
    assert(ArgSize && "We need size information to pass in literal args");
    assert(!ArgDesc);
    // Our codegen uses indirection for literal args.
    if (ArgSize <= (int)sizeof(void *)) {
      // If it is possible to type pun to pointer (i.e. the type width is no
      // bigger than a pointer, then pass it in literally.
      void *LiteralValue = *reinterpret_cast<void ***>(ArgPtr);
      KI->addArg(LiteralValue);
      return;
    } else {
      REPORT_FATAL() << "TODO need to move memory to device";
      // KI->addArg(DeviceArgPtr);
      return;
    }
  }

  assert(ArgPtr && "We need to have a pointer for data mapping");

  void *ParentAllocation = nullptr;
  void *DescriptorAddr;
  bool IsPtrAndObj = ArgType & TGT_ACC_MAPTYPE_PTR_AND_OBJ;
  if (IsPtrAndObj) {
    ODBG() << "We got a parent object.";
    assert(ArgBasePtr);
    if (Device.getMappingInfo().getTgtPtrBegin(HDTTMap, ArgBasePtr, 1)) {
      ParentAllocation = ArgBasePtr;
      DescriptorAddr = ArgBasePtr;
    } else {
      // PTR_AND_OBJ but parent not present on device (e.g. enter data copyin
      // of a pointer component without its parent struct). Data is already
      // mapped standalone; skip descriptor attach.
      ODBG() << "Parent not present on device - mapping standalone.";
      IsPtrAndObj = false;
      DescriptorAddr = ArgPtr;
    }
  } else {
    DescriptorAddr = ArgPtr;
  }

  if (ArgSize > 0) {
    ODBG() << "We got size from the compiler - no descriptor parsing needed.";

    if (IsPtrAndObj) {
      // Map the pointee data, then release entry lock before looking up
      // the parent for pointer attachment.
      void *MemTgtPtr = nullptr;
      {
        TargetPointerResultTy TPR = Device.getMappingInfo().getTargetPointer(
            HDTTMap, ArgPtr, ArgPtr, 0, ArgSize, ArgName, HasFlagTo,
            /*HasFlagAlways=*/false, /*IsImplicit=*/false,
            /*UpdateRefCount=*/true, /*HasCloseModifier=*/false,
            /*HasPresentModifier=*/false,
            /*HasHoldModifier=*/MapType == AccRefCountingType::Structured,
            IsNoCreate, AsyncInfo, /*OwnedTPR=*/nullptr,
            /*ReleaseHDTTMap=*/false);
        AddArg(TPR, TPR.TargetPointer, ArgPtr);
        if (TPR.isPresent())
          MemTgtPtr = TPR.TargetPointer;
        else
          assert(IsNoCreate);
      }

      // Update the parent's pointer field on device.
      if (MemTgtPtr) {
        LookupResult ParentLR = Device.getMappingInfo().lookupMapping(
            HDTTMap, DescriptorAddr, sizeof(void *));
        if (ParentLR.TPR.getEntry()) {
          void **HstPtrAddr = reinterpret_cast<void **>(DescriptorAddr);
          uintptr_t TgtDescAddr = ParentLR.TPR.getEntry()->TgtPtrBegin +
                                  (reinterpret_cast<uintptr_t>(DescriptorAddr) -
                                   ParentLR.TPR.getEntry()->HstPtrBegin);
          void **TgtPtrAddr = reinterpret_cast<void **>(TgtDescAddr);

          void *HstPteeBase = *HstPtrAddr;
          void *TgtPteeBase = reinterpret_cast<void *>(
              reinterpret_cast<uintptr_t>(MemTgtPtr) -
              (reinterpret_cast<uintptr_t>(ArgPtr) -
               reinterpret_cast<uintptr_t>(HstPteeBase)));

          if (ParentLR.TPR.getEntry()->addShadowPointer(
                  ShadowPtrInfoTy{HstPtrAddr, TgtPtrAddr, TgtPteeBase,
                                  static_cast<int64_t>(sizeof(void *))})) {
            ODBG() << "PTR_AND_OBJ attach: device field " << TgtPtrAddr
                   << " -> " << TgtPteeBase;
            void *&Buf = AsyncInfo.getVoidPtrLocation();
            Buf = TgtPteeBase;
            Device.submitData(TgtPtrAddr, &Buf, sizeof(void *), AsyncInfo,
                              ParentLR.TPR.getEntry(), &HDTTMap);
            ParentLR.TPR.getEntry()->addEventIfNecessary(Device, AsyncInfo);
          }
        }
      }

    } else {
      TargetPointerResultTy TPR = Device.getMappingInfo().getTargetPointer(
          HDTTMap, ArgPtr, ArgPtr, 0, ArgSize, ArgName, HasFlagTo,
          /*HasFlagAlways=*/false, /*IsImplicit=*/false,
          /*UpdateRefCount=*/true, /*HasCloseModifier=*/false,
          /*HasPresentModifier=*/false,
          /*HasHoldModifier=*/MapType == AccRefCountingType::Structured,
          IsNoCreate, AsyncInfo, /*OwnedTPR=*/nullptr,
          /*ReleaseHDTTMap=*/false);
      AddArg(TPR, TPR.TargetPointer, ArgPtr);
    }
    return;
  }

  ArgDescriptorsTy Descs = parseAndVerifyArgDescs(Loc, ArgDesc);
  Descs.dataBegin(Loc, ArgPtr, DescriptorAddr, ParentAllocation, IsPtrAndObj,
                  ArgName, HasFlagTo, IsNoCreate, MapType, AsyncInfo, Device,
                  HDTTMap, KI);
}

bool isPresent(DeviceTy &Device, void *Ptr) {
  TargetPointerResultTy TPR = Device.getMappingInfo().getTgtPtrBegin(
      Ptr, 1, /*UpdateRefCount=*/false, /*UseHoldRefCount=*/false);
  return TPR.isPresent();
}

void accTargetDataEnd(ident_t *Loc, void *ArgBasePtr, void *ArgPtr,
                      int64_t ArgSize, int64_t ArgType, char *ArgName,
                      AccDataDesc *ArgDesc, AccRefCountingType MapType,
                      AsyncInfoTy &AsyncInfo, DeviceTy &Device) {
  // clang-format off
  ODBG(ADT_Interface)
      << "targetDataEnd "
      << "ArgName=" << getNameFromMapping(ArgName) << ", "
      << "ArgBasePtr=" << ArgBasePtr << ", "
      << "ArgPtr=" << ArgPtr << ", "
      << "ArgSize=" << ArgSize << ", "
      << "ArgType=" << mapTypeToString(ArgType)
      << " (" << llvm::format_hex(ArgType, 0) << "), "
      << "ArgDesc=" << ArgDesc;
  // clang-format on

  // OpenACC 3.4: `if_present` is only valid on `host_data` and `update`
  // directives.
  assert(!(ArgType & TGT_ACC_MAPTYPE_IF_PRESENT));

  assert(!!ArgBasePtr == !!(ArgType & TGT_ACC_MAPTYPE_PTR_AND_OBJ));

  // These types are only for kernel launches
  if ((ArgType & TGT_ACC_MAPTYPE_VECTOR_PRIVATE) ||
      (ArgType & TGT_ACC_MAPTYPE_GANG_PRIVATE) ||
      (ArgType & TGT_ACC_MAPTYPE_WORKER_PRIVATE) ||
      (ArgType & TGT_ACC_MAPTYPE_LITERAL) ||
      (ArgType & TGT_ACC_MAPTYPE_DEVPTR) ||
      (ArgType & TGT_ACC_MAPTYPE_PRIVATE)) {
    ODBG(ADT_Interface) << "Kernel launch argument - ignoring.";
    return;
  }

  assert(ArgPtr && "We need to have a pointer for data mapping");

  void *ParentAllocation = nullptr;
  void *DescriptorAddr;
  bool IsPtrAndObj = ArgType & TGT_ACC_MAPTYPE_PTR_AND_OBJ;
  if (IsPtrAndObj) {
    ODBG(ADT_Mapping) << "We got a parent object.";
    assert(ArgBasePtr);
    if (isPresent(Device, ArgBasePtr)) {
      ParentAllocation = ArgBasePtr;
      DescriptorAddr = ArgBasePtr;
    } else {
      // PTR_AND_OBJ but parent not present on device (e.g. enter data copyin
      // of a pointer component without its parent struct). Data is already
      // mapped standalone; skip descriptor attach.
      ODBG(ADT_Mapping) << "Parent not present on device - mapping standalone.";
      IsPtrAndObj = false;
      DescriptorAddr = ArgPtr;
    }
  } else {
    DescriptorAddr = ArgPtr;
  }

  const bool ForceDelete = ArgType & TGT_ACC_MAPTYPE_FINALIZE;
  const bool HasFlagFrom = ArgType & TGT_ACC_MAPTYPE_FROM;
  const bool IsNoCreate = ArgType & TGT_ACC_MAPTYPE_NO_CREATE;
  AccCopyOutType CopyType = AccCopyOutType::Never;
  if (HasFlagFrom)
    CopyType = AccCopyOutType::OnDelete;
  if (ArgSize > 0) {
    ODBG(ADT_Mapping)
        << "We got size from the compiler - no descriptor parsing needed.";

    handleSingleDataEnd<int64_t>(Loc, DescriptorAddr, ArgPtr, ArgSize,
                                 ForceDelete, IsNoCreate, CopyType, MapType,
                                 AsyncInfo, Device);
    return;
  }

  ArgDescriptorsTy Descs = parseAndVerifyArgDescs(Loc, ArgDesc);
  Descs.dataEnd(Loc, ArgPtr, DescriptorAddr, ParentAllocation, ArgType,
                ForceDelete, IsNoCreate, CopyType, MapType, AsyncInfo, Device);
}

void accTargetDataUpdate(ident_t *Loc, void *ArgBasePtr, void *ArgPtr,
                         int64_t ArgSize, int64_t ArgType, char *ArgName,
                         AccDataDesc *ArgDesc, AsyncInfoTy &AsyncInfo,
                         DeviceTy &Device) {
  // clang-format off
  ODBG(ADT_Interface)
      << "update "
      << "ArgName=" << getNameFromMapping(ArgName) << ", "
      << "ArgPtr=" << ArgPtr << ", "
      << "ArgSize=" << ArgSize << ", "
      << "ArgType=" << mapTypeToString(ArgType)
      << " (" << llvm::format_hex(ArgType, 0) << "), "
      << "ArgDesc=" << ArgDesc;
  // clang-format on

  if ((ArgType & TGT_ACC_MAPTYPE_LITERAL) ||
      (ArgType & TGT_ACC_MAPTYPE_PRIVATE) || (ArgType & TGT_ACC_MAPTYPE_DEVPTR))
    return;

  if (!ArgPtr)
    return;

  if (ArgSize > 0) {
    TargetPointerResultTy TPR = Device.getMappingInfo().getTgtPtrBegin(
        ArgPtr, ArgSize, /*UpdateRefCount=*/false, /*UseHoldRefCount=*/false,
        /*MustContain=*/true);
    if (!TPR.isPresent()) {
      if (ArgType & TGT_ACC_MAPTYPE_IF_PRESENT) {
        ODBG(ADT_Interface) << "Not present, if_present - skipping update.";
        return;
      }
      REPORT_FATAL() << "Device mapping does not exist for update at " << Loc;
    }
    void *TgtPtr = TPR.TargetPointer;
    if (ArgType & TGT_ACC_MAPTYPE_TO) {
      ODBG(ADT_Interface) << "Update TO: " << ArgSize << " bytes hst:" << ArgPtr
                          << " -> tgt:" << TgtPtr;
      Device.submitData(TgtPtr, ArgPtr, ArgSize, AsyncInfo, TPR.getEntry());
    }
    if (ArgType & TGT_ACC_MAPTYPE_FROM) {
      ODBG(ADT_Interface) << "Update FROM: " << ArgSize
                          << " bytes tgt:" << TgtPtr << " -> hst:" << ArgPtr;
      Device.retrieveData(ArgPtr, TgtPtr, ArgSize, AsyncInfo, TPR.getEntry());
    }
    return;
  }

  ArgDescriptorsTy Descs = parseAndVerifyArgDescs(Loc, ArgDesc);
  Descs.dataUpdate(Loc, ArgPtr, ArgType, AsyncInfo, Device);
}

template <typename T>
void withDeviceAndQueue(int64_t DeviceType, int64_t Async, T Callback) {
  llvm::Expected<DeviceTy &> DeviceOrErr =
      DM->getDevice(static_cast<acc_device_t>(DeviceType));
  if (!DeviceOrErr)
    REPORT_FATAL() << "Failed to get device: "
                   << toString(DeviceOrErr.takeError());

  DeviceTy &Device = *DeviceOrErr;

  ODBG(ADT_Interface) << "with device type " << DeviceType << " and async "
                      << asyncToString(Async);

  if (Async == AccAsyncSync) {
    AsyncInfoTy AsyncInfo(Device);
    Callback(Device, AsyncInfo);
  } else {
    QueueAsyncInfoWrapperTy QueueAsyncInfo(Device, Async);
    AsyncInfoTy &AsyncInfo = QueueAsyncInfo;
    Callback(Device, AsyncInfo);
  }
}

template <typename FuncTy, typename... ArgsTy>
void forEachArg(FuncTy Func, bool Increasing, ident_t *Loc, uint32_t ArgNum,
                void **ArgBasePtrs, void **ArgPtrs, int64_t *ArgSizes,
                int64_t *ArgTypes, char **ArgNames, void **ArgMappers,
                AccDataDesc **ArgDescs, ArgsTy &&...Args) {
  assert(!ArgMappers && "we currently do not generate mappers");
  ODBG(ADT_Interface) << "Got " << ArgNum << " args at " << Loc;
  int32_t Start = Increasing ? 0 : ArgNum - 1;
  int32_t End = Increasing ? ArgNum : -1;
  int32_t Increment = Increasing ? 1 : -1;
  for (int32_t I = Start; I != End; I += Increment) {
    ODBG(ADT_Interface) << "Handling arg #" << I;
    char *Name = ArgNames ? ArgNames[I] : nullptr;
    Func(Loc, ArgBasePtrs[I], ArgPtrs[I], ArgSizes[I], ArgTypes[I], Name,
         ArgDescs[I], Args...);
  }
}
} // namespace

namespace llvm::acc::target {
void *accDataEnter(void *ArgBasePtr, void *ArgPtr, int64_t ArgSize,
                   int64_t ArgType, int64_t Async) {
  void *Result = nullptr;
  withDeviceAndQueue(
      acc_device_default, Async, [&](DeviceTy &Device, AsyncInfoTy &AsyncInfo) {
        {
          AccKernelArgsTy KA = {};
          KernelArgsMappingInfoTy KI{KA, {}, {}, {}};
          MappingInfoTy::HDTTMapAccessorTy HDTTMap =
              Device.getMappingInfo()
                  .HostDataToTargetMap.getExclusiveAccessor();
          accTargetDataBegin(nullptr, ArgBasePtr, ArgPtr, ArgSize, ArgType,
                             nullptr, nullptr, AccRefCountingType::Dynamic,
                             AsyncInfo, Device, HDTTMap, &KI);
          assert(KI.Args.size() == 1);
          Result = KI.Args[0];
        }
        dumpTargetPointerMappings(nullptr, Device);
      });
  return Result;
}
} // namespace llvm::acc::target

EXTERN void __tgt_acc_declare(ident_t *Loc, int64_t Flags, int64_t DeviceType,
                              uint32_t ArgNum, void **ArgBasePtrs,
                              void **ArgPtrs, int64_t *ArgSizes,
                              int64_t *ArgTypes, char **ArgNames,
                              void **ArgMappers, AccDataDesc **ArgDescs,
                              int64_t Async, __tgt_bin_desc *Desc) {
  FUNC_LOGGER(Loc);
  assert(!Desc);

  withDeviceAndQueue(
      DeviceType, Async, [&](DeviceTy &Device, AsyncInfoTy &AsyncInfo) {
        {
          MappingInfoTy::HDTTMapAccessorTy HDTTMap =
              Device.getMappingInfo()
                  .HostDataToTargetMap.getExclusiveAccessor();
          forEachArg(accTargetDataBegin, /*Increasing=*/true, Loc, ArgNum,
                     ArgBasePtrs, ArgPtrs, ArgSizes, ArgTypes, ArgNames,
                     ArgMappers, ArgDescs, AccRefCountingType::Structured,
                     AsyncInfo, Device, HDTTMap, /*KI=*/nullptr);
        }
        dumpTargetPointerMappings(Loc, Device);
      });
}

EXTERN void __tgt_acc_data_update(ident_t *Loc, int64_t Flags,
                                  int64_t DeviceType, uint32_t ArgNum,
                                  void **ArgBasePtrs, void **ArgPtrs,
                                  int64_t *ArgSizes, int64_t *ArgTypes,
                                  char **ArgNames, void **ArgMappers,
                                  AccDataDesc **ArgDescs, int64_t Async) {
  FUNC_LOGGER(Loc);
  withDeviceAndQueue(
      DeviceType, Async, [&](DeviceTy &Device, AsyncInfoTy &AsyncInfo) {
        forEachArg(accTargetDataUpdate, /*Increasing=*/true, Loc, ArgNum,
                   ArgBasePtrs, ArgPtrs, ArgSizes, ArgTypes, ArgNames,
                   ArgMappers, ArgDescs, AsyncInfo, Device);
      });
}

EXTERN void __tgt_acc_data_enter(ident_t *Loc, int64_t Flags,
                                 int64_t DeviceType, uint32_t ArgNum,
                                 void **ArgBasePtrs, void **ArgPtrs,
                                 int64_t *ArgSizes, int64_t *ArgTypes,
                                 char **ArgNames, void **ArgMappers,
                                 AccDataDesc **ArgDescs, int64_t Async) {
  FUNC_LOGGER(Loc);
  withDeviceAndQueue(
      DeviceType, Async, [&](DeviceTy &Device, AsyncInfoTy &AsyncInfo) {
        {
          MappingInfoTy::HDTTMapAccessorTy HDTTMap =
              Device.getMappingInfo()
                  .HostDataToTargetMap.getExclusiveAccessor();
          forEachArg(accTargetDataBegin, /*Increasing=*/true, Loc, ArgNum,
                     ArgBasePtrs, ArgPtrs, ArgSizes, ArgTypes, ArgNames,
                     ArgMappers, ArgDescs, AccRefCountingType::Dynamic,
                     AsyncInfo, Device, HDTTMap, /*KI=*/nullptr);
        }
        dumpTargetPointerMappings(Loc, Device);
      });
}

EXTERN void __tgt_acc_data_exit(ident_t *Loc, int64_t Flags, int64_t DeviceType,
                                uint32_t ArgNum, void **ArgBasePtrs,
                                void **ArgPtrs, int64_t *ArgSizes,
                                int64_t *ArgTypes, char **ArgNames,
                                void **ArgMappers, AccDataDesc **ArgDescs,
                                int64_t Async) {
  FUNC_LOGGER(Loc);
  withDeviceAndQueue(
      DeviceType, Async, [&](DeviceTy &Device, AsyncInfoTy &AsyncInfo) {
        forEachArg(accTargetDataEnd, /*Increasing=*/false, Loc, ArgNum,
                   ArgBasePtrs, ArgPtrs, ArgSizes, ArgTypes, ArgNames,
                   ArgMappers, ArgDescs, AccRefCountingType::Dynamic, AsyncInfo,
                   Device);
        dumpTargetPointerMappings(Loc, Device);
      });
}

EXTERN void __tgt_acc_data_begin(ident_t *Loc, int64_t Flags,
                                 int64_t DeviceType, uint32_t ArgNum,
                                 void **ArgBasePtrs, void **ArgPtrs,
                                 int64_t *ArgSizes, int64_t *ArgTypes,
                                 char **ArgNames, void **ArgMappers,
                                 AccDataDesc **ArgDescs, int64_t Async) {
  FUNC_LOGGER(Loc);
  withDeviceAndQueue(
      DeviceType, Async, [&](DeviceTy &Device, AsyncInfoTy &AsyncInfo) {
        {
          MappingInfoTy::HDTTMapAccessorTy HDTTMap =
              Device.getMappingInfo()
                  .HostDataToTargetMap.getExclusiveAccessor();
          forEachArg(accTargetDataBegin, /*Increasing=*/true, Loc, ArgNum,
                     ArgBasePtrs, ArgPtrs, ArgSizes, ArgTypes, ArgNames,
                     ArgMappers, ArgDescs, AccRefCountingType::Structured,
                     AsyncInfo, Device, HDTTMap, /*KI=*/nullptr);
        }
        dumpTargetPointerMappings(Loc, Device);
      });
}

EXTERN void __tgt_acc_data_end(ident_t *Loc, int64_t Flags, int64_t DeviceType,
                               uint32_t ArgNum, void **ArgBasePtrs,
                               void **ArgPtrs, int64_t *ArgSizes,
                               int64_t *ArgTypes, char **ArgNames,
                               void **ArgMappers, AccDataDesc **ArgDescs,
                               int64_t Async) {
  FUNC_LOGGER(Loc);
  withDeviceAndQueue(
      DeviceType, Async, [&](DeviceTy &Device, AsyncInfoTy &AsyncInfo) {
        forEachArg(accTargetDataEnd, /*Increasing=*/false, Loc, ArgNum,
                   ArgBasePtrs, ArgPtrs, ArgSizes, ArgTypes, ArgNames,
                   ArgMappers, ArgDescs, AccRefCountingType::Structured,
                   AsyncInfo, Device);
        dumpTargetPointerMappings(Loc, Device);
      });
}

void *getDeviceEntryPtr(void *HostPtr, DeviceTy &Device) {
  int32_t DeviceId = Device.DeviceID;
  TableMap *TM = llvm::offload::getTableMap(HostPtr);
  __tgt_target_table *TargetTable = nullptr;
  {
    std::lock_guard<std::mutex> TrlTblLock(PM->TrlTblMtx);
    assert(TM->Table->TargetsTable.size() > (size_t)DeviceId &&
           "Not expecting a device ID outside the table's bounds!");
    TargetTable = TM->Table->TargetsTable[DeviceId];
  }
  assert(TargetTable && "Global data has not been mapped\n");

  void *TgtEntryPtr = TargetTable->EntriesBegin[TM->Index].Address;
  ODBG(ADT_Kernel) << "Launching target execution "
                   << TargetTable->EntriesBegin[TM->Index].SymbolName
                   << " with pointer " << TgtEntryPtr << " (index=" << TM->Index
                   << ").";
  return TgtEntryPtr;
}

EXTERN int __tgt_acc_kernel(ident_t *Loc, void *Kernel, int64_t Flags,
                            int64_t DeviceType, AccKernelArgsTy *Args,
                            int64_t Async, const char *KernelName,
                            __tgt_bin_desc *Desc) {
  FUNC_LOGGER(Loc);
  assert(!Desc);

  withDeviceAndQueue(
      DeviceType, Async, [&](DeviceTy &Device, AsyncInfoTy &AsyncInfo) {
        MappingInfoTy::HDTTMapAccessorTy HDTTMap =
            Device.getMappingInfo().HostDataToTargetMap.getExclusiveAccessor();

        SmallVector<void *> TgtArgs;
        SmallVector<ptrdiff_t> TgtOffsets;
        KernelArgsMappingInfoTy KI{*Args, {}, {}, {}};
        forEachArg(accTargetDataBegin, /*Increasing=*/true, Loc, Args->ArgNum,
                   Args->ArgBasePtrs, Args->ArgPtrs, Args->ArgSizes,
                   Args->ArgTypes, Args->ArgNames, Args->ArgMappers,
                   Args->ArgDescs, AccRefCountingType::Structured, AsyncInfo,
                   Device, HDTTMap, &KI);
        HDTTMap.destroy();

        KernelLaunchParamsTy LaunchParams = KI.getLaunchArgs();
        KernelArgsTy DeviceArgs = {0};
        DeviceArgs.Version = 4;
        DeviceArgs.ArgPtrs = reinterpret_cast<void **>(&LaunchParams);
        DeviceArgs.Flags.IsCUDA = true;
        DeviceArgs.DynCGroupMem = Args->SmemSize;

        DeviceArgs.UserNumBlocks[0] = Args->NumGangs[0];
        DeviceArgs.UserNumBlocks[1] = Args->NumGangs[1];
        DeviceArgs.UserNumBlocks[2] = Args->NumGangs[2];
        DeviceArgs.UserThreadLimit[0] = Args->VectorLength;
        DeviceArgs.UserThreadLimit[1] = Args->NumWorkers;
        DeviceArgs.UserThreadLimit[2] = 1;

        void *TgtEntryPtr = getDeviceEntryPtr(Kernel, Device);
        ODBG(ADT_Interface)
            << "Launching device kernel " << KernelName
            << " with entry hst: " << Kernel << " tgt: " << TgtEntryPtr
            << " with " << KI.Args.size() << " (" << Args->ArgNum << ") args";
        ODBG(ADT_Interface) << "NumGangs " << Args->NumGangs[0] << ", "
                            << Args->NumGangs[1] << ", " << Args->NumGangs[2];
        ODBG(ADT_Interface) << "VectorLength " << Args->VectorLength;
        ODBG(ADT_Interface) << "NumWorkers " << Args->NumWorkers;
        ODBG(ADT_Interface) << "SmemSize " << Args->SmemSize;

        assert(KI.Args.size() * sizeof(void *) == LaunchParams.Size);
        for (unsigned I = 0; I < KI.Args.size(); I++)
          ODBG(ADT_Interface) << "Arg #" << I << ": " << KI.Args[I];

        if (Device.launchKernel(TgtEntryPtr, TgtArgs.data(), TgtOffsets.data(),
                                DeviceArgs, nullptr,
                                AsyncInfo) != OFFLOAD_SUCCESS)
          REPORT_FATAL() << "Kernel launch failed";

        forEachArg(accTargetDataEnd, /*Increasing=*/false, Loc, Args->ArgNum,
                   Args->ArgBasePtrs, Args->ArgPtrs, Args->ArgSizes,
                   Args->ArgTypes, Args->ArgNames, Args->ArgMappers,
                   Args->ArgDescs, AccRefCountingType::Structured, AsyncInfo,
                   Device);

        dumpTargetPointerMappings(Loc, Device);

        auto LaunchAllocDeleter = [Device = &Device,
                                   LaunchAllocs = KI.LaunchAllocs]() {
          for (void *LaunchAlloc : LaunchAllocs)
            if (int32_t Ret = Device->deleteData(LaunchAlloc);
                Ret != OFFLOAD_SUCCESS)
              return Ret;
          return OFFLOAD_SUCCESS;
        };
        AsyncInfo.addPostProcessingFunction(LaunchAllocDeleter);
      });
  return OFFLOAD_SUCCESS;
}

EXTERN void *__tgt_acc_get_deviceptr(ident_t *Loc, void *BasePtr, int64_t Flags,
                                     void *HostPtr) {
  FUNC_LOGGER(Loc);
  ODBG(ADT_Interface) << Loc << "BasePtr: " << BasePtr << ", "
                      << "Flags: " << llvm::format_hex(Flags, 0) << ", "
                      << "HostPtr: " << HostPtr;

  void *DevicePtr = nullptr;

  llvm::Expected<DeviceTy &> DeviceOrErr = DM->getDevice();
  if (!DeviceOrErr)
    REPORT_FATAL() << "Failed to get device: "
                   << toString(DeviceOrErr.takeError());
  DeviceTy &Device = *DeviceOrErr;

  MappingInfoTy::HDTTMapAccessorTy HDTTMap =
      Device.getMappingInfo().HostDataToTargetMap.getExclusiveAccessor();
  DevicePtr = Device.getMappingInfo().getTgtPtrBegin(HDTTMap, HostPtr, 0);

  ODBG(ADT_Interface) << "DevicePtr: " << DevicePtr;

  return DevicePtr;
}

EXTERN void __tgt_acc_set_default_async(ident_t *Loc, int64_t Async) {
  FUNC_LOGGER(Loc);
  ODBG(ADT_Interface) << Loc << ": Set async=" << asyncToString(Async);

  if (Async == AccAsyncSync) {
    REPORT_FATAL() << Loc
                   << "The default queue cannot be set to `acc_async_sync'";
  } else if (Async == AccAsyncNoval) {
    REPORT_FATAL() << Loc
                   << "The default queue cannot be set to `acc_async_noval'";
  } else if (Async == AccAsyncDefault) {
    Async = AccAsyncDefaultQueue;
  } else if (Async < 0) {
    REPORT_FATAL() << Loc << "Negative queues are invalid";
  }

  icv::AccDefaultAsyncVar = Async;
}

EXTERN void __tgt_acc_set_device_num(ident_t *Loc, int64_t Flags,
                                     int64_t DeviceType, int64_t DeviceNum) {
  FUNC_LOGGER(Loc);
  // OpenACC 3.3: If the value of device_num argument is negative, the runtime
  // will revert to the default behavior, which is implementation-defined. A set
  // device_num directive is functionally equivalent
  if (DeviceNum < 0) {
    DeviceNum = 0;
  }

  // OpenACC 3.3: If the value of the device_type argument is zero or the clause
  // does not appear, the selected device number will be used for all attached
  // accelerator types.
  if (DeviceType == 0) {
    DM->setAllDeviceId(DeviceNum);
    return;
  }

  DM->setDeviceType(static_cast<acc_device_t>(DeviceType));
  DM->setDeviceId(DeviceNum);
}

EXTERN void __tgt_acc_set_device_type(ident_t *Loc, int64_t Flags,
                                      int64_t DeviceType) {
  FUNC_LOGGER(Loc);
  DM->setDeviceType(static_cast<acc_device_t>(DeviceType));
}

EXTERN int __tgt_acc_wait(ident_t *Loc, int64_t Flags, int64_t DeviceType,
                          int32_t DeviceNum, uint32_t WaitNum,
                          int64_t *WaitList, int64_t Async) {
  FUNC_LOGGER(Loc);
  ODBG(ADT_Interface) << Loc << "\n"
                      << "DeviceNum: " << DeviceNum << ", "
                      << "DeviceType: " << DeviceType << ", "
                      << "WaitNum: " << WaitNum;
  for (size_t I = 0; I < WaitNum; I++) {
    ODBG(ADT_Interface) << "WaitList[" << I
                        << "]: " << asyncToString(WaitList[I]);
  }
  ODBG(ADT_Interface) << "Async: " << asyncToString(Async)
                      << " Flags: " << llvm::format_hex(Flags, 0);

  accAsyncWait(Loc, DM->getPMDeviceId(), WaitNum, WaitList);

  return 0;
}

namespace {
static std::mutex InitMutex;
uint32_t InitRefCount = 0;

static void initAccRuntime() {
  FUNC_LOGGER();
  initRuntime(/*OffloadEnabled=*/true);
  // TODO Blindly register all rtls for now. In reality we should only be
  // initializing the requested types in case we come from __tgt_acc_init(), or
  // only the ones we have device code for.
  __tgt_init_all_rtls();

  InitRefCount++;
  if (InitRefCount == 1) {
    llvm::acc::target::DM = new llvm::acc::target::DeviceManagerTy();
    llvm::acc::target::DM->init();

    llvm::acc::target::QueueManager = new llvm::acc::target::QueueManagerTy();
    llvm::acc::target::QueueManager->init();
  }
  llvm::acc::target::DM->refreshDeviceMapping(/*UpdateDeviceType=*/true);
}

static void deinitAccRuntime() {
  FUNC_LOGGER();
  if (InitRefCount == 1) {
    llvm::acc::target::QueueManager->deinit();
    delete llvm::acc::target::QueueManager;
    llvm::acc::target::QueueManager = nullptr;

    llvm::acc::target::DM->deinit();
    delete llvm::acc::target::DM;
    llvm::acc::target::DM = nullptr;
  }
  InitRefCount--;

  deinitRuntime();
}
} // namespace

EXTERN void __tgt_acc_init(ident_t *Loc, int64_t Flags, int64_t DeviceType,
                           int64_t DeviceNum) {
  std::scoped_lock<decltype(InitMutex)> Lock(InitMutex);
  FUNC_LOGGER(Loc);
  REPORT_WARN() << "acc init ignores user's request and initializes all "
                   "available devices.";
  initAccRuntime();
  std::atexit([]() {
    std::scoped_lock<decltype(InitMutex)> Lock(InitMutex);
    FUNC_LOGGER();
    deinitAccRuntime();
  });
}

EXTERN void __tgt_acc_shutdown(ident_t *Loc, int64_t Flags, int64_t DeviceType,
                               int64_t DeviceNum) {
  std::scoped_lock<decltype(InitMutex)> Lock(InitMutex);
  FUNC_LOGGER(Loc);
  REPORT_WARN() << "acc shutdown is ignored.";
}

EXTERN void __tgt_acc_register_lib(__tgt_bin_desc *Desc) {
  std::scoped_lock<decltype(InitMutex)> Lock(InitMutex);
  FUNC_LOGGER();
  initAccRuntime();
  if (PM->delayRegisterLib(__tgt_acc_register_lib, Desc))
    return;

  PM->registerLib(Desc);
  llvm::acc::target::DM->refreshDeviceMapping(/*UpdateDeviceType=*/true);
}

EXTERN void __tgt_acc_unregister_lib(__tgt_bin_desc *Desc) {
  std::scoped_lock<decltype(InitMutex)> Lock(InitMutex);
  FUNC_LOGGER();
  PM->unregisterLib(Desc);

  deinitAccRuntime();
}
