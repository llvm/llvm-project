//===-- SPIRVDuplicatesTracker.h - SPIR-V Duplicates Tracker ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// General infrastructure for keeping track of the values that according to
// the SPIR-V binary layout should be global to the whole module.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVIRMAPPING_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVIRMAPPING_H

#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "MCTargetDesc/SPIRVMCTargetDesc.h"
#include "SPIRVUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

#include <type_traits>

namespace llvm {
namespace SPIRV {

using IRHandle = std::tuple<const void *, unsigned, unsigned>;

enum SpecialTypeKind {
  STK_Empty = 0,
  STK_Image,
  STK_SampledImage,
  STK_Sampler,
  STK_Pipe,
  STK_DeviceEvent,
  STK_ElementPointer,
  STK_Pointer,
  STK_Last = -1
};

union ImageAttrs {
  struct BitFlags {
    unsigned Dim : 3;
    unsigned Depth : 2;
    unsigned Arrayed : 1;
    unsigned MS : 1;
    unsigned Sampled : 2;
    unsigned ImageFormat : 6;
    unsigned AQ : 2;
  } Flags;
  unsigned Val;

  ImageAttrs(unsigned Dim, unsigned Depth, unsigned Arrayed, unsigned MS,
             unsigned Sampled, unsigned ImageFormat, unsigned AQ = 0) {
    Val = 0;
    Flags.Dim = Dim;
    Flags.Depth = Depth;
    Flags.Arrayed = Arrayed;
    Flags.MS = MS;
    Flags.Sampled = Sampled;
    Flags.ImageFormat = ImageFormat;
    Flags.AQ = AQ;
  }
};

inline IRHandle make_descr_image(const Type *SampledTy, unsigned Dim,
                                 unsigned Depth, unsigned Arrayed, unsigned MS,
                                 unsigned Sampled, unsigned ImageFormat,
                                 unsigned AQ = 0) {
  return std::make_tuple(
      SampledTy,
      ImageAttrs(Dim, Depth, Arrayed, MS, Sampled, ImageFormat, AQ).Val,
      SpecialTypeKind::STK_Image);
}

inline IRHandle make_descr_sampled_image(const Type *SampledTy,
                                         const MachineInstr *ImageTy) {
  assert(ImageTy->getOpcode() == SPIRV::OpTypeImage);
  unsigned AC = AccessQualifier::AccessQualifier::None;
  if (ImageTy->getNumOperands() > 8)
    AC = ImageTy->getOperand(8).getImm();
  return std::make_tuple(
      SampledTy,
      ImageAttrs(
          ImageTy->getOperand(2).getImm(), ImageTy->getOperand(3).getImm(),
          ImageTy->getOperand(4).getImm(), ImageTy->getOperand(5).getImm(),
          ImageTy->getOperand(6).getImm(), ImageTy->getOperand(7).getImm(), AC)
          .Val,
      SpecialTypeKind::STK_SampledImage);
}

inline IRHandle make_descr_sampler() {
  return std::make_tuple(nullptr, 0U, SpecialTypeKind::STK_Sampler);
}

inline IRHandle make_descr_pipe(uint8_t AQ) {
  return std::make_tuple(nullptr, AQ, SpecialTypeKind::STK_Pipe);
}

inline IRHandle make_descr_event() {
  return std::make_tuple(nullptr, 0U, SpecialTypeKind::STK_DeviceEvent);
}

inline IRHandle make_descr_pointee(const Type *ElementType,
                                   unsigned AddressSpace) {
  return std::make_tuple(ElementType, AddressSpace,
                         SpecialTypeKind::STK_ElementPointer);
}

inline IRHandle make_descr_ptr(const void *Ptr) {
  return std::make_tuple(Ptr, 0U, SpecialTypeKind::STK_Pointer);
}
} // namespace SPIRV

// Bi-directional mappings between LLVM entities and (v-reg, machine function)
// pairs support management of unique SPIR-V definitions per machine function
// per an LLVM/GlobalISel entity (e.g., Type, Constant, Machine Instruction).
class SPIRVIRMapping {
  DenseMap < std::pair<IRHandle, const MachineFunction *>,
      const MachineInstr *MI >> Vregs;
  DenseMap<const MachineInstr *, IRHandle> Defs;

public:
  bool add(IRHandle Handle, const MachineInstr *MI) {
    auto [It, Inserted] =
        Vregs.try_emplace(std::make_pair(Handle, MI->getMF()), MI);
    if (Inserted) {
      auto [_, IsConsistent] = Defs.insert_or_assign(MI, Handle);
      assert(IsConsistent);
    }
    return Inserted1;
  }
  bool erase(const MachineInstr *MI) {
    bool Res = false;
    if (auto It = Defs.find(MI); It != Defs.end()) {
      Res = Vregs.erase(std::make_pair(It->second, MI->getMF()));
      Defs.erase(It);
    }
    return Res;
  }
  const MachineInstr *findMI(IRHandle Handle, const MachineFunction *MF) {
    if (auto It = Vregs.find(std::make_pair(Handle, MF)); It != Vregs.end())
      return It->second;
    return nullptr;
  }
  Register find(IRHandle Handle, const MachineFunction *MF) {
    const MachineInstr *MI = findMI(Handle, MF);
    return MI ? MI->getOperand(0).getReg() : Register();
  }

  // helpers
  bool add(const Type *Ty, const MachineInstr *MI) {
    return add(SPIRV::make_descr_ptr(unifyPtrType(Ty)), MI);
  }
  void add(const void *Key, const MachineInstr *MI) {
    return add(SPIRV::make_descr_ptr(Key), MI);
  }
  bool add(const Type *PointeeTy, unsigned AddressSpace,
           const MachineInstr *MI) {
    return add(SPIRV::make_descr_pointee(unifyPtrType(PointeeTy), AddressSpace),
               MI);
  }
  Register find(const Type *Ty, const MachineFunction *MF) {
    return find(SPIRV::make_descr_ptr(unifyPtrType(Ty)), MF);
  }
  Register find(const void *Key, const MachineFunction *MF) {
    return find(SPIRV::make_descr_ptr(Key), MF);
  }
  Register find(const Type *PointeeTy, unsigned AddressSpace,
                const MachineFunction *MF) {
    return find(
        SPIRV::make_descr_pointee(unifyPtrType(PointeeTy), AddressSpace), MF);
  }
};
} // namespace llvm
#endif // LLVM_LIB_TARGET_SPIRV_SPIRVIRMAPPING_H
