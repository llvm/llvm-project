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
#include "llvm/ADT/Hashing.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

#include <type_traits>

namespace llvm {
namespace SPIRV {

inline size_t to_hash(const MachineInstr *MI,
                      std::unordered_set<const MachineInstr *> &Visited) {
  if (!MI || !Visited.insert(MI).second)
    return 0;
  const MachineRegisterInfo &MRI = MI->getMF()->getRegInfo();
  SmallVector<size_t, 16> Codes{MI->getOpcode()};
  size_t H;
  for (unsigned I = MI->getNumDefs(); I < MI->getNumOperands(); ++I) {
    const MachineOperand &MO = MI->getOperand(I);
    H = MO.isReg() ? to_hash(getDef(MO, &MRI), Visited)
                   : size_t(llvm::hash_value(MO));
    Codes.push_back(H);
  }
  return llvm::hash_combine(Codes.begin(), Codes.end());
}

inline size_t to_hash(const MachineInstr *MI) {
  std::unordered_set<const MachineInstr *> Visited;
  return to_hash(MI, Visited);
}

using MIHandle = std::pair<const MachineInstr *, size_t>;

inline MIHandle getMIKey(const MachineInstr *MI) {
  return std::make_pair(MI, SPIRV::to_hash(MI));
}

using IRHandle = std::tuple<const void *, unsigned, unsigned>;
using IRHandleMF = std::pair<IRHandle, const MachineFunction *>;

inline IRHandleMF getIRHandleMF(IRHandle Handle, const MachineFunction *MF) {
  return std::make_pair(Handle, MF);
}

enum SpecialTypeKind {
  STK_Empty = 0,
  STK_Image,
  STK_SampledImage,
  STK_Sampler,
  STK_Pipe,
  STK_DeviceEvent,
  STK_ElementPointer,
  STK_Type,
  STK_Value,
  STK_MachineInstr,
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

inline IRHandle irhandle_image(const Type *SampledTy, unsigned Dim,
                               unsigned Depth, unsigned Arrayed, unsigned MS,
                               unsigned Sampled, unsigned ImageFormat,
                               unsigned AQ = 0) {
  return std::make_tuple(
      SampledTy,
      ImageAttrs(Dim, Depth, Arrayed, MS, Sampled, ImageFormat, AQ).Val,
      SpecialTypeKind::STK_Image);
}

inline IRHandle irhandle_sampled_image(const Type *SampledTy,
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

inline IRHandle irhandle_sampler() {
  return std::make_tuple(nullptr, 0U, SpecialTypeKind::STK_Sampler);
}

inline IRHandle irhandle_pipe(uint8_t AQ) {
  return std::make_tuple(nullptr, AQ, SpecialTypeKind::STK_Pipe);
}

inline IRHandle irhandle_event() {
  return std::make_tuple(nullptr, 0U, SpecialTypeKind::STK_DeviceEvent);
}

inline IRHandle irhandle_pointee(const Type *ElementType,
                                 unsigned AddressSpace) {
  return std::make_tuple(unifyPtrType(ElementType), AddressSpace,
                         SpecialTypeKind::STK_ElementPointer);
}

inline IRHandle irhandle_ptr(const void *Ptr, unsigned Arg,
                             enum SpecialTypeKind STK) {
  return std::make_tuple(Ptr, Arg, STK);
}

inline IRHandle handle(const Type *Ty) {
  const Type *WrpTy = unifyPtrType(Ty);
  return irhandle_ptr(WrpTy, Ty->getTypeID(), STK_Type);
}

inline IRHandle handle(const Value *V) {
  return irhandle_ptr(V, V->getValueID(), STK_Value);
}

inline IRHandle handle(const MachineInstr *KeyMI) {
  return irhandle_ptr(KeyMI, SPIRV::to_hash(KeyMI), STK_MachineInstr);
}

} // namespace SPIRV

// Bi-directional mappings between LLVM entities and (v-reg, machine function)
// pairs support management of unique SPIR-V definitions per machine function
// per an LLVM/GlobalISel entity (e.g., Type, Constant, Machine Instruction).
class SPIRVIRMapping {
  DenseMap<SPIRV::IRHandleMF, SPIRV::MIHandle> Vregs;
  DenseMap<SPIRV::MIHandle, SPIRV::IRHandle> Defs;

public:
  bool add(SPIRV::IRHandle Handle, const MachineInstr *MI) {
    if (std::get<1>(Handle) == 17 && std::get<2>(Handle) == 8) {
      const Value *Ptr = (const Value *)std::get<0>(Handle);
      if (const ConstantInt *CI = dyn_cast_or_null<ConstantInt>(Ptr)) {
        if (CI->getZExtValue() == 8 || CI->getZExtValue() == 5) {
          [[maybe_unused]] uint64_t v = CI->getZExtValue();
        }
      }
    }
    auto MIKey = SPIRV::getMIKey(MI);
    auto [It, Inserted] =
        Vregs.try_emplace(std::make_pair(Handle, MI->getMF()), MIKey);
    if (Inserted) {
      [[maybe_unused]] auto [_, IsConsistent] =
          Defs.insert_or_assign(MIKey, Handle);
      assert(IsConsistent);
    }
    return Inserted;
  }
  bool erase(const MachineInstr *MI) {
    bool Res = false;
    if (auto It = Defs.find(SPIRV::getMIKey(MI)); It != Defs.end()) {
      Res = Vregs.erase(SPIRV::getIRHandleMF(It->second, MI->getMF()));
      Defs.erase(It);
    }
    return Res;
  }
  const MachineInstr *findMI(SPIRV::IRHandle Handle,
                             const MachineFunction *MF) {
    auto It = Vregs.find(SPIRV::getIRHandleMF(Handle, MF));
    if (It == Vregs.end())
      return nullptr;
    auto [MI, Hash] = It->second;
    if (SPIRV::to_hash(MI) != Hash) {
      erase(MI);
      return nullptr;
    }
    return MI;
  }
  Register find(SPIRV::IRHandle Handle, const MachineFunction *MF) {
    const MachineInstr *MI = findMI(Handle, MF);
    return MI ? MI->getOperand(0).getReg() : Register();
  }

  // helpers
  bool add(const Type *PointeeTy, unsigned AddressSpace,
           const MachineInstr *MI) {
    return add(SPIRV::irhandle_pointee(PointeeTy, AddressSpace), MI);
  }
  Register find(const Type *PointeeTy, unsigned AddressSpace,
                const MachineFunction *MF) {
    return find(SPIRV::irhandle_pointee(PointeeTy, AddressSpace), MF);
  }
  const MachineInstr *findMI(const Type *PointeeTy, unsigned AddressSpace,
                             const MachineFunction *MF) {
    return findMI(SPIRV::irhandle_pointee(PointeeTy, AddressSpace), MF);
  }

  template <typename T> bool add(const T *Obj, const MachineInstr *MI) {
    return add(SPIRV::handle(Obj), MI);
  }
  template <typename T> Register find(const T *Obj, const MachineFunction *MF) {
    return find(SPIRV::handle(Obj), MF);
  }
  template <typename T>
  const MachineInstr *findMI(const T *Obj, const MachineFunction *MF) {
    return findMI(SPIRV::handle(Obj), MF);
  }
};
} // namespace llvm
#endif // LLVM_LIB_TARGET_SPIRV_SPIRVIRMAPPING_H
