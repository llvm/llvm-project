//===------------ SPIRVMapping.h - SPIR-V Duplicates Tracker ----*- C++ -*-===//
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

inline size_t to_hash(const MachineInstr *MI) {
  hash_code H = llvm::hash_combine(MI->getOpcode(), MI->getNumOperands());
  for (unsigned I = MI->getNumDefs(); I < MI->getNumOperands(); ++I) {
    const MachineOperand &MO = MI->getOperand(I);
    if (MO.getType() == MachineOperand::MO_CImmediate)
      H = llvm::hash_combine(H, MO.getType(), MO.getCImm());
    else if (MO.getType() == MachineOperand::MO_FPImmediate)
      H = llvm::hash_combine(H, MO.getType(), MO.getFPImm());
    else
      H = llvm::hash_combine(H, MO.getType());
  }
  return H;
}

using MIHandle = std::tuple<const MachineInstr *, Register, size_t>;

inline MIHandle getMIKey(const MachineInstr *MI) {
  return std::make_tuple(MI, MI->getOperand(0).getReg(), SPIRV::to_hash(MI));
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
  STK_VkBuffer,
  STK_ExplictLayoutType,
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

inline IRHandle irhandle_vkbuffer(const Type *ElementType,
                                  StorageClass::StorageClass SC,
                                  bool IsWriteable) {
  return std::make_tuple(ElementType, (SC << 1) | IsWriteable,
                         SpecialTypeKind::STK_VkBuffer);
}

inline IRHandle irhandle_explict_layout_type(const Type *Ty) {
  const Type *WrpTy = unifyPtrType(Ty);
  return irhandle_ptr(WrpTy, Ty->getTypeID(), STK_ExplictLayoutType);
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

inline bool type_has_layout_decoration(const Type *T) {
  return (isa<StructType>(T) || isa<ArrayType>(T));
}

} // namespace SPIRV

// Bi-directional mappings between LLVM entities and (v-reg, machine function)
// pairs support management of unique SPIR-V definitions per machine function
// per an LLVM/GlobalISel entity (e.g., Type, Constant, Machine Instruction).
class SPIRVIRMapping {
  DenseMap<SPIRV::IRHandleMF, SPIRV::MIHandle> Vregs;
  DenseMap<const MachineInstr *, SPIRV::IRHandleMF> Defs;

public:
  bool add(SPIRV::IRHandle Handle, const MachineInstr *MI) {
    if (auto DefIt = Defs.find(MI); DefIt != Defs.end()) {
      auto [ExistHandle, ExistMF] = DefIt->second;
      if (Handle == ExistHandle && MI->getMF() == ExistMF)
        return false; // already exists
      // invalidate the record
      Vregs.erase(DefIt->second);
      Defs.erase(DefIt);
    }
    SPIRV::IRHandleMF HandleMF = SPIRV::getIRHandleMF(Handle, MI->getMF());
    SPIRV::MIHandle MIKey = SPIRV::getMIKey(MI);
    auto It1 = Vregs.try_emplace(HandleMF, MIKey);
    if (!It1.second) {
      // there is an expired record that we need to invalidate
      Defs.erase(std::get<0>(It1.first->second));
      // update the record
      It1.first->second = MIKey;
    }
    [[maybe_unused]] auto It2 = Defs.try_emplace(MI, HandleMF);
    assert(It2.second);
    return true;
  }
  bool erase(const MachineInstr *MI) {
    bool Res = false;
    if (auto It = Defs.find(MI); It != Defs.end()) {
      Res = Vregs.erase(It->second);
      Defs.erase(It);
    }
    return Res;
  }
  const MachineInstr *findMI(SPIRV::IRHandle Handle,
                             const MachineFunction *MF) {
    SPIRV::IRHandleMF HandleMF = SPIRV::getIRHandleMF(Handle, MF);
    auto It = Vregs.find(HandleMF);
    if (It == Vregs.end())
      return nullptr;
    auto [MI, Reg, Hash] = It->second;
    const MachineInstr *Def = MF->getRegInfo().getVRegDef(Reg);
    if (!Def || Def != MI || SPIRV::to_hash(MI) != Hash) {
      // there is an expired record that we need to invalidate
      erase(MI);
      return nullptr;
    }
    assert(Defs.contains(MI) && Defs.find(MI)->second == HandleMF);
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

  bool add(const Value *V, const MachineInstr *MI) {
    return add(SPIRV::handle(V), MI);
  }

  bool add(const Type *T, bool RequiresExplicitLayout, const MachineInstr *MI) {
    if (RequiresExplicitLayout && SPIRV::type_has_layout_decoration(T)) {
      return add(SPIRV::irhandle_explict_layout_type(T), MI);
    }
    return add(SPIRV::handle(T), MI);
  }

  bool add(const MachineInstr *Obj, const MachineInstr *MI) {
    return add(SPIRV::handle(Obj), MI);
  }

  Register find(const Value *V, const MachineFunction *MF) {
    return find(SPIRV::handle(V), MF);
  }

  Register find(const Type *T, bool RequiresExplicitLayout,
                const MachineFunction *MF) {
    if (RequiresExplicitLayout && SPIRV::type_has_layout_decoration(T))
      return find(SPIRV::irhandle_explict_layout_type(T), MF);
    return find(SPIRV::handle(T), MF);
  }

  Register find(const MachineInstr *MI, const MachineFunction *MF) {
    return find(SPIRV::handle(MI), MF);
  }

  const MachineInstr *findMI(const Value *Obj, const MachineFunction *MF) {
    return findMI(SPIRV::handle(Obj), MF);
  }

  const MachineInstr *findMI(const Type *T, bool RequiresExplicitLayout,
                             const MachineFunction *MF) {
    if (RequiresExplicitLayout && SPIRV::type_has_layout_decoration(T))
      return findMI(SPIRV::irhandle_explict_layout_type(T), MF);
    return findMI(SPIRV::handle(T), MF);
  }

  const MachineInstr *findMI(const MachineInstr *Obj,
                             const MachineFunction *MF) {
    return findMI(SPIRV::handle(Obj), MF);
  }
};
} // namespace llvm
#endif // LLVM_LIB_TARGET_SPIRV_SPIRVIRMAPPING_H
