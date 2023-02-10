//===- ARMRandezvousInstrumentor.h - A helper class for instrumentation ---===//
//
// Copyright (c) 2021-2022, University of Rochester
//
// Part of the Randezvous Project, under the Apache License v2.0 with
// LLVM Exceptions.  See LICENSE.txt in the llvm directory for license
// information.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces of a class that can help passes of its
// subclass easily instrument ARM machine IR without concerns of breaking IT
// blocks.
//
//===----------------------------------------------------------------------===//

#ifndef ARM_RANDEZVOUS_INSTRUMENTOR
#define ARM_RANDEZVOUS_INSTRUMENTOR

#include "ARMBaseInstrInfo.h"
#include <deque>
namespace llvm {
  //====================================================================
  // Static inline functions.
  //====================================================================

  static inline size_t getBasicBlockCodeSize(const MachineBasicBlock & MBB) {
    const MachineFunction & MF = *MBB.getParent();
    const TargetInstrInfo * TII = MF.getSubtarget().getInstrInfo();

    size_t CodeSize = 0ul;
    for (const MachineInstr & MI : MBB) {
      CodeSize += TII->getInstSizeInBytes(MI);
    }

    return CodeSize;
  }

  //
  // Function: getFunctionCodeSize()
  //
  // Description:
  //   This function computes the code size of a machine function.
  //
  // Input:
  //   MF - A reference to the target machine function.
  //
  // Return value:
  //   The size (in bytes) of the machine function.
  //
  static inline size_t getFunctionCodeSize(const MachineFunction & MF) {
    size_t CodeSize = 0ul;
    for (const MachineBasicBlock & MBB : MF) {
      CodeSize += getBasicBlockCodeSize(MBB);
    }

    return CodeSize;
  }

  //
  // Function: containsFunctionPointerType()
  //
  // Description:
  //   This function examines a Type to see whether it can explicitly contain one
  //   or more function pointers.  Note that this function recurses on aggregate
  //   types.
  //
  // Input:
  //   Ty - A pointer to a Type to examine.
  //
  // Return value:
  //   true  - The Type can contain one or more function pointers.
  //   false - The Type does not contain a function pointer.
  //
//   static inline bool
//   containsFunctionPointerType(Type * Ty) {
//     // Pointer
//     if (PointerType * PtrTy = dyn_cast<PointerType>(Ty)) {
//       return PtrTy->getParamElementType()->isFunctionTy();
//     }

//     // Array
//     if (ArrayType * ArrayTy = dyn_cast<ArrayType>(Ty)) {
//       return containsFunctionPointerType(ArrayTy->getElementType());
//     }

//     // Struct
//     if (StructType * StructTy = dyn_cast<StructType>(Ty)) {
//       for (Type * ElementTy : StructTy->elements()) {
//         if (containsFunctionPointerType(ElementTy)) {
//           return true;
//         }
//       }
//     }

//     // Other types do not contain function pointers
//     return false;
//   }

  //
  // Function: createNonZeroInitializerFor()
  //
  // Description:
  //   This function creates a non-zero Constant initializer for a give Type,
  //   which is supposed to contain one or more function pointers.  Note that
  //   this function recurses on aggregate types.
  //
  // Input:
  //   Ty - A pointer to a Type for which to create an initializer.
  //
  // Return value:
  //   A pointer to a created Constant.
  //
  static inline Constant *
  createNonZeroInitializerFor(Type * Ty) {
    // Pointer: this is where we insert non-zero values
    if (PointerType * PtrTy = dyn_cast<PointerType>(Ty)) {
      return ConstantExpr::getIntToPtr(
        ConstantInt::get(Type::getInt32Ty(Ty->getContext()), 1), Ty
      );
    }

    // Array
    if (ArrayType * ArrayTy = dyn_cast<ArrayType>(Ty)) {
      std::vector<Constant *> InitArray;
      for (uint64_t i = 0; i < ArrayTy->getNumElements(); ++i) {
        InitArray.push_back(createNonZeroInitializerFor(ArrayTy->getElementType()));
      }
      return ConstantArray::get(ArrayTy, InitArray);
    }

    // Struct
    if (StructType * StructTy = dyn_cast<StructType>(Ty)) {
      std::vector<Constant *> InitArray;
      for (unsigned i = 0; i < StructTy->getNumElements(); ++i) {
        InitArray.push_back(createNonZeroInitializerFor(StructTy->getElementType(i)));
      }
      return ConstantStruct::get(StructTy, InitArray);
    }

    // Zeroing out other types are fine
    return Constant::getNullValue(Ty);
  }

  //====================================================================
  // Class ARMRandezvousInstrumentor.
  //====================================================================

  struct ARMRandezvousInstrumentor {
    void insertInstBefore(MachineInstr & MI, MachineInstr * Inst);

    void insertInstAfter(MachineInstr & MI, MachineInstr * Inst);

    void insertInstsBefore(MachineInstr & MI, ArrayRef<MachineInstr *> Insts);

    void insertInstsAfter(MachineInstr & MI, ArrayRef<MachineInstr *> Insts);

    void removeInst(MachineInstr & MI);

    MachineBasicBlock * splitBasicBlockBefore(MachineInstr & MI);

    MachineBasicBlock * splitBasicBlockAfter(MachineInstr & MI);

    std::vector<Register> findFreeRegistersBefore(const MachineInstr & MI,
                                                  bool Thumb = false);
    std::vector<Register> findFreeRegistersAfter(const MachineInstr & MI,
                                                 bool Thumb = false);

  private:
    unsigned getITBlockSize(const MachineInstr & IT);
    MachineInstr * findIT(MachineInstr & MI, unsigned & distance);
    const MachineInstr * findIT(const MachineInstr & MI, unsigned & distance);
    std::deque<bool> decodeITMask(unsigned Mask);
    unsigned encodeITMask(std::deque<bool> DQMask);
  };
}

#endif