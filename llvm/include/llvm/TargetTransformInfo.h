//===- llvm/Transforms/TargetTransformInfo.h --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass exposes codegen information to IR-level passes. Every
// transformation that uses codegen information is broken into three parts:
// 1. The IR-level analysis pass.
// 2. The IR-level transformation interface which provides the needed
//    information.
// 3. Codegen-level implementation which uses target-specific hooks.
//
// This file defines #2, which is the interface that IR-level transformations
// use for querying the codegen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TARGET_TRANSFORM_INTERFACE
#define LLVM_TRANSFORMS_TARGET_TRANSFORM_INTERFACE

#include "llvm/Pass.h"
#include "llvm/AddressingMode.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Type.h"

namespace llvm {

class ScalarTargetTransformInfo;
class VectorTargetTransformInfo;

/// TargetTransformInfo - This pass provides access to the codegen
/// interfaces that are needed for IR-level transformations.
class TargetTransformInfo : public ImmutablePass {
private:
  const ScalarTargetTransformInfo *STTI;
  const VectorTargetTransformInfo *VTTI;
public:
  /// Default ctor.
  ///
  /// @note This has to exist, because this is a pass, but it should never be
  /// used.
  TargetTransformInfo();

  TargetTransformInfo(const ScalarTargetTransformInfo* S,
                      const VectorTargetTransformInfo *V)
      : ImmutablePass(ID), STTI(S), VTTI(V) {
    initializeTargetTransformInfoPass(*PassRegistry::getPassRegistry());
  }

  TargetTransformInfo(const TargetTransformInfo &T) :
    ImmutablePass(ID), STTI(T.STTI), VTTI(T.VTTI) { }

  const ScalarTargetTransformInfo* getScalarTargetTransformInfo() const {
    return STTI;
  }
  const VectorTargetTransformInfo* getVectorTargetTransformInfo() const {
    return VTTI;
  }

  /// Pass identification, replacement for typeid.
  static char ID;
};

// ---------------------------------------------------------------------------//
//  The classes below are inherited and implemented by target-specific classes
//  in the codegen.
// ---------------------------------------------------------------------------//

/// ScalarTargetTransformInfo - This interface is used by IR-level passes
/// that need target-dependent information for generic scalar transformations.
/// LSR, and LowerInvoke use this interface.
class ScalarTargetTransformInfo {
public:
  virtual ~ScalarTargetTransformInfo() {}

  /// isLegalAddImmediate - Return true if the specified immediate is legal
  /// add immediate, that is the target has add instructions which can add
  /// a register with the immediate without having to materialize the
  /// immediate into a register.
  virtual bool isLegalAddImmediate(int64_t) const {
    return false;
  }
  /// isLegalICmpImmediate - Return true if the specified immediate is legal
  /// icmp immediate, that is the target has icmp instructions which can compare
  /// a register against the immediate without having to materialize the
  /// immediate into a register.
  virtual bool isLegalICmpImmediate(int64_t) const {
    return false;
  }
  /// isLegalAddressingMode - Return true if the addressing mode represented by
  /// AM is legal for this target, for a load/store of the specified type.
  /// The type may be VoidTy, in which case only return true if the addressing
  /// mode is legal for a load/store of any legal type.
  /// TODO: Handle pre/postinc as well.
  virtual bool isLegalAddressingMode(const AddrMode &AM, Type *Ty) const {
    return false;
  }
  /// isTruncateFree - Return true if it's free to truncate a value of
  /// type Ty1 to type Ty2. e.g. On x86 it's free to truncate a i32 value in
  /// register EAX to i16 by referencing its sub-register AX.
  virtual bool isTruncateFree(Type *Ty1, Type *Ty2) const {
    return false;
  }
  /// Is this type legal.
  virtual bool isTypeLegal(Type *Ty) const {
    return false;
  }
  /// getJumpBufAlignment - returns the target's jmp_buf alignment in bytes
  virtual unsigned getJumpBufAlignment() const {
    return 0;
  }
  /// getJumpBufSize - returns the target's jmp_buf size in bytes.
  virtual unsigned getJumpBufSize() const {
    return 0;
  }
  /// shouldBuildLookupTables - Return true if switches should be turned into
  /// lookup tables for the target.
  virtual bool shouldBuildLookupTables() const {
    return true;
  }
};

/// VectorTargetTransformInfo - This interface is used by the vectorizers
/// to estimate the profitability of vectorization for different instructions.
class VectorTargetTransformInfo {
public:
  virtual ~VectorTargetTransformInfo() {}

  /// Returns the expected cost of the instruction opcode. The opcode is one of
  /// the enums like Instruction::Add. The type arguments are the type of the
  /// operation.
  /// Most instructions only use the first type and in that case the second
  /// operand is ignored.
  ///
  /// Exceptions:
  /// * Br instructions do not use any of the types.
  /// * Select instructions pass the return type as Ty1 and the selector as Ty2.
  /// * Cast instructions pass the destination as Ty1 and the source as Ty2.
  /// * Insert/Extract element pass only the vector type as Ty1.
  /// * ShuffleVector, Load, Store do not use this call.
  virtual unsigned getInstrCost(unsigned Opcode,
                                Type *Ty1 = 0,
                                Type *Ty2 = 0) const {
    return 1;
  }

  /// Returns the expected cost of arithmetic ops, such as mul, xor, fsub, etc.
  virtual unsigned getArithmeticInstrCost(unsigned Opcode, Type *Ty) const {
    return 1;
  }

  /// Returns the cost of a vector broadcast of a scalar at place zero to a
  /// vector of type 'Tp'.
  virtual unsigned getBroadcastCost(Type *Tp) const {
    return 1;
  }

  /// Returns the expected cost of cast instructions, such as bitcast, trunc,
  /// zext, etc.
  virtual unsigned getCastInstrCost(unsigned Opcode, Type *Dst,
                                    Type *Src) const {
    return 1;
  }

  /// Returns the expected cost of control-flow related instrutctions such as
  /// Phi, Ret, Br.
  virtual unsigned getCFInstrCost(unsigned Opcode) const {
    return 1;
  }

  /// Returns the expected cost of compare and select instructions.
  virtual unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                      Type *CondTy = 0) const {
    return 1;
  }

  /// Returns the expected cost of vector Insert and Extract.
  /// Use -1 to indicate that there is no information on the index value.
  virtual unsigned getVectorInstrCost(unsigned Opcode, Type *Val,
                                      unsigned Index = -1) const {
    return 1;
  }

  /// Returns the cost of Load and Store instructions.
  virtual unsigned getMemoryOpCost(unsigned Opcode, Type *Src,
                                   unsigned Alignment,
                                   unsigned AddressSpace) const {
    return 1;
  }

  /// Returns the number of pieces into which the provided type must be
  /// split during legalization. Zero is returned when the answer is unknown.
  virtual unsigned getNumberOfParts(Type *Tp) const {
    return 0;
  }
};

} // End llvm namespace

#endif
