//===- Intrinsics.h - LLVM Intrinsic Function Handling ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a set of enums which allow processing of intrinsic
// functions. Values of these enum types are returned by
// Function::getIntrinsicID.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_INTRINSICS_H
#define LLVM_IR_INTRINSICS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/TypeSize.h"
#include <optional>
#include <string>

namespace llvm {

class Type;
class FunctionType;
class Function;
class LLVMContext;
class Module;
class AttributeList;
class AttributeSet;

/// This namespace contains an enum with a value for every intrinsic/builtin
/// function known by LLVM. The enum values are returned by
/// Function::getIntrinsicID().
namespace Intrinsic {
  // Abstraction for the arguments of the noalias intrinsics
  static const int NoAliasScopeDeclScopeArg = 0;

  // Intrinsic ID type. This is an opaque typedef to facilitate splitting up
  // the enum into target-specific enums.
  typedef unsigned ID;

  enum IndependentIntrinsics : unsigned {
    not_intrinsic = 0, // Must be zero

  // Get the intrinsic enums generated from Intrinsics.td
#define GET_INTRINSIC_ENUM_VALUES
#include "llvm/IR/IntrinsicEnums.inc"
#undef GET_INTRINSIC_ENUM_VALUES
  };

  /// Return the LLVM name for an intrinsic, such as "llvm.ppc.altivec.lvx".
  /// Note, this version is for intrinsics with no overloads.  Use the other
  /// version of getName if overloads are required.
  LLVM_ABI StringRef getName(ID id);

  /// Return the LLVM name for an intrinsic, without encoded types for
  /// overloading, such as "llvm.ssa.copy".
  LLVM_ABI StringRef getBaseName(ID id);

  /// Return the LLVM name for an intrinsic, such as "llvm.ppc.altivec.lvx" or
  /// "llvm.ssa.copy.p0s_s.1". Note, this version of getName supports overloads.
  /// This is less efficient than the StringRef version of this function.  If no
  /// overloads are required, it is safe to use this version, but better to use
  /// the StringRef version. If one of the types is based on an unnamed type, a
  /// function type will be computed. Providing FT will avoid this computation.
  LLVM_ABI std::string getName(ID Id, ArrayRef<Type *> Tys, Module *M,
                               FunctionType *FT = nullptr);

  /// Return the LLVM name for an intrinsic. This is a special version only to
  /// be used by LLVMIntrinsicCopyOverloadedName. It only supports overloads
  /// based on named types.
  LLVM_ABI std::string getNameNoUnnamedTypes(ID Id, ArrayRef<Type *> Tys);

  /// Return the function type for an intrinsic.
  LLVM_ABI FunctionType *getType(LLVMContext &Context, ID id,
                                 ArrayRef<Type *> Tys = {});

  /// Returns true if the intrinsic can be overloaded.
  LLVM_ABI bool isOverloaded(ID id);

  /// isTargetIntrinsic - Returns true if IID is an intrinsic specific to a
  /// certain target. If it is a generic intrinsic false is returned.
  LLVM_ABI bool isTargetIntrinsic(ID IID);

  LLVM_ABI ID lookupIntrinsicID(StringRef Name);

  /// Return the attributes for an intrinsic.
  LLVM_ABI AttributeList getAttributes(LLVMContext &C, ID id, FunctionType *FT);

  /// Return the function attributes for an intrinsic.
  LLVM_ABI AttributeSet getFnAttributes(LLVMContext &C, ID id);

  /// Look up the Function declaration of the intrinsic \p id in the Module
  /// \p M. If it does not exist, add a declaration and return it. Otherwise,
  /// return the existing declaration.
  ///
  /// The \p Tys parameter is for intrinsics with overloaded types (e.g., those
  /// using iAny, fAny, vAny, or pAny).  For a declaration of an overloaded
  /// intrinsic, Tys must provide exactly one type for each overloaded type in
  /// the intrinsic.
  LLVM_ABI Function *getOrInsertDeclaration(Module *M, ID id,
                                            ArrayRef<Type *> Tys = {});

  /// Look up the Function declaration of the intrinsic \p id in the Module
  /// \p M and return it if it exists. Otherwise, return nullptr. This version
  /// supports non-overloaded intrinsics.
  LLVM_ABI Function *getDeclarationIfExists(const Module *M, ID id);

  /// This version supports overloaded intrinsics.
  LLVM_ABI Function *getDeclarationIfExists(Module *M, ID id,
                                            ArrayRef<Type *> Tys,
                                            FunctionType *FT = nullptr);

  /// Map a Clang builtin name to an intrinsic ID.
  LLVM_ABI ID getIntrinsicForClangBuiltin(StringRef TargetPrefix,
                                          StringRef BuiltinName);

  /// Map a MS builtin name to an intrinsic ID.
  LLVM_ABI ID getIntrinsicForMSBuiltin(StringRef TargetPrefix,
                                       StringRef BuiltinName);

  /// Returns true if the intrinsic ID is for one of the "Constrained
  /// Floating-Point Intrinsics".
  LLVM_ABI bool isConstrainedFPIntrinsic(ID QID);

  /// Returns true if the intrinsic ID is for one of the "Constrained
  /// Floating-Point Intrinsics" that take rounding mode metadata.
  LLVM_ABI bool hasConstrainedFPRoundingModeOperand(ID QID);

  /// This is a type descriptor which explains the type requirements of an
  /// intrinsic. This is returned by getIntrinsicInfoTableEntries.
  struct IITDescriptor {
    enum IITDescriptorKind {
      Void,
      VarArg,
      MMX,
      Token,
      Metadata,
      Half,
      BFloat,
      Float,
      Double,
      Quad,
      Integer,
      Vector,
      Pointer,
      Struct,
      Argument,
      ExtendArgument,
      TruncArgument,
      OneNthEltsVecArgument,
      SameVecWidthArgument,
      VecOfAnyPtrsToElt,
      VecElementArgument,
      Subdivide2Argument,
      Subdivide4Argument,
      VecOfBitcastsToInt,
      AMX,
      PPCQuad,
      AArch64Svcount,
    } Kind;

    union {
      unsigned Integer_Width;
      unsigned Float_Width;
      unsigned Pointer_AddressSpace;
      unsigned Struct_NumElements;
      unsigned Argument_Info;
      ElementCount Vector_Width;
    };

    // AK_% : Defined in Intrinsics.td
    enum ArgKind {
#define GET_INTRINSIC_ARGKIND
#include "llvm/IR/IntrinsicEnums.inc"
#undef GET_INTRINSIC_ARGKIND
    };

    unsigned getArgumentNumber() const {
      assert(Kind == Argument || Kind == ExtendArgument ||
             Kind == TruncArgument || Kind == SameVecWidthArgument ||
             Kind == VecElementArgument || Kind == Subdivide2Argument ||
             Kind == Subdivide4Argument || Kind == VecOfBitcastsToInt);
      return Argument_Info >> 3;
    }
    ArgKind getArgumentKind() const {
      assert(Kind == Argument || Kind == ExtendArgument ||
             Kind == TruncArgument || Kind == SameVecWidthArgument ||
             Kind == VecElementArgument || Kind == Subdivide2Argument ||
             Kind == Subdivide4Argument || Kind == VecOfBitcastsToInt);
      return (ArgKind)(Argument_Info & 7);
    }

    // VecOfAnyPtrsToElt uses both an overloaded argument (for address space)
    // and a reference argument (for matching vector width and element types)
    unsigned getOverloadArgNumber() const {
      assert(Kind == VecOfAnyPtrsToElt);
      return Argument_Info >> 16;
    }
    // OneNthEltsVecArguments uses both a divisor N and a reference argument for
    // the full-width vector to match
    unsigned getVectorDivisor() const {
      assert(Kind == OneNthEltsVecArgument);
      return Argument_Info >> 16;
    }
    unsigned getRefArgNumber() const {
      assert(Kind == VecOfAnyPtrsToElt || Kind == OneNthEltsVecArgument);
      return Argument_Info & 0xFFFF;
    }

    static IITDescriptor get(IITDescriptorKind K, unsigned Field) {
      IITDescriptor Result = { K, { Field } };
      return Result;
    }

    static IITDescriptor get(IITDescriptorKind K, unsigned short Hi,
                             unsigned short Lo) {
      unsigned Field = Hi << 16 | Lo;
      IITDescriptor Result = {K, {Field}};
      return Result;
    }

    static IITDescriptor getVector(unsigned Width, bool IsScalable) {
      IITDescriptor Result = {Vector, {0}};
      Result.Vector_Width = ElementCount::get(Width, IsScalable);
      return Result;
    }
  };

  /// Return the IIT table descriptor for the specified intrinsic into an array
  /// of IITDescriptors.
  LLVM_ABI void getIntrinsicInfoTableEntries(ID id,
                                             SmallVectorImpl<IITDescriptor> &T);

  enum MatchIntrinsicTypesResult {
    MatchIntrinsicTypes_Match = 0,
    MatchIntrinsicTypes_NoMatchRet = 1,
    MatchIntrinsicTypes_NoMatchArg = 2,
  };

  /// Match the specified function type with the type constraints specified by
  /// the .td file. If the given type is an overloaded type it is pushed to the
  /// ArgTys vector.
  ///
  /// Returns false if the given type matches with the constraints, true
  /// otherwise.
  LLVM_ABI MatchIntrinsicTypesResult
  matchIntrinsicSignature(FunctionType *FTy, ArrayRef<IITDescriptor> &Infos,
                          SmallVectorImpl<Type *> &ArgTys);

  /// Verify if the intrinsic has variable arguments. This method is intended to
  /// be called after all the fixed arguments have been matched first.
  ///
  /// This method returns true on error.
  LLVM_ABI bool matchIntrinsicVarArg(bool isVarArg,
                                     ArrayRef<IITDescriptor> &Infos);

  /// Gets the type arguments of an intrinsic call by matching type contraints
  /// specified by the .td file. The overloaded types are pushed into the
  /// AgTys vector.
  ///
  /// Returns false if the given ID and function type combination is not a
  /// valid intrinsic call.
  LLVM_ABI bool getIntrinsicSignature(Intrinsic::ID, FunctionType *FT,
                                      SmallVectorImpl<Type *> &ArgTys);

  /// Same as previous, but accepts a Function instead of ID and FunctionType.
  LLVM_ABI bool getIntrinsicSignature(Function *F,
                                      SmallVectorImpl<Type *> &ArgTys);

  // Checks if the intrinsic name matches with its signature and if not
  // returns the declaration with the same signature and remangled name.
  // An existing GlobalValue with the wanted name but with a wrong prototype
  // or of the wrong kind will be renamed by adding ".renamed" to the name.
  LLVM_ABI std::optional<Function *> remangleIntrinsicFunction(Function *F);

  /// Returns the corresponding llvm.vector.interleaveN intrinsic for factor N.
  LLVM_ABI Intrinsic::ID getInterleaveIntrinsicID(unsigned Factor);

  /// Returns the corresponding llvm.vector.deinterleaveN intrinsic for factor
  /// N.
  LLVM_ABI Intrinsic::ID getDeinterleaveIntrinsicID(unsigned Factor);

  } // namespace Intrinsic

  } // namespace llvm

#endif
