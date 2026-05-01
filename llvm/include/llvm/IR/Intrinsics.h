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
class raw_ostream;
class Constant;

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
  LLVM_ABI std::string getName(ID Id, ArrayRef<Type *> OverloadTys, Module *M,
                               FunctionType *FT = nullptr);

  /// Return the LLVM name for an intrinsic. This is a special version only to
  /// be used by LLVMIntrinsicCopyOverloadedName. It only supports overloads
  /// based on named types.
  LLVM_ABI std::string getNameNoUnnamedTypes(ID Id,
                                             ArrayRef<Type *> OverloadTys);

  /// Return the function type for an intrinsic.
  LLVM_ABI FunctionType *getType(LLVMContext &Context, ID id,
                                 ArrayRef<Type *> OverloadTys = {});

  /// Returns true if the intrinsic can be overloaded.
  LLVM_ABI bool isOverloaded(ID id);

  /// Returns true if the intrinsic is trivially scalarizable.
  /// This means that the intrinsic's argument types are all scalars for the
  /// scalar form and all vectors for the vector form.
  LLVM_ABI bool isTriviallyScalarizable(ID id);

  /// Returns true if the intrinsic has pretty printed immediate arguments.
  LLVM_ABI bool hasPrettyPrintedArgs(ID id);

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
  /// The \p OverloadTys parameter is for intrinsics with overloaded types
  // (e.g., those using iAny, fAny, vAny, or pAny).  For a declaration of an
  // overloaded intrinsic, OverloadTys must provide exactly one type for each
  // overloaded type in the intrinsic.
  LLVM_ABI Function *getOrInsertDeclaration(Module *M, ID id,
                                            ArrayRef<Type *> OverloadTys = {});

  /// Look up the Function declaration of the intrinsic \p IID in the Module
  /// \p M. If it does not exist, add a declaration and return it. Otherwise,
  /// return the existing declaration.
  ///
  /// This overload automatically resolves overloaded intrinsics based on the
  /// provided return type and argument types. For non-overloaded intrinsics,
  /// the return type and argument types are ignored.
  ///
  /// \param M - The module to get or insert the intrinsic declaration.
  /// \param IID - The intrinsic ID.
  /// \param RetTy - The return type of the intrinsic.
  /// \param ArgTys - The argument types of the intrinsic.
  LLVM_ABI Function *getOrInsertDeclaration(Module *M, ID IID, Type *RetTy,
                                            ArrayRef<Type *> ArgTys);

  /// Look up the Function declaration of the intrinsic \p id in the Module
  /// \p M and return it if it exists. Otherwise, return nullptr. This version
  /// supports non-overloaded intrinsics.
  LLVM_ABI Function *getDeclarationIfExists(const Module *M, ID id);

  /// This version supports overloaded intrinsics.
  LLVM_ABI Function *getDeclarationIfExists(Module *M, ID id,
                                            ArrayRef<Type *> OverloadTys,
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
      // Concrete types. Additional qualifiers listed in comments.
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
      Integer, // Width of the integer in IntegerWidth.
      Vector,  // Width of the vector in VectorWidth.
      Pointer, // Address space of the pointer in PointerAddressSpace.
      Struct,  // Number of elements in StructNumElements.
      AMX,
      PPCQuad,
      AArch64Svcount,

      // Overloaded type.
      Overloaded, // AnyKind and overload index in OverloadInfo.

      // Fully dependent types. Overload index in OverloadInfo.
      Extend,
      Trunc,
      OneNthEltsVec,
      SameVecWidth,
      VecElement,
      Subdivide2,
      Subdivide4,
      VecOfBitcastsToInt,

      // Partially dependent types. Overload index (self and of the overload
      // type it depends on) in OverloadInfo.
      VecOfAnyPtrsToElt,

    } Kind;

    union {
      unsigned IntegerWidth;
      unsigned PointerAddressSpace;
      unsigned StructNumElements;
      unsigned OverloadInfo;
      ElementCount VectorWidth;
    };

    // AK_% : Defined in Intrinsics.td
    enum AnyKind {
#define GET_INTRINSIC_ANYKIND
#include "llvm/IR/IntrinsicEnums.inc"
    };

    unsigned getOverloadIndex() const {
      assert(Kind == Overloaded || Kind == Extend || Kind == Trunc ||
             Kind == SameVecWidth || Kind == VecElement || Kind == Subdivide2 ||
             Kind == Subdivide4 || Kind == VecOfBitcastsToInt ||
             Kind == VecOfAnyPtrsToElt || Kind == OneNthEltsVec);
      // Overload index is packed into lower 5 bits.
      return OverloadInfo & 0x1f;
    }

    AnyKind getOverloadKind() const {
      // Overload kind is packed into upper 3 bits.
      assert(Kind == Overloaded);
      return (AnyKind)((OverloadInfo >> 5) & 0x7);
    }

    // OneNthEltsVecArguments uses both a divisor N and a reference argument for
    // the full-width vector to match.
    unsigned getVectorDivisor() const {
      assert(Kind == OneNthEltsVec);
      return OverloadInfo >> 16;
    }

    unsigned getRefOverloadIndex() const {
      assert(Kind == VecOfAnyPtrsToElt);
      return OverloadInfo >> 16;
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
      Result.VectorWidth = ElementCount::get(Width, IsScalable);
      return Result;
    }
  };

  /// Return the IIT table descriptor for the specified intrinsic into an array
  /// of IITDescriptors.
  LLVM_ABI void getIntrinsicInfoTableEntries(ID id,
                                             SmallVectorImpl<IITDescriptor> &T);

  /// Returns true if \p FT is a valid function type for intrinsic \p ID. If
  /// `ID` is an overloaded intrinsic, the overload types are pushed into the
  /// OverloadTys vector.
  ///
  /// Returns false if the given ID and function type combination is not a
  /// valid intrinsic call. Also prints the error message to indicate the reason
  /// of the mismatch to \p OS.
  LLVM_ABI bool isSignatureValid(Intrinsic::ID ID, FunctionType *FT,
                                 SmallVectorImpl<Type *> &OverloadTys,
                                 raw_ostream &OS = nulls());

  /// Same as previous, but accepts a Function instead of ID and FunctionType.
  LLVM_ABI bool isSignatureValid(Function *F,
                                 SmallVectorImpl<Type *> &OverloadTys,
                                 raw_ostream &OS = nulls());

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

  /// Print the argument info for the arguments with ArgInfo.
  LLVM_ABI void printImmArg(ID IID, unsigned ArgIdx, raw_ostream &OS,
                            const Constant *ImmArgVal);

  } // namespace Intrinsic

  } // namespace llvm

#endif
