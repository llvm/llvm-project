//===- CodeGenIntrinsic.h - Intrinsic Class Wrapper ------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a wrapper class for the 'Intrinsic' TableGen class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_CODEGENINTRINSICS_H
#define LLVM_UTILS_TABLEGEN_CODEGENINTRINSICS_H

#include "SDNodeProperties.h"
#include "llvm/Support/MachineValueType.h"
#include <string>
#include <vector>

namespace llvm {
class Record;
class RecordKeeper;

struct CodeGenIntrinsic {
  Record *TheDef;             // The actual record defining this intrinsic.
  std::string Name;           // The name of the LLVM function "llvm.bswap.i32"
  std::string EnumName;       // The name of the enum "bswap_i32"
  std::string GCCBuiltinName; // Name of the corresponding GCC builtin, or "".
  std::string MSBuiltinName;  // Name of the corresponding MS builtin, or "".
  std::string TargetPrefix;   // Target prefix, e.g. "ppc" for t-s intrinsics.

  /// This structure holds the return values and parameter values of an
  /// intrinsic. If the number of return values is > 1, then the intrinsic
  /// implicitly returns a first-class aggregate. The numbering of the types
  /// starts at 0 with the first return value and continues from there through
  /// the parameter list. This is useful for "matching" types.
  struct IntrinsicSignature {
    /// The MVT::SimpleValueType for each return type. Note that this list is
    /// only populated when in the context of a target .td file. When building
    /// Intrinsics.td, this isn't available, because we don't know the target
    /// pointer size.
    std::vector<MVT::SimpleValueType> RetVTs;

    /// The records for each return type.
    std::vector<Record *> RetTypeDefs;

    /// The MVT::SimpleValueType for each parameter type. Note that this list is
    /// only populated when in the context of a target .td file.  When building
    /// Intrinsics.td, this isn't available, because we don't know the target
    /// pointer size.
    std::vector<MVT::SimpleValueType> ParamVTs;

    /// The records for each parameter type.
    std::vector<Record *> ParamTypeDefs;
  };

  IntrinsicSignature IS;

  /// Bit flags describing the type (ref/mod) and location of memory
  /// accesses that may be performed by the intrinsics. Analogous to
  /// \c FunctionModRefBehaviour.
  enum ModRefBits {
    /// The intrinsic may access memory that is otherwise inaccessible via
    /// LLVM IR.
    MR_InaccessibleMem = 1,

    /// The intrinsic may access memory through pointer arguments.
    /// LLVM IR.
    MR_ArgMem = 2,

    /// The intrinsic may access memory anywhere, i.e. it is not restricted
    /// to access through pointer arguments.
    MR_Anywhere = 4 | MR_ArgMem | MR_InaccessibleMem,

    /// The intrinsic may read memory.
    MR_Ref = 8,

    /// The intrinsic may write memory.
    MR_Mod = 16,

    /// The intrinsic may both read and write memory.
    MR_ModRef = MR_Ref | MR_Mod,
  };

  /// Memory mod/ref behavior of this intrinsic, corresponding to intrinsic
  /// properties (IntrReadMem, IntrArgMemOnly, etc.).
  enum ModRefBehavior {
    NoMem = 0,
    ReadArgMem = MR_Ref | MR_ArgMem,
    ReadInaccessibleMem = MR_Ref | MR_InaccessibleMem,
    ReadInaccessibleMemOrArgMem = MR_Ref | MR_ArgMem | MR_InaccessibleMem,
    ReadMem = MR_Ref | MR_Anywhere,
    WriteArgMem = MR_Mod | MR_ArgMem,
    WriteInaccessibleMem = MR_Mod | MR_InaccessibleMem,
    WriteInaccessibleMemOrArgMem = MR_Mod | MR_ArgMem | MR_InaccessibleMem,
    WriteMem = MR_Mod | MR_Anywhere,
    ReadWriteArgMem = MR_ModRef | MR_ArgMem,
    ReadWriteInaccessibleMem = MR_ModRef | MR_InaccessibleMem,
    ReadWriteInaccessibleMemOrArgMem = MR_ModRef | MR_ArgMem |
                                       MR_InaccessibleMem,
    ReadWriteMem = MR_ModRef | MR_Anywhere,
  };
  ModRefBehavior ModRef;

  /// SDPatternOperator Properties applied to the intrinsic.
  unsigned Properties;

  /// This is set to true if the intrinsic is overloaded by its argument
  /// types.
  bool isOverloaded;

  /// True if the intrinsic is commutative.
  bool isCommutative;

  /// True if the intrinsic can throw.
  bool canThrow;

  /// True if the intrinsic is marked as noduplicate.
  bool isNoDuplicate;

  /// True if the intrinsic is marked as nomerge.
  bool isNoMerge;

  /// True if the intrinsic is no-return.
  bool isNoReturn;

  /// True if the intrinsic is no-sync.
  bool isNoSync;

  /// True if the intrinsic is no-free.
  bool isNoFree;

  /// True if the intrinsic is will-return.
  bool isWillReturn;

  /// True if the intrinsic is cold.
  bool isCold;

  /// True if the intrinsic is marked as convergent.
  bool isConvergent;

  /// True if the intrinsic has side effects that aren't captured by any
  /// of the other flags.
  bool hasSideEffects;

  // True if the intrinsic is marked as speculatable.
  bool isSpeculatable;

  enum ArgAttrKind {
    NoCapture,
    NoAlias,
    NoUndef,
    Returned,
    ReadOnly,
    WriteOnly,
    ReadNone,
    ImmArg,
    Alignment
  };

  struct ArgAttribute {
    unsigned Index;
    ArgAttrKind Kind;
    uint64_t Value;

    ArgAttribute(unsigned Idx, ArgAttrKind K, uint64_t V)
        : Index(Idx), Kind(K), Value(V) {}

    bool operator<(const ArgAttribute &Other) const {
      return std::tie(Index, Kind, Value) <
             std::tie(Other.Index, Other.Kind, Other.Value);
    }
  };

  std::vector<ArgAttribute> ArgumentAttributes;

  bool hasProperty(enum SDNP Prop) const {
    return Properties & (1 << Prop);
  }

  /// Goes through all IntrProperties that have IsDefault
  /// value set and sets the property.
  void setDefaultProperties(Record *R, std::vector<Record *> DefaultProperties);

  /// Helper function to set property \p Name to true;
  void setProperty(Record *R);

  /// Returns true if the parameter at \p ParamIdx is a pointer type. Returns
  /// false if the parameter is not a pointer, or \p ParamIdx is greater than
  /// the size of \p IS.ParamVTs.
  ///
  /// Note that this requires that \p IS.ParamVTs is available.
  bool isParamAPointer(unsigned ParamIdx) const;

  bool isParamImmArg(unsigned ParamIdx) const;

  CodeGenIntrinsic(Record *R, std::vector<Record *> DefaultProperties);
};

class CodeGenIntrinsicTable {
  std::vector<CodeGenIntrinsic> Intrinsics;

public:
  struct TargetSet {
    std::string Name;
    size_t Offset;
    size_t Count;
  };
  std::vector<TargetSet> Targets;

  explicit CodeGenIntrinsicTable(const RecordKeeper &RC);
  CodeGenIntrinsicTable() = default;

  bool empty() const { return Intrinsics.empty(); }
  size_t size() const { return Intrinsics.size(); }
  CodeGenIntrinsic &operator[](size_t Pos) { return Intrinsics[Pos]; }
  const CodeGenIntrinsic &operator[](size_t Pos) const {
    return Intrinsics[Pos];
  }
};
}

#endif
