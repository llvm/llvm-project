//===- CodeGenIntrinsic.h - Intrinsic Class Wrapper ------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a wrapper class for the 'Intrinsic' TableGen class.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_INTRINSIC_H
#define CODEGEN_INTRINSIC_H

#include <string>
#include <vector>
#include "llvm/CodeGen/ValueTypes.h"

namespace llvm {
  class Record;
  class RecordKeeper;
  class CodeGenTarget;

  struct CodeGenIntrinsic {
    Record *TheDef;            // The actual record defining this intrinsic.
    std::string Name;          // The name of the LLVM function "llvm.bswap.i32"
    std::string EnumName;      // The name of the enum "bswap_i32"
    std::string GCCBuiltinName;// Name of the corresponding GCC builtin, or "".
    std::string TargetPrefix;  // Target prefix, e.g. "ppc" for t-s intrinsics.

    /// IntrinsicSignature - This structure holds the return values and
    /// parameter values of an intrinsic. If the number of return values is > 1,
    /// then the intrinsic implicitly returns a first-class aggregate. The
    /// numbering of the types starts at 0 with the first return value and
    /// continues from there through the parameter list. This is useful for
    /// "matching" types.
    struct IntrinsicSignature {
      /// RetVTs - The MVT::SimpleValueType for each return type. Note that this
      /// list is only populated when in the context of a target .td file. When
      /// building Intrinsics.td, this isn't available, because we don't know
      /// the target pointer size.
      std::vector<MVT::SimpleValueType> RetVTs;

      /// RetTypeDefs - The records for each return type.
      std::vector<Record*> RetTypeDefs;

      /// ParamVTs - The MVT::SimpleValueType for each parameter type. Note that
      /// this list is only populated when in the context of a target .td file.
      /// When building Intrinsics.td, this isn't available, because we don't
      /// know the target pointer size.
      std::vector<MVT::SimpleValueType> ParamVTs;

      /// ParamTypeDefs - The records for each parameter type.
      std::vector<Record*> ParamTypeDefs;
    };

    IntrinsicSignature IS;

    // Memory mod/ref behavior of this intrinsic.
    enum {
      NoMem, ReadArgMem, ReadMem, ReadWriteArgMem, ReadWriteMem
    } ModRef;

    /// This is set to true if the intrinsic is overloaded by its argument
    /// types.
    bool isOverloaded;

    /// isCommutative - True if the intrinsic is commutative.
    bool isCommutative;

    /// canThrow - True if the intrinsic can throw.
    bool canThrow;

    /// isNoReturn - True if the intrinsic is no-return.
    bool isNoReturn;

    enum ArgAttribute {
      NoCapture
    };
    std::vector<std::pair<unsigned, ArgAttribute> > ArgumentAttributes;

    CodeGenIntrinsic(Record *R);
  };

  /// LoadIntrinsics - Read all of the intrinsics defined in the specified
  /// .td file.
  std::vector<CodeGenIntrinsic> LoadIntrinsics(const RecordKeeper &RC,
                                               bool TargetOnly);
}

#endif
