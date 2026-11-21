//===--- NVVMRequiresSMTraits.h - NVVM Requires SM Traits -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines op traits for the NVVM Dialect in MLIR
//
//===----------------------------------------------------------------------===//

#ifndef NVVM_DIALECT_NVVM_IR_NVVMREQUIRESSMTRAITS_H_
#define NVVM_DIALECT_NVVM_IR_NVVMREQUIRESSMTRAITS_H_

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/Bitset.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/NVPTXTargetParser.h"

namespace mlir {

namespace NVVM {
using NVVMArchSet = llvm::Bitset<llvm::NVPTX::GK_LAST + 1>;

/// Like llvm::NVPTX::parseArch, but maps the old sm_101* names to sm_110*.
llvm::NVPTX::GPUKind parseTargetArch(StringRef chip);
} // namespace NVVM
} // namespace mlir

#include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.h.inc"

namespace mlir {

namespace OpTrait {
template <llvm::NVPTX::GPUKind... SupportedArchs>
class NVVMRequiresSM {
public:
  template <typename ConcreteOp>
  class Impl
      : public OpTrait::TraitBase<ConcreteOp,
                                  NVVMRequiresSM<SupportedArchs...>::Impl>,
        public mlir::NVVM::RequiresSMInterface::Trait<ConcreteOp> {
  public:
    NVVM::NVVMArchSet getSupportedArchs() const {
      return NVVM::NVVMArchSet{SupportedArchs...};
    }
  };
};
} // namespace OpTrait
} // namespace mlir
#endif // NVVM_DIALECT_NVVM_IR_NVVMREQUIRESSMTRAITS_H_
