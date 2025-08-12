//===- BasicPtxBuilderInterface.td - PTX builder interface -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to build PTX (Parallel Thread Execution) from NVVM Ops
// automatically. It is used by NVVM to LLVM pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/BasicPtxBuilderInterface.h"

#define DEBUG_TYPE "ptx-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

//===----------------------------------------------------------------------===//
// BasicPtxBuilderInterface
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/BasicPtxBuilderInterface.cpp.inc"

using namespace mlir;
using namespace NVVM;

static constexpr int64_t kSharedMemorySpace = 3;

static char getRegisterType(Type type) {
  if (type.isInteger(1))
    return 'b';
  if (type.isInteger(16))
    return 'h';
  if (type.isInteger(32))
    return 'r';
  if (type.isInteger(64))
    return 'l';
  if (type.isF32())
    return 'f';
  if (type.isF64())
    return 'd';
  if (auto ptr = dyn_cast<LLVM::LLVMPointerType>(type)) {
    // Shared address spaces is addressed with 32-bit pointers.
    if (ptr.getAddressSpace() == kSharedMemorySpace) {
      return 'r';
    }
    return 'l';
  }
  // register type for struct is not supported.
  llvm_unreachable("The register type could not deduced from MLIR type");
  return '?';
}

static char getRegisterType(Value v) {
  if (v.getDefiningOp<LLVM::ConstantOp>())
    return 'n';
  return getRegisterType(v.getType());
}

void PtxBuilder::insertValue(Value v, PTXRegisterMod itype) {
  LLVM_DEBUG(DBGS() << v << "\t Modifier : " << &itype << "\n");
  auto getModifier = [&]() -> const char * {
    if (itype == PTXRegisterMod::ReadWrite) {
      assert(false && "Read-Write modifier is not supported. Try setting the "
                      "same value as Write and Read separately.");
      return "+";
    }
    if (itype == PTXRegisterMod::Write) {
      return "=";
    }
    return "";
  };
  auto addValue = [&](Value v) {
    if (itype == PTXRegisterMod::Read) {
      ptxOperands.push_back(v);
      return;
    }
    if (itype == PTXRegisterMod::ReadWrite)
      ptxOperands.push_back(v);
    hasResult = true;
  };

  llvm::raw_string_ostream ss(registerConstraints);
  // Handle Structs
  if (auto stype = dyn_cast<LLVM::LLVMStructType>(v.getType())) {
    if (itype == PTXRegisterMod::Write) {
      addValue(v);
    }
    for (auto [idx, t] : llvm::enumerate(stype.getBody())) {
      if (itype != PTXRegisterMod::Write) {
        Value extractValue = LLVM::ExtractValueOp::create(
            rewriter, interfaceOp->getLoc(), v, idx);
        addValue(extractValue);
      }
      if (itype == PTXRegisterMod::ReadWrite) {
        ss << idx << ",";
      } else {
        ss << getModifier() << getRegisterType(t) << ",";
      }
    }
    return;
  }
  // Handle Scalars
  addValue(v);
  ss << getModifier() << getRegisterType(v) << ",";
}

LLVM::InlineAsmOp PtxBuilder::build() {
  auto asmDialectAttr = LLVM::AsmDialectAttr::get(interfaceOp->getContext(),
                                                  LLVM::AsmDialect::AD_ATT);

  auto resultTypes = interfaceOp->getResultTypes();

  // Remove the last comma from the constraints string.
  if (!registerConstraints.empty() &&
      registerConstraints[registerConstraints.size() - 1] == ',')
    registerConstraints.pop_back();

  std::string ptxInstruction = interfaceOp.getPtx();

  // Add the predicate to the asm string.
  if (interfaceOp.getPredicate().has_value() &&
      interfaceOp.getPredicate().value()) {
    std::string predicateStr = "@%";
    predicateStr += std::to_string((ptxOperands.size() - 1));
    ptxInstruction = predicateStr + " " + ptxInstruction;
  }

  // Tablegen doesn't accept $, so we use %, but inline assembly uses $.
  // Replace all % with $
  llvm::replace(ptxInstruction, '%', '$');

  return LLVM::InlineAsmOp::create(
      rewriter, interfaceOp->getLoc(),
      /*result types=*/resultTypes,
      /*operands=*/ptxOperands,
      /*asm_string=*/llvm::StringRef(ptxInstruction),
      /*constraints=*/registerConstraints.data(),
      /*has_side_effects=*/interfaceOp.hasSideEffect(),
      /*is_align_stack=*/false, LLVM::TailCallKind::None,
      /*asm_dialect=*/asmDialectAttr,
      /*operand_attrs=*/ArrayAttr());
}

void PtxBuilder::buildAndReplaceOp() {
  LLVM::InlineAsmOp inlineAsmOp = build();
  LLVM_DEBUG(DBGS() << "\n Generated PTX \n\t" << inlineAsmOp << "\n");
  if (inlineAsmOp->getNumResults() == interfaceOp->getNumResults()) {
    rewriter.replaceOp(interfaceOp, inlineAsmOp);
  } else {
    rewriter.eraseOp(interfaceOp);
  }
}
