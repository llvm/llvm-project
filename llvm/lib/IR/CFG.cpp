//===- CFG.cpp --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/CFG.h"

#include "llvm/IR/ModuleSlotTracker.h"

using namespace llvm;

IrCfgTraits::Printer::Printer(const IrCfgTraits &) {}
IrCfgTraits::Printer::~Printer() {}

void IrCfgTraits::Printer::printValue(raw_ostream &out, ValueRef value) const {
  if (!m_moduleSlotTracker) {
    const Function *function = nullptr;

    if (auto *instruction = dyn_cast<Instruction>(value)) {
      function = instruction->getParent()->getParent();
    } else if (auto *argument = dyn_cast<Argument>(value)) {
      function = argument->getParent();
    }

    if (function)
      ensureModuleSlotTracker(*function);
  }

  if (m_moduleSlotTracker) {
    value->print(out, *m_moduleSlotTracker, true);
  } else {
    value->print(out, true);
  }
}

void IrCfgTraits::Printer::printBlockName(raw_ostream &out,
                                          BlockRef block) const {
  if (block->hasName()) {
    out << block->getName();
  } else {
    ensureModuleSlotTracker(*block->getParent());
    out << m_moduleSlotTracker->getLocalSlot(block);
  }
}

void IrCfgTraits::Printer::ensureModuleSlotTracker(
    const Function &function) const {
  if (!m_moduleSlotTracker) {
    m_moduleSlotTracker =
        std::make_unique<ModuleSlotTracker>(function.getParent(), false);
    m_moduleSlotTracker->incorporateFunction(function);
  }
}
