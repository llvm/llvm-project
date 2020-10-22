//===- MachineCfgTraits.h - Traits for Machine IR CFGs ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines the MachineCfgTraits to allow generic CFG algorithms to
/// operate on MachineIR in SSA form.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINECFGTRAITS_H
#define LLVM_CODEGEN_MACHINECFGTRAITS_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CfgTraits.h"

namespace llvm {

class MachineCfgTraitsBase : public CfgTraitsBase {
public:
  using ParentType = MachineFunction;
  using BlockRef = MachineBasicBlock *;
  using ValueRef = Register;

  static CfgBlockRef wrapRef(BlockRef block) {
    return makeOpaque<CfgBlockRefTag>(block);
  }
  static CfgValueRef wrapRef(ValueRef value) {
    // Physical registers are unsupported by design.
    assert(!value.isValid() || value.isVirtual());
    uintptr_t wrapped = value.id();
    assert((wrapped != 0) == value.isValid());

    // Guard against producing values reserved for DenseMap markers. This is de
    // facto impossible, because it'd require 2^31 virtual registers to be in
    // use on a 32-bit architecture.
    assert(wrapped != (uintptr_t)-1 && wrapped != (uintptr_t)-2);

    return makeOpaque<CfgValueRefTag>(reinterpret_cast<void *>(wrapped));
  }
  static BlockRef unwrapRef(CfgBlockRef block) {
    return static_cast<BlockRef>(getOpaque(block));
  }
  static ValueRef unwrapRef(CfgValueRef value) {
    uintptr_t wrapped = reinterpret_cast<uintptr_t>(getOpaque(value));
    return Register(wrapped);
  }
};

/// \brief CFG traits for Machine IR in SSA form.
class MachineCfgTraits
    : public CfgTraits<MachineCfgTraitsBase, MachineCfgTraits> {
private:
  MachineRegisterInfo *m_regInfo;

public:
  explicit MachineCfgTraits(MachineFunction *parent)
      : m_regInfo(&parent->getRegInfo()) {}

  static MachineFunction *getBlockParent(MachineBasicBlock *block) {
    return block->getParent();
  }

  struct const_blockref_iterator
      : iterator_adaptor_base<const_blockref_iterator,
                              MachineFunction::iterator> {
    using Base = iterator_adaptor_base<const_blockref_iterator,
                                       MachineFunction::iterator>;

    const_blockref_iterator() = default;

    explicit const_blockref_iterator(MachineFunction::iterator i) : Base(i) {}

    MachineBasicBlock *operator*() const { return &Base::operator*(); }
  };

  static iterator_range<const_blockref_iterator>
  blocks(MachineFunction *function) {
    return {const_blockref_iterator(function->begin()),
            const_blockref_iterator(function->end())};
  }

  static auto predecessors(MachineBasicBlock *block) {
    return block->predecessors();
  }
  static auto successors(MachineBasicBlock *block) {
    return block->successors();
  }

  /// Get the defining block of a value.
  MachineBasicBlock *getValueDefBlock(ValueRef value) const {
    if (!value)
      return nullptr;
    return m_regInfo->getVRegDef(value)->getParent();
  }

  struct blockdef_iterator
      : iterator_facade_base<blockdef_iterator, std::forward_iterator_tag,
                             Register> {
  private:
    MachineBasicBlock::instr_iterator m_instr;
    MachineInstr::mop_iterator m_def;

  public:
    blockdef_iterator() = default;

    explicit blockdef_iterator(MachineBasicBlock &block)
        : m_instr(block.instr_begin()) {
      if (m_instr != block.end())
        m_def = m_instr->defs().begin();
    }
    blockdef_iterator(MachineBasicBlock &block, bool)
        : m_instr(block.instr_end()), m_def() {}

    bool operator==(const blockdef_iterator &rhs) const {
      return m_instr == rhs.m_instr && m_def == rhs.m_def;
    }

    Register operator*() const {
      assert(m_def->isReg() && !m_def->getSubReg() && m_def->isDef());
      return m_def->getReg();
    }

    blockdef_iterator &operator++() {
      ++m_def;

      while (m_def == m_instr->defs().end()) {
        ++m_instr;
        if (m_instr.isEnd()) {
          m_def = {};
          return *this;
        }

        m_def = m_instr->defs().begin();
      }

      return *this;
    }
  };

  static auto blockdefs(MachineBasicBlock *block) {
    return llvm::make_range(blockdef_iterator(*block),
                            blockdef_iterator(*block, true));
  }

  struct Printer {
    explicit Printer(const MachineCfgTraits &traits)
        : m_regInfo(traits.m_regInfo) {}

    void printBlockName(raw_ostream &out, MachineBasicBlock *block) const;
    void printValue(raw_ostream &out, Register value) const;

  private:
    MachineRegisterInfo *m_regInfo;
  };
};

template <> struct CfgTraitsFor<MachineBasicBlock> {
  using CfgTraits = MachineCfgTraits;
};

} // namespace llvm

#endif // LLVM_CODEGEN_MACHINECFGTRAITS_H
