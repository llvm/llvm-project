//===- X86InstrRelaxTables.cpp - X86 Instruction Relaxation Tables -*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 instruction relaxation tables.
//
//===----------------------------------------------------------------------===//

#include "X86InstrRelaxTables.h"
#include "X86InstrInfo.h"
#include "llvm/ADT/STLExtras.h"
#include <atomic>

using namespace llvm;

// These tables are sorted by their ShortOp value allowing them to be binary
// searched at runtime without the need for additional storage. The enum values
// are currently emitted in X86GenInstrInfo.inc in alphabetical order. Which
// makes sorting these tables a simple matter of alphabetizing the table.
static const X86InstrRelaxTableEntry InstrRelaxTable[] = {
  // ADC
  { X86::ADC16mi8,   X86::ADC16mi     },
  { X86::ADC16ri8,   X86::ADC16ri     },
  { X86::ADC32mi8,   X86::ADC32mi     },
  { X86::ADC32ri8,   X86::ADC32ri     },
  { X86::ADC64mi8,   X86::ADC64mi32   },
  { X86::ADC64ri8,   X86::ADC64ri32   },
  // ADD
  { X86::ADD16mi8,   X86::ADD16mi     },
  { X86::ADD16ri8,   X86::ADD16ri     },
  { X86::ADD32mi8,   X86::ADD32mi     },
  { X86::ADD32ri8,   X86::ADD32ri     },
  { X86::ADD64mi8,   X86::ADD64mi32   },
  { X86::ADD64ri8,   X86::ADD64ri32   },
  // AND
  { X86::AND16mi8,   X86::AND16mi     },
  { X86::AND16ri8,   X86::AND16ri     },
  { X86::AND32mi8,   X86::AND32mi     },
  { X86::AND32ri8,   X86::AND32ri     },
  { X86::AND64mi8,   X86::AND64mi32   },
  { X86::AND64ri8,   X86::AND64ri32   },
  // CMP
  { X86::CMP16mi8,   X86::CMP16mi     },
  { X86::CMP16ri8,   X86::CMP16ri     },
  { X86::CMP32mi8,   X86::CMP32mi     },
  { X86::CMP32ri8,   X86::CMP32ri     },
  { X86::CMP64mi8,   X86::CMP64mi32   },
  { X86::CMP64ri8,   X86::CMP64ri32   },
  // IMUL
  { X86::IMUL16rmi8, X86::IMUL16rmi   },
  { X86::IMUL16rri8, X86::IMUL16rri   },
  { X86::IMUL32rmi8, X86::IMUL32rmi   },
  { X86::IMUL32rri8, X86::IMUL32rri   },
  { X86::IMUL64rmi8, X86::IMUL64rmi32 },
  { X86::IMUL64rri8, X86::IMUL64rri32 },
  // OR
  { X86::OR16mi8,    X86::OR16mi      },
  { X86::OR16ri8,    X86::OR16ri      },
  { X86::OR32mi8,    X86::OR32mi      },
  { X86::OR32ri8,    X86::OR32ri      },
  { X86::OR64mi8,    X86::OR64mi32    },
  { X86::OR64ri8,    X86::OR64ri32    },
  // PUSH
  { X86::PUSH16i8,   X86::PUSHi16     },
  { X86::PUSH32i8,   X86::PUSHi32     },
  { X86::PUSH64i8,   X86::PUSH64i32   },
  // SBB
  { X86::SBB16mi8,   X86::SBB16mi     },
  { X86::SBB16ri8,   X86::SBB16ri     },
  { X86::SBB32mi8,   X86::SBB32mi     },
  { X86::SBB32ri8,   X86::SBB32ri     },
  { X86::SBB64mi8,   X86::SBB64mi32   },
  { X86::SBB64ri8,   X86::SBB64ri32   },
  // SUB
  { X86::SUB16mi8,   X86::SUB16mi     },
  { X86::SUB16ri8,   X86::SUB16ri     },
  { X86::SUB32mi8,   X86::SUB32mi     },
  { X86::SUB32ri8,   X86::SUB32ri     },
  { X86::SUB64mi8,   X86::SUB64mi32   },
  { X86::SUB64ri8,   X86::SUB64ri32   },
  // XOR
  { X86::XOR16mi8,   X86::XOR16mi     },
  { X86::XOR16ri8,   X86::XOR16ri     },
  { X86::XOR32mi8,   X86::XOR32mi     },
  { X86::XOR32ri8,   X86::XOR32ri     },
  { X86::XOR64mi8,   X86::XOR64mi32   },
  { X86::XOR64ri8,   X86::XOR64ri32   },
};

static const X86InstrRelaxTableEntry *
lookupRelaxTableImpl(ArrayRef<X86InstrRelaxTableEntry> Table,
                     unsigned ShortOp) {
#ifndef NDEBUG
  // Make sure the tables are sorted.
  static std::atomic<bool> RelaxTableChecked(false);
  if (!RelaxTableChecked.load(std::memory_order_relaxed)) {
    assert(llvm::is_sorted(InstrRelaxTable) &&
           std::adjacent_find(std::begin(InstrRelaxTable),
                              std::end(InstrRelaxTable)) ==
               std::end(InstrRelaxTable) &&
           "InstrRelaxTable is not sorted and unique!");
    RelaxTableChecked.store(true, std::memory_order_relaxed);
  }
#endif

  const X86InstrRelaxTableEntry *Data = llvm::lower_bound(Table, ShortOp);
  if (Data != Table.end() && Data->KeyOp == ShortOp)
    return Data;
  return nullptr;
}

const X86InstrRelaxTableEntry *llvm::lookupRelaxTable(unsigned ShortOp) {
  return lookupRelaxTableImpl(InstrRelaxTable, ShortOp);
}

namespace {

// This class stores the short form tables. It is instantiated as a
// function scope static variable to lazily init the short form table.
struct X86ShortFormTable {
  // Stores relaxation table entries sorted by relaxed form opcode.
  SmallVector<X86InstrRelaxTableEntry, 0> Table;

  X86ShortFormTable() {
    for (const X86InstrRelaxTableEntry &Entry : InstrRelaxTable)
      Table.push_back({Entry.DstOp, Entry.KeyOp});

    llvm::sort(Table);

    // Now that it's sorted, ensure its unique.
    assert(std::adjacent_find(Table.begin(), Table.end()) == Table.end() &&
           "Short form table is not unique!");
  }
};
} // namespace

const X86InstrRelaxTableEntry *llvm::lookupShortTable(unsigned RelaxOp) {
  static X86ShortFormTable ShortTable;
  auto &Table = ShortTable.Table;
  auto I = llvm::lower_bound(Table, RelaxOp);
  if (I != Table.end() && I->KeyOp == RelaxOp)
    return &*I;
  return nullptr;
}

namespace llvm {

/// Get the short instruction opcode for a given relaxed opcode.
unsigned X86::getShortOpcodeArith(unsigned RelaxOp) {
  if (const X86InstrRelaxTableEntry *I = lookupShortTable(RelaxOp))
    return I->DstOp;
  return RelaxOp;
}

/// Get the relaxed instruction opcode for a given short opcode.
unsigned X86::getRelaxedOpcodeArith(unsigned ShortOp) {
  if (const X86InstrRelaxTableEntry *I = lookupRelaxTable(ShortOp))
    return I->DstOp;
  return ShortOp;
}
} // namespace llvm
