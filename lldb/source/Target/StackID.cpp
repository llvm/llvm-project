//===-- StackID.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/StackID.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Stream.h"

using namespace lldb_private;

bool StackID::IsCFAOnStack(Process &process) const {
  if (m_cfa_on_stack == eLazyBoolCalculate) {
    // Conservatively assume stack memory
    m_cfa_on_stack = eLazyBoolYes;
    if (m_cfa != LLDB_INVALID_ADDRESS) {
      MemoryRegionInfo mem_info;
      if (process.GetMemoryRegionInfo(m_cfa, mem_info).Success())
        if (mem_info.IsStackMemory() == MemoryRegionInfo::eNo)
          m_cfa_on_stack = eLazyBoolNo;
    }
  }
  return m_cfa_on_stack == eLazyBoolYes;
}

void StackID::Dump(Stream *s) {
  s->Printf("StackID (pc = 0x%16.16" PRIx64 ", cfa = 0x%16.16" PRIx64
            ", cfa_on_stack = %d, symbol_scope = %p",
            m_pc, m_cfa, m_cfa_on_stack, static_cast<void *>(m_symbol_scope));
  if (m_symbol_scope) {
    SymbolContext sc;

    m_symbol_scope->CalculateSymbolContext(&sc);
    if (sc.block)
      s->Printf(" (Block {0x%8.8" PRIx64 "})", sc.block->GetID());
    else if (sc.symbol)
      s->Printf(" (Symbol{0x%8.8x})", sc.symbol->GetID());
  }
  s->PutCString(") ");
}

bool lldb_private::operator==(const StackID &lhs, const StackID &rhs) {
  if (lhs.GetCallFrameAddress() != rhs.GetCallFrameAddress())
    return false;

  SymbolContextScope *lhs_scope = lhs.GetSymbolContextScope();
  SymbolContextScope *rhs_scope = rhs.GetSymbolContextScope();

  // Only compare the PC values if both symbol context scopes are nullptr
  if (lhs_scope == nullptr && rhs_scope == nullptr)
    return lhs.GetPC() == rhs.GetPC();

  return lhs_scope == rhs_scope;
}

bool lldb_private::operator!=(const StackID &lhs, const StackID &rhs) {
  if (lhs.GetCallFrameAddress() != rhs.GetCallFrameAddress())
    return true;

  SymbolContextScope *lhs_scope = lhs.GetSymbolContextScope();
  SymbolContextScope *rhs_scope = rhs.GetSymbolContextScope();

  if (lhs_scope == nullptr && rhs_scope == nullptr)
    return lhs.GetPC() != rhs.GetPC();

  return lhs_scope != rhs_scope;
}

// BEGIN SWIFT
/// Given two async contexts, source and maybe_parent, chase continuation
/// pointers to check if maybe_parent can be reached from source. The search
/// stops when it hits the end of the chain (parent_ctx == 0) or a safety limit
/// in case of an invalid continuation chain.
static llvm::Expected<bool> IsReachableParent(lldb::addr_t source,
                                              lldb::addr_t maybe_parent,
                                              Process &process) {
  auto max_num_frames = 512;
  for (lldb::addr_t parent_ctx = source; parent_ctx && max_num_frames;
       max_num_frames--) {
    Status error;
    lldb::addr_t old_parent_ctx = parent_ctx;
    // The continuation's context is the first field of an async context.
    parent_ctx = process.ReadPointerFromMemory(old_parent_ctx, error);
    if (error.Fail())
      return llvm::createStringError(llvm::formatv(
          "Failed to read parent async context of: {0:x}. Error: {1}",
          old_parent_ctx, error.AsCString()));
    if (parent_ctx == maybe_parent)
      return true;
  }
  if (max_num_frames == 0)
    return llvm::createStringError(
        llvm::formatv("Failed to read continuation chain from {0:x} to "
                      "possible parent {1:x}. Reached limit of frames.",
                      source, maybe_parent));
  return false;
}

enum class HeapCFAComparisonResult { Younger, Older, NoOpinion };
/// If at least one of the stack IDs (lhs, rhs) is a heap CFA, perform the
/// swift-specific async frame comparison. Otherwise, returns NoOpinion.
static HeapCFAComparisonResult
CompareHeapCFAs(const StackID &lhs, const StackID &rhs, Process &process) {
  const bool lhs_cfa_on_stack = lhs.IsCFAOnStack(process);
  const bool rhs_cfa_on_stack = rhs.IsCFAOnStack(process);
  if (lhs_cfa_on_stack && rhs_cfa_on_stack)
    return HeapCFAComparisonResult::NoOpinion;

  // If one of the frames has a CFA on the stack and the other doesn't, we are
  // at the boundary between an asynchronous and a synchronous function.
  // Synchronous functions cannot call asynchronous functions, therefore the
  // synchronous frame is always younger.
  if (lhs_cfa_on_stack && !rhs_cfa_on_stack)
    return HeapCFAComparisonResult::Younger;
  if (!lhs_cfa_on_stack && rhs_cfa_on_stack)
    return HeapCFAComparisonResult::Older;

  const lldb::addr_t lhs_cfa = lhs.GetCallFrameAddress();
  const lldb::addr_t rhs_cfa = rhs.GetCallFrameAddress();
  // If the cfas are the same, fallback to the usual scope comparison.
  if (lhs_cfa == rhs_cfa)
    return HeapCFAComparisonResult::NoOpinion;

  // Both CFAs are on the heap and they are distinct.
  // LHS is younger if and only if its continuation async context is (directly
  // or indirectly) RHS.
  llvm::Expected<bool> lhs_younger =
      IsReachableParent(lhs_cfa, rhs_cfa, process);
  if (auto E = lhs_younger.takeError())
    LLDB_LOG_ERROR(GetLog(LLDBLog::Unwind), std::move(E), "{0}");
  else if (*lhs_younger)
    return HeapCFAComparisonResult::Younger;
  return HeapCFAComparisonResult::NoOpinion;
}
// END SWIFT

bool StackID::IsYounger(const StackID &lhs, const StackID &rhs,
                        Process &process) {
  // BEGIN SWIFT
  switch (CompareHeapCFAs(lhs, rhs, process)) {
  case HeapCFAComparisonResult::Younger:
    return true;
  case HeapCFAComparisonResult::Older:
    return false;
  case HeapCFAComparisonResult::NoOpinion:
    break;
  }
  // END SWIFT

  const lldb::addr_t lhs_cfa = lhs.GetCallFrameAddress();
  const lldb::addr_t rhs_cfa = rhs.GetCallFrameAddress();

  // FIXME: We are assuming that the stacks grow downward in memory.  That's not
  // necessary, but true on
  // all the machines we care about at present.  If this changes, we'll have to
  // deal with that.  The ABI is the agent who knows this ordering, but the
  // StackID has no access to the ABI. The most straightforward way to handle
  // this is to add a "m_grows_downward" bool to the StackID, and set it in the
  // constructor. But I'm not going to waste a bool per StackID on this till we
  // need it.

  if (lhs_cfa != rhs_cfa)
    return lhs_cfa < rhs_cfa;

  SymbolContextScope *lhs_scope = lhs.GetSymbolContextScope();
  SymbolContextScope *rhs_scope = rhs.GetSymbolContextScope();

  if (lhs_scope != nullptr && rhs_scope != nullptr) {
    // Same exact scope, lhs is not less than (younger than rhs)
    if (lhs_scope == rhs_scope)
      return false;

    SymbolContext lhs_sc;
    SymbolContext rhs_sc;
    lhs_scope->CalculateSymbolContext(&lhs_sc);
    rhs_scope->CalculateSymbolContext(&rhs_sc);

    // Items with the same function can only be compared
    if (lhs_sc.function == rhs_sc.function && lhs_sc.function != nullptr &&
        lhs_sc.block != nullptr && rhs_sc.function != nullptr &&
        rhs_sc.block != nullptr) {
      return rhs_sc.block->Contains(lhs_sc.block);
    }
  }
  return false;
}
