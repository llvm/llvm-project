//===-- FuncUnwinders.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/FuncUnwinders.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Symbol/ArmUnwindInfo.h"
#include "lldb/Symbol/CallFrameInfo.h"
#include "lldb/Symbol/CompactUnwindInfo.h"
#include "lldb/Symbol/DWARFCallFrameInfo.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Symbol/UnwindTable.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/RegisterNumber.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/UnwindAssembly.h"

#include <memory>

using namespace lldb;
using namespace lldb_private;

FuncUnwinders::FuncUnwinders(UnwindTable &unwind_table, Address addr,
                             AddressRanges ranges)
    : m_unwind_table(unwind_table), m_addr(std::move(addr)),
      m_ranges(std::move(ranges)), m_tried_unwind_plan_assembly(false),
      m_tried_unwind_plan_eh_frame(false),
      m_tried_unwind_plan_object_file(false),
      m_tried_unwind_plan_debug_frame(false),
      m_tried_unwind_plan_object_file_augmented(false),
      m_tried_unwind_plan_eh_frame_augmented(false),
      m_tried_unwind_plan_debug_frame_augmented(false),
      m_tried_unwind_plan_compact_unwind(false),
      m_tried_unwind_plan_arm_unwind(false),
      m_tried_unwind_plan_symbol_file(false), m_tried_unwind_fast(false),
      m_tried_unwind_arch_default(false),
      m_tried_unwind_arch_default_at_func_entry(false),
      m_first_non_prologue_insn() {}

/// destructor

FuncUnwinders::~FuncUnwinders() = default;

std::shared_ptr<const UnwindPlan>
FuncUnwinders::GetUnwindPlanAtCallSite(Target &target, Thread &thread) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);

  if (std::shared_ptr<const UnwindPlan> plan_sp =
          GetObjectFileUnwindPlan(target))
    return plan_sp;
  if (std::shared_ptr<const UnwindPlan> plan_sp =
          GetSymbolFileUnwindPlan(thread))
    return plan_sp;
  if (std::shared_ptr<const UnwindPlan> plan_sp =
          GetDebugFrameUnwindPlan(target))
    return plan_sp;
  if (std::shared_ptr<const UnwindPlan> plan_sp = GetEHFrameUnwindPlan(target))
    return plan_sp;
  if (std::shared_ptr<const UnwindPlan> plan_sp =
          GetCompactUnwindUnwindPlan(target))
    return plan_sp;
  if (std::shared_ptr<const UnwindPlan> plan_sp =
          GetArmUnwindUnwindPlan(target))
    return plan_sp;

  return nullptr;
}

std::shared_ptr<const UnwindPlan>
FuncUnwinders::GetCompactUnwindUnwindPlan(Target &target) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (m_unwind_plan_compact_unwind.size() > 0)
    return m_unwind_plan_compact_unwind[0]; // FIXME support multiple compact
                                            // unwind plans for one func
  if (m_tried_unwind_plan_compact_unwind)
    return nullptr;

  m_tried_unwind_plan_compact_unwind = true;
  // Only continuous functions are supported.
  if (m_ranges.size() == 1) {
    Address current_pc(m_ranges[0].GetBaseAddress());
    CompactUnwindInfo *compact_unwind = m_unwind_table.GetCompactUnwindInfo();
    if (compact_unwind) {
      auto unwind_plan_sp =
          std::make_shared<UnwindPlan>(lldb::eRegisterKindGeneric);
      if (compact_unwind->GetUnwindPlan(target, current_pc, *unwind_plan_sp)) {
        m_unwind_plan_compact_unwind.push_back(unwind_plan_sp);
        return m_unwind_plan_compact_unwind[0]; // FIXME support multiple
                                                // compact unwind plans for one
                                                // func
      }
    }
  }
  return nullptr;
}

std::shared_ptr<const UnwindPlan>
FuncUnwinders::GetObjectFileUnwindPlan(Target &target) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (m_unwind_plan_object_file_sp.get() ||
      m_tried_unwind_plan_object_file)
    return m_unwind_plan_object_file_sp;

  m_tried_unwind_plan_object_file = true;
  if (CallFrameInfo *object_file_frame =
          m_unwind_table.GetObjectFileUnwindInfo())
    m_unwind_plan_object_file_sp =
        object_file_frame->GetUnwindPlan(m_ranges, m_addr);
  return m_unwind_plan_object_file_sp;
}

std::shared_ptr<const UnwindPlan>
FuncUnwinders::GetEHFrameUnwindPlan(Target &target) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (m_unwind_plan_eh_frame_sp.get() || m_tried_unwind_plan_eh_frame)
    return m_unwind_plan_eh_frame_sp;

  m_tried_unwind_plan_eh_frame = true;
  if (m_addr.IsValid()) {
    if (DWARFCallFrameInfo *eh_frame = m_unwind_table.GetEHFrameInfo())
      m_unwind_plan_eh_frame_sp = eh_frame->GetUnwindPlan(m_ranges, m_addr);
  }
  return m_unwind_plan_eh_frame_sp;
}

std::shared_ptr<const UnwindPlan>
FuncUnwinders::GetDebugFrameUnwindPlan(Target &target) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (m_unwind_plan_debug_frame_sp || m_tried_unwind_plan_debug_frame)
    return m_unwind_plan_debug_frame_sp;

  m_tried_unwind_plan_debug_frame = true;
  if (!m_ranges.empty()) {
    if (DWARFCallFrameInfo *debug_frame = m_unwind_table.GetDebugFrameInfo())
      m_unwind_plan_debug_frame_sp =
          debug_frame->GetUnwindPlan(m_ranges, m_addr);
  }
  return m_unwind_plan_debug_frame_sp;
}

std::shared_ptr<const UnwindPlan>
FuncUnwinders::GetArmUnwindUnwindPlan(Target &target) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (m_unwind_plan_arm_unwind_sp.get() || m_tried_unwind_plan_arm_unwind)
    return m_unwind_plan_arm_unwind_sp;

  m_tried_unwind_plan_arm_unwind = true;
  // Only continuous functions are supported.
  if (m_ranges.size() == 1) {
    Address current_pc = m_ranges[0].GetBaseAddress();
    ArmUnwindInfo *arm_unwind_info = m_unwind_table.GetArmUnwindInfo();
    if (arm_unwind_info) {
      auto plan_sp = std::make_shared<UnwindPlan>(lldb::eRegisterKindGeneric);
      if (arm_unwind_info->GetUnwindPlan(target, current_pc, *plan_sp))
        m_unwind_plan_arm_unwind_sp = std::move(plan_sp);
    }
  }
  return m_unwind_plan_arm_unwind_sp;
}

namespace {
class RegisterContextToInfo: public SymbolFile::RegisterInfoResolver {
public:
  RegisterContextToInfo(RegisterContext &ctx) : m_ctx(ctx) {}

  const RegisterInfo *ResolveName(llvm::StringRef name) const override {
    return m_ctx.GetRegisterInfoByName(name);
  }
  const RegisterInfo *ResolveNumber(lldb::RegisterKind kind,
                                    uint32_t number) const override {
    return m_ctx.GetRegisterInfo(kind, number);
  }

private:
  RegisterContext &m_ctx;
};
} // namespace

std::shared_ptr<const UnwindPlan>
FuncUnwinders::GetSymbolFileUnwindPlan(Thread &thread) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (m_unwind_plan_symbol_file_sp.get() || m_tried_unwind_plan_symbol_file)
    return m_unwind_plan_symbol_file_sp;

  m_tried_unwind_plan_symbol_file = true;
  if (SymbolFile *symfile = m_unwind_table.GetSymbolFile();
      symfile && m_ranges.size() == 1) {
    m_unwind_plan_symbol_file_sp = symfile->GetUnwindPlan(
        m_ranges[0].GetBaseAddress(),
        RegisterContextToInfo(*thread.GetRegisterContext()));
  }
  return m_unwind_plan_symbol_file_sp;
}

std::shared_ptr<const UnwindPlan>
FuncUnwinders::GetObjectFileAugmentedUnwindPlan(Target &target,
                                                Thread &thread) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (m_unwind_plan_object_file_augmented_sp.get() ||
      m_tried_unwind_plan_object_file_augmented)
    return m_unwind_plan_object_file_augmented_sp;

  m_tried_unwind_plan_object_file_augmented = true;

  std::shared_ptr<const UnwindPlan> object_file_unwind_plan =
      GetObjectFileUnwindPlan(target);
  if (!object_file_unwind_plan)
    return m_unwind_plan_object_file_augmented_sp;

  // Augment the instructions with epilogue descriptions if necessary
  // so the UnwindPlan can be used at any instruction in the function.

  UnwindAssemblySP assembly_profiler_sp(GetUnwindAssemblyProfiler(target));
  // Only continuous functions are supported.
  if (assembly_profiler_sp && m_ranges.size() == 1) {
    auto plan_sp = std::make_shared<UnwindPlan>(*object_file_unwind_plan);

    if (assembly_profiler_sp->AugmentUnwindPlanFromCallSite(m_ranges[0], thread,
                                                            *plan_sp))
      m_unwind_plan_object_file_augmented_sp = std::move(plan_sp);
  }
  return m_unwind_plan_object_file_augmented_sp;
}

std::shared_ptr<const UnwindPlan>
FuncUnwinders::GetEHFrameAugmentedUnwindPlan(Target &target, Thread &thread) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (m_unwind_plan_eh_frame_augmented_sp.get() ||
      m_tried_unwind_plan_eh_frame_augmented)
    return m_unwind_plan_eh_frame_augmented_sp;

  // Only supported on x86 architectures where we get eh_frame from the
  // compiler that describes the prologue instructions perfectly, and sometimes
  // the epilogue instructions too.
  if (target.GetArchitecture().GetCore() != ArchSpec::eCore_x86_32_i386 &&
      target.GetArchitecture().GetCore() != ArchSpec::eCore_x86_64_x86_64 &&
      target.GetArchitecture().GetCore() != ArchSpec::eCore_x86_64_x86_64h) {
    m_tried_unwind_plan_eh_frame_augmented = true;
    return m_unwind_plan_eh_frame_augmented_sp;
  }

  m_tried_unwind_plan_eh_frame_augmented = true;

  std::shared_ptr<const UnwindPlan> eh_frame_plan =
      GetEHFrameUnwindPlan(target);
  if (!eh_frame_plan)
    return m_unwind_plan_eh_frame_augmented_sp;

  // Augment the eh_frame instructions with epilogue descriptions if necessary
  // so the UnwindPlan can be used at any instruction in the function.

  UnwindAssemblySP assembly_profiler_sp(GetUnwindAssemblyProfiler(target));
  // Only continuous functions are supported.
  if (assembly_profiler_sp && m_ranges.size() == 1) {
    auto plan_sp = std::make_shared<UnwindPlan>(*eh_frame_plan);
    if (assembly_profiler_sp->AugmentUnwindPlanFromCallSite(m_ranges[0], thread,
                                                            *plan_sp))
      m_unwind_plan_eh_frame_augmented_sp = std::move(plan_sp);
  }
  return m_unwind_plan_eh_frame_augmented_sp;
}

std::shared_ptr<const UnwindPlan>
FuncUnwinders::GetDebugFrameAugmentedUnwindPlan(Target &target,
                                                Thread &thread) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (m_unwind_plan_debug_frame_augmented_sp.get() ||
      m_tried_unwind_plan_debug_frame_augmented)
    return m_unwind_plan_debug_frame_augmented_sp;

  // Only supported on x86 architectures where we get debug_frame from the
  // compiler that describes the prologue instructions perfectly, and sometimes
  // the epilogue instructions too.
  if (target.GetArchitecture().GetCore() != ArchSpec::eCore_x86_32_i386 &&
      target.GetArchitecture().GetCore() != ArchSpec::eCore_x86_64_x86_64 &&
      target.GetArchitecture().GetCore() != ArchSpec::eCore_x86_64_x86_64h) {
    m_tried_unwind_plan_debug_frame_augmented = true;
    return m_unwind_plan_debug_frame_augmented_sp;
  }

  m_tried_unwind_plan_debug_frame_augmented = true;

  std::shared_ptr<const UnwindPlan> debug_frame_plan =
      GetDebugFrameUnwindPlan(target);
  if (!debug_frame_plan)
    return m_unwind_plan_debug_frame_augmented_sp;

  // Augment the debug_frame instructions with epilogue descriptions if
  // necessary so the UnwindPlan can be used at any instruction in the
  // function.

  UnwindAssemblySP assembly_profiler_sp(GetUnwindAssemblyProfiler(target));
  // Only continuous functions are supported.
  if (assembly_profiler_sp && m_ranges.size() == 1) {
    auto plan_sp = std::make_shared<UnwindPlan>(*debug_frame_plan);

    if (assembly_profiler_sp->AugmentUnwindPlanFromCallSite(m_ranges[0], thread,
                                                            *plan_sp))
      m_unwind_plan_debug_frame_augmented_sp = std::move(plan_sp);
  }
  return m_unwind_plan_debug_frame_augmented_sp;
}

std::shared_ptr<const UnwindPlan>
FuncUnwinders::GetAssemblyUnwindPlan(Target &target, Thread &thread) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (m_unwind_plan_assembly_sp.get() || m_tried_unwind_plan_assembly ||
      !m_unwind_table.GetAllowAssemblyEmulationUnwindPlans()) {
    return m_unwind_plan_assembly_sp;
  }

  m_tried_unwind_plan_assembly = true;

  UnwindAssemblySP assembly_profiler_sp(GetUnwindAssemblyProfiler(target));
  // Only continuous functions are supported.
  if (assembly_profiler_sp && m_ranges.size() == 1) {
    // Don't analyze more than 10 megabytes of instructions,
    // if a function is legitimately larger than that, we'll
    // miss the epilogue instructions, but guard against a
    // bogusly large function and analyzing large amounts of
    // non-instruction data.
    AddressRange range = m_ranges[0];
    const addr_t func_size =
        std::min(range.GetByteSize(), (addr_t)1024 * 10 * 10);
    range.SetByteSize(func_size);

    auto plan_sp = std::make_shared<UnwindPlan>(lldb::eRegisterKindGeneric);
    if (assembly_profiler_sp->GetNonCallSiteUnwindPlanFromAssembly(
            range, thread, *plan_sp))
      m_unwind_plan_assembly_sp = std::move(plan_sp);
  }
  return m_unwind_plan_assembly_sp;
}

// This method compares the pc unwind rule in the first row of two UnwindPlans.
// If they have the same way of getting the pc value (e.g. "CFA - 8" + "CFA is
// sp"), then it will return LazyBoolTrue.
LazyBool FuncUnwinders::CompareUnwindPlansForIdenticalInitialPCLocation(
    Thread &thread, const std::shared_ptr<const UnwindPlan> &a,
    const std::shared_ptr<const UnwindPlan> &b) {
  if (!a || !b)
    return eLazyBoolCalculate;

  const UnwindPlan::Row *a_first_row = a->GetRowAtIndex(0);
  const UnwindPlan::Row *b_first_row = b->GetRowAtIndex(0);
  if (!a_first_row || !b_first_row)
    return eLazyBoolCalculate;

  RegisterNumber pc_reg(thread, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
  uint32_t a_pc_regnum = pc_reg.GetAsKind(a->GetRegisterKind());
  uint32_t b_pc_regnum = pc_reg.GetAsKind(b->GetRegisterKind());

  UnwindPlan::Row::AbstractRegisterLocation a_pc_regloc;
  UnwindPlan::Row::AbstractRegisterLocation b_pc_regloc;

  a_first_row->GetRegisterInfo(a_pc_regnum, a_pc_regloc);
  b_first_row->GetRegisterInfo(b_pc_regnum, b_pc_regloc);

  if (a_first_row->GetCFAValue() != b_first_row->GetCFAValue())
    return eLazyBoolNo;
  if (a_pc_regloc != b_pc_regloc)
    return eLazyBoolNo;

  return eLazyBoolYes;
}

std::shared_ptr<const UnwindPlan>
FuncUnwinders::GetUnwindPlanAtNonCallSite(Target &target, Thread &thread) {
  std::shared_ptr<const UnwindPlan> eh_frame_sp = GetEHFrameUnwindPlan(target);
  if (!eh_frame_sp)
    eh_frame_sp = GetDebugFrameUnwindPlan(target);
  if (!eh_frame_sp)
    eh_frame_sp = GetObjectFileUnwindPlan(target);
  std::shared_ptr<const UnwindPlan> arch_default_at_entry_sp =
      GetUnwindPlanArchitectureDefaultAtFunctionEntry(thread);
  std::shared_ptr<const UnwindPlan> arch_default_sp =
      GetUnwindPlanArchitectureDefault(thread);
  std::shared_ptr<const UnwindPlan> assembly_sp =
      GetAssemblyUnwindPlan(target, thread);

  // This point of this code is to detect when a function is using a non-
  // standard ABI, and the eh_frame correctly describes that alternate ABI.
  // This is addressing a specific situation on x86_64 linux systems where one
  // function in a library pushes a value on the stack and jumps to another
  // function.  So using an assembly instruction based unwind will not work
  // when you're in the second function - the stack has been modified in a non-
  // ABI way.  But we have eh_frame that correctly describes how to unwind from
  // this location.  So we're looking to see if the initial pc register save
  // location from the eh_frame is different from the assembly unwind, the arch
  // default unwind, and the arch default at initial function entry.
  //
  // We may have eh_frame that describes the entire function -- or we may have
  // eh_frame that only describes the unwind after the prologue has executed --
  // so we need to check both the arch default (once the prologue has executed)
  // and the arch default at initial function entry.  And we may be running on
  // a target where we have only some of the assembly/arch default unwind plans
  // available.

  if (CompareUnwindPlansForIdenticalInitialPCLocation(
          thread, eh_frame_sp, arch_default_at_entry_sp) == eLazyBoolNo &&
      CompareUnwindPlansForIdenticalInitialPCLocation(
          thread, eh_frame_sp, arch_default_sp) == eLazyBoolNo &&
      CompareUnwindPlansForIdenticalInitialPCLocation(
          thread, assembly_sp, arch_default_sp) == eLazyBoolNo) {
    return eh_frame_sp;
  }

  if (std::shared_ptr<const UnwindPlan> plan_sp =
          GetSymbolFileUnwindPlan(thread))
    return plan_sp;
  if (std::shared_ptr<const UnwindPlan> plan_sp =
          GetDebugFrameAugmentedUnwindPlan(target, thread))
    return plan_sp;
  if (std::shared_ptr<const UnwindPlan> plan_sp =
          GetEHFrameAugmentedUnwindPlan(target, thread))
    return plan_sp;
  if (std::shared_ptr<const UnwindPlan> plan_sp =
          GetObjectFileAugmentedUnwindPlan(target, thread))
    return plan_sp;

  return assembly_sp;
}

std::shared_ptr<const UnwindPlan>
FuncUnwinders::GetUnwindPlanFastUnwind(Target &target, Thread &thread) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (m_unwind_plan_fast_sp.get() || m_tried_unwind_fast)
    return m_unwind_plan_fast_sp;

  m_tried_unwind_fast = true;

  UnwindAssemblySP assembly_profiler_sp(GetUnwindAssemblyProfiler(target));
  if (assembly_profiler_sp && m_ranges.size() == 1) {
    auto plan_sp = std::make_shared<UnwindPlan>(lldb::eRegisterKindGeneric);
    if (assembly_profiler_sp->GetFastUnwindPlan(m_ranges[0], thread, *plan_sp))
      m_unwind_plan_fast_sp = std::move(plan_sp);
  }
  return m_unwind_plan_fast_sp;
}

std::shared_ptr<const UnwindPlan>
FuncUnwinders::GetUnwindPlanArchitectureDefault(Thread &thread) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (m_unwind_plan_arch_default_sp.get() || m_tried_unwind_arch_default)
    return m_unwind_plan_arch_default_sp;

  m_tried_unwind_arch_default = true;

  ProcessSP process_sp(thread.CalculateProcess());
  if (process_sp) {
    if (ABI *abi = process_sp->GetABI().get())
      m_unwind_plan_arch_default_sp = abi->CreateDefaultUnwindPlan();
  }

  return m_unwind_plan_arch_default_sp;
}

std::shared_ptr<const UnwindPlan>
FuncUnwinders::GetUnwindPlanArchitectureDefaultAtFunctionEntry(Thread &thread) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (m_unwind_plan_arch_default_at_func_entry_sp.get() ||
      m_tried_unwind_arch_default_at_func_entry)
    return m_unwind_plan_arch_default_at_func_entry_sp;

  m_tried_unwind_arch_default_at_func_entry = true;

  Address current_pc;
  ProcessSP process_sp(thread.CalculateProcess());
  if (process_sp) {
    if (ABI *abi = process_sp->GetABI().get()) {
      m_unwind_plan_arch_default_at_func_entry_sp =
          abi->CreateFunctionEntryUnwindPlan();
    }
  }

  return m_unwind_plan_arch_default_at_func_entry_sp;
}

const Address &FuncUnwinders::GetFunctionStartAddress() const { return m_addr; }

lldb::UnwindAssemblySP
FuncUnwinders::GetUnwindAssemblyProfiler(Target &target) {
  UnwindAssemblySP assembly_profiler_sp;
  if (ArchSpec arch = m_unwind_table.GetArchitecture()) {
    arch.MergeFrom(target.GetArchitecture());
    assembly_profiler_sp = UnwindAssembly::FindPlugin(arch);
  }
  return assembly_profiler_sp;
}
