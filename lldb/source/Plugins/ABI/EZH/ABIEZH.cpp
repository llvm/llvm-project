//===-- ABIEZH.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIEZH.h"

#include <array>

#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/RegisterValue.h"

#define DEFINE_REG_NAME(reg_num)      lldb_private::ConstString(#reg_num).GetCString()
#define DEFINE_REG_NAME_STR(reg_name) lldb_private::ConstString(reg_name).GetCString()

#define DEFINE_GENERIC_REGISTER_STUB(dwarf_num, str_name, generic_num)        \
  {                                                                           \
    DEFINE_REG_NAME(dwarf_num), DEFINE_REG_NAME_STR(str_name),                \
    4, 0, lldb::eEncodingUint, lldb::eFormatHex,                              \
    { dwarf_num, dwarf_num, generic_num, dwarf_num, dwarf_num },              \
    nullptr, nullptr, nullptr,                                                \
  }

#define DEFINE_REGISTER_STUB(dwarf_num, str_name) \
  DEFINE_GENERIC_REGISTER_STUB(dwarf_num, str_name, LLDB_INVALID_REGNUM)

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE_ADV(ABIEZH, ABIEZH)

namespace {
namespace dwarf {
enum regnums {
  r0, r1, r2, r3, r4, r5, r6, r7,
  gpo, gpd, cfs, cfm,
  sp, pc, gpi, ra, flags
};

static const std::array<RegisterInfo, 17> g_register_infos = { {
    DEFINE_GENERIC_REGISTER_STUB(r0, nullptr, LLDB_REGNUM_GENERIC_ARG1),
    DEFINE_GENERIC_REGISTER_STUB(r1, nullptr, LLDB_REGNUM_GENERIC_ARG2),
    DEFINE_GENERIC_REGISTER_STUB(r2, nullptr, LLDB_REGNUM_GENERIC_ARG3),
    DEFINE_GENERIC_REGISTER_STUB(r3, nullptr, LLDB_REGNUM_GENERIC_ARG4),
    DEFINE_REGISTER_STUB(r4, nullptr),
    DEFINE_REGISTER_STUB(r5, nullptr),
    DEFINE_REGISTER_STUB(r6, nullptr),
    DEFINE_REGISTER_STUB(r7, nullptr),
    DEFINE_REGISTER_STUB(gpo, nullptr),
    DEFINE_REGISTER_STUB(gpd, nullptr),
    DEFINE_REGISTER_STUB(cfs, nullptr),
    DEFINE_REGISTER_STUB(cfm, nullptr),
    DEFINE_GENERIC_REGISTER_STUB(sp, nullptr, LLDB_REGNUM_GENERIC_SP),
    DEFINE_GENERIC_REGISTER_STUB(pc, nullptr, LLDB_REGNUM_GENERIC_PC),
    DEFINE_REGISTER_STUB(gpi, nullptr),
    DEFINE_GENERIC_REGISTER_STUB(ra, nullptr, LLDB_REGNUM_GENERIC_RA),
    DEFINE_REGISTER_STUB(flags, nullptr)} };
} // namespace dwarf
} // namespace

const RegisterInfo *ABIEZH::GetRegisterInfoArray(uint32_t &count) {
  count = dwarf::g_register_infos.size();
  return dwarf::g_register_infos.data();
}

ABISP ABIEZH::CreateInstance(ProcessSP process_sp, const ArchSpec &arch) {
  if (arch.GetTriple().getArchName().starts_with("ezh")) {
    return ABISP(new ABIEZH(std::move(process_sp), MakeMCRegisterInfo(arch)));
  }
  return ABISP();
}

UnwindPlanSP ABIEZH::CreateFunctionEntryUnwindPlan() {
  UnwindPlan::Row row;
  // CFA is SP value
  row.GetCFAValue().SetIsRegisterPlusOffset(dwarf::sp, 0);
  // PC is in RA at function entry
  row.SetRegisterLocationToRegister(dwarf::pc, dwarf::ra, true);

  auto plan_sp = std::make_shared<UnwindPlan>(eRegisterKindDWARF);
  plan_sp->AppendRow(std::move(row));
  plan_sp->SetSourceName("ezh at-func-entry default");
  plan_sp->SetSourcedFromCompiler(eLazyBoolNo);
  return plan_sp;
}

UnwindPlanSP ABIEZH::CreateDefaultUnwindPlan() {
  UnwindPlan::Row row;
  // CFA is SP value
  row.GetCFAValue().SetIsRegisterPlusOffset(dwarf::sp, 0);
  // PC is in RA by default
  row.SetRegisterLocationToRegister(dwarf::pc, dwarf::ra, true);

  auto plan_sp = std::make_shared<UnwindPlan>(eRegisterKindDWARF);
  plan_sp->AppendRow(std::move(row));
  plan_sp->SetSourceName("ezh default unwind plan");
  plan_sp->SetSourcedFromCompiler(eLazyBoolNo);
  return plan_sp;
}

void ABIEZH::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "System ABI for EZH targets", CreateInstance);
}

void ABIEZH::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
