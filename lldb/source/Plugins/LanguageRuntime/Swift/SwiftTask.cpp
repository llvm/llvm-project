#include "SwiftTask.h"
#include "SwiftLanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/Support/Error.h"

using namespace llvm;
using namespace lldb;

namespace lldb_private {

ThreadTask::ThreadTask(tid_t tid, addr_t async_ctx, addr_t resume_fn,
                       ExecutionContext &exe_ctx)
    : Thread(exe_ctx.GetProcessRef(), tid, true),
      m_reg_info_sp(exe_ctx.GetFrameSP()->GetRegisterContext()),
      m_async_ctx(async_ctx), m_resume_fn(resume_fn) {}

RegisterContextSP lldb_private::ThreadTask::GetRegisterContext() {
  if (!m_async_reg_ctx_sp)
    m_async_reg_ctx_sp = std::make_shared<RegisterContextTask>(
        *this, m_reg_info_sp, m_resume_fn, m_async_ctx);
  return m_async_reg_ctx_sp;
}

RegisterContextTask::RegisterContextTask(Thread &thread,
                                         RegisterContextSP reg_info_sp,
                                         addr_t resume_fn, addr_t async_ctx)
    : RegisterContext(thread, 0), m_reg_info_sp(reg_info_sp),
      m_async_ctx(async_ctx), m_resume_fn(resume_fn) {
  auto &target = thread.GetProcess()->GetTarget();
  auto triple = target.GetArchitecture().GetTriple();
  if (auto regnums = GetAsyncUnwindRegisterNumbers(triple.getArch()))
    m_async_ctx_regnum = regnums->async_ctx_regnum;
}

bool RegisterContextTask::ReadRegister(const RegisterInfo *reg_info,
                                       RegisterValue &reg_value) {
  if (!reg_info)
    return false;

  if (reg_info->kinds[eRegisterKindGeneric] == LLDB_REGNUM_GENERIC_PC) {
    reg_value = m_resume_fn;
    return true;
  }
  if (reg_info->kinds[eRegisterKindLLDB] == m_async_ctx_regnum) {
    reg_value = m_async_ctx;
    return true;
  }
  return false;
}

} // namespace lldb_private
