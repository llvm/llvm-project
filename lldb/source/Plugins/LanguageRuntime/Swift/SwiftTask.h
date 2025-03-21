
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {

using namespace lldb;

/// Provides a subset of Thread operations for Swift Tasks.
///
/// Currently, this supports backtraces of Tasks, and selecting frames in the
/// backtrace. Async frames make available the variables that are stored in the
/// Task's "async context" (instead of the stack).
///
/// See `Task<Success, Failure>` and `UnsafeCurrentTask`
class ThreadTask final : public Thread {
public:
  ThreadTask(tid_t tid, addr_t async_ctx, addr_t resume_fn,
             ExecutionContext &exe_ctx);

  /// Returns a Task specific register context (RegisterContextTask).
  RegisterContextSP GetRegisterContext() override;

  ~ThreadTask() override { DestroyThread(); }

  // No-op overrides.
  void RefreshStateAfterStop() override {}
  lldb::RegisterContextSP
  CreateRegisterContextForFrame(StackFrame *frame) override {
    return {};
  }
  bool CalculateStopInfo() override { return false; }

private:
  /// A register context that is the source of `RegisterInfo` data.
  RegisterContextSP m_reg_info_sp;
  /// Lazily initialized `RegisterContextTask`.
  RegisterContextSP m_async_reg_ctx_sp;
  /// The Task's async context.
  addr_t m_async_ctx = LLDB_INVALID_ADDRESS;
  /// The address of the async context's resume function.
  addr_t m_resume_fn = LLDB_INVALID_ADDRESS;
};

/// A Swift Task specific register context. Supporting class for `ThreadTask`,
/// see its documentation for details.
class RegisterContextTask final : public RegisterContext {
public:
  RegisterContextTask(Thread &thread, RegisterContextSP reg_info_sp,
                      addr_t resume_fn, addr_t async_ctx);

  /// RegisterContextTask supports readonly from only two (necessary)
  /// registers. Namely, the pc and the async context registers.
  bool ReadRegister(const RegisterInfo *reg_info,
                    RegisterValue &reg_value) override;

  // Pass through overrides.
  size_t GetRegisterCount() override {
    return m_reg_info_sp->GetRegisterCount();
  }
  const RegisterInfo *GetRegisterInfoAtIndex(size_t idx) override {
    return m_reg_info_sp->GetRegisterInfoAtIndex(idx);
  }
  size_t GetRegisterSetCount() override {
    return m_reg_info_sp->GetRegisterSetCount();
  }
  const RegisterSet *GetRegisterSet(size_t reg_set) override {
    return m_reg_info_sp->GetRegisterSet(reg_set);
  }
  lldb::ByteOrder GetByteOrder() override {
    return m_reg_info_sp->GetByteOrder();
  }

  // No-op overrides.
  void InvalidateAllRegisters() override {}
  bool WriteRegister(const RegisterInfo *reg_info,
                     const RegisterValue &reg_value) override {
    return false;
  }

private:
  /// A register context that is the source of `RegisterInfo` data.
  RegisterContextSP m_reg_info_sp;
  /// The architecture specific regnum (LLDB) which holds the async context.
  uint32_t m_async_ctx_regnum = LLDB_INVALID_REGNUM;
  /// The Task's async context.
  RegisterValue m_async_ctx;
  /// The address of the async context's resume function.
  RegisterValue m_resume_fn;
};

} // namespace lldb_private
