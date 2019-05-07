//===-- ABIrte_dpu.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ABIrte_dpu_h_
#define liblldb_ABIrte_dpu_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/ABI.h"
#include "lldb/lldb-private.h"

class ABIrte_dpu : public lldb_private::ABI {
public:
  ~ABIrte_dpu() override = default;

  size_t GetRedZoneSize() const override;

  bool PrepareTrivialCall(lldb_private::Thread &thread, lldb::addr_t sp,
                          lldb::addr_t func_addr, lldb::addr_t returnAddress,
                          llvm::ArrayRef<lldb::addr_t> args) const override;

  bool GetArgumentValues(lldb_private::Thread &thread,
                         lldb_private::ValueList &values) const override;

  lldb_private::Status
  SetReturnValueObject(lldb::StackFrameSP &frame_sp,
                       lldb::ValueObjectSP &new_value_sp) override;

  bool
  CreateFunctionEntryUnwindPlan(lldb_private::UnwindPlan &unwind_plan) override;

  bool CreateDefaultUnwindPlan(lldb_private::UnwindPlan &unwind_plan) override;

  bool RegisterIsVolatile(const lldb_private::RegisterInfo *reg_info) override;

  bool CallFrameAddressIsValid(lldb::addr_t cfa) override;

  bool CodeAddressIsValid(lldb::addr_t pc) override;

  const lldb_private::RegisterInfo *
  GetRegisterInfoArray(uint32_t &count) override;

  //------------------------------------------------------------------
  // Static Functions
  //------------------------------------------------------------------

  static void Initialize();

  static void Terminate();

  static lldb::ABISP CreateInstance(lldb::ProcessSP process_sp,
                                    const lldb_private::ArchSpec &arch);

  static lldb_private::ConstString GetPluginNameStatic();

  //------------------------------------------------------------------
  // PluginInterface protocol
  //------------------------------------------------------------------

  lldb_private::ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

protected:
  lldb::ValueObjectSP GetReturnValueObjectImpl(
      lldb_private::Thread &thread,
      lldb_private::CompilerType &return_compiler_type) const override;

private:
  ABIrte_dpu(lldb::ProcessSP process_sp) : lldb_private::ABI(process_sp) {
    // Call CreateInstance instead.
  }

  bool RegisterIsCalleeSaved(const lldb_private::RegisterInfo *reg_info);
};

#endif // liblldb_ABIrte_dpu_h_
