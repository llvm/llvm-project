//===-- ABIWindows_x86_64.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ABIWindows_x86_64_h_
#define liblldb_ABIWindows_x86_64_h_

#include "lldb/Target/ABI.h"
#include "lldb/lldb-private.h"

class ABIWindows_x86_64 : public lldb_private::ABI {
public:
  ~ABIWindows_x86_64() override = default;

  size_t GetRedZoneSize() const override;

  bool PrepareTrivialCall(lldb_private::Thread &thread, lldb::addr_t sp,
                          lldb::addr_t functionAddress,
                          lldb::addr_t returnAddress,
                          llvm::ArrayRef<lldb::addr_t> args) const override;

  bool GetArgumentValues(lldb_private::Thread &thread,
                         lldb_private::ValueList &values) const override;

  lldb_private::Status
  SetReturnValueObject(lldb::StackFrameSP &frame_sp,
                       lldb::ValueObjectSP &new_value) override;

  lldb::ValueObjectSP
  GetReturnValueObjectImpl(lldb_private::Thread &thread,
                           lldb_private::CompilerType &type) const override;

  bool
  CreateFunctionEntryUnwindPlan(lldb_private::UnwindPlan &unwind_plan) override;

  bool CreateDefaultUnwindPlan(lldb_private::UnwindPlan &unwind_plan) override;

  bool RegisterIsVolatile(const lldb_private::RegisterInfo *reg_info) override;

  // In Windows_x86_64 ABI, stack will always be maintained 16-byte aligned
  bool CallFrameAddressIsValid(lldb::addr_t cfa) override {
	  if (cfa & (16ull - 1ull))
      return false; // Not 16 byte aligned
    if (cfa == 0)
      return false; // Zero is not a valid stack address
    return true;
  }

  bool CodeAddressIsValid(lldb::addr_t pc) override {
    // We have a 64 bit address space, so anything is valid as opcodes
    // aren't fixed width...
    return true;
  }

  const lldb_private::RegisterInfo *
  GetRegisterInfoArray(uint32_t &count) override;

  bool GetPointerReturnRegister(const char *&name) override;

  //------------------------------------------------------------------
  // Static Functions
  //------------------------------------------------------------------

  static void Initialize();

  static void Terminate();

  static lldb::ABISP CreateInstance(lldb::ProcessSP process_sp, const lldb_private::ArchSpec &arch);

  static lldb_private::ConstString GetPluginNameStatic();

  //------------------------------------------------------------------
  // PluginInterface protocol
  //------------------------------------------------------------------

  lldb_private::ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

protected:
  void CreateRegisterMapIfNeeded();

  lldb::ValueObjectSP
  GetReturnValueObjectSimple(lldb_private::Thread &thread,
                             lldb_private::CompilerType &ast_type) const;

  bool RegisterIsCalleeSaved(const lldb_private::RegisterInfo *reg_info);

private:
  ABIWindows_x86_64(lldb::ProcessSP process_sp) : lldb_private::ABI(process_sp) {
    // Call CreateInstance instead.
  }
};

#endif // liblldb_ABISysV_x86_64_h_
