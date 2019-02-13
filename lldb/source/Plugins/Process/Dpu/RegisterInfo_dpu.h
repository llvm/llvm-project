//===-- RegisterInfo_dpu.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterInfo_dpu_h_
#define liblldb_RegisterInfo_dpu_h_

#include "Plugins/Process/Utility/RegisterInfoInterface.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/lldb-private.h"

class RegisterInfo_dpu : public lldb_private::RegisterInfoInterface {
public:
  struct GPR {
    uint32_t r[24]; // R0-R23
    uint32_t pc;    // PC
  };

  RegisterInfo_dpu();

  size_t GetGPRSize() const override;

  const lldb_private::RegisterInfo *GetRegisterInfo() const override;

  uint32_t GetRegisterCount() const override;
};

#endif // liblldb_RegisterInfo_dpu_h_
