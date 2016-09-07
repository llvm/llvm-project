//===-- ProcessWindows.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_Common_ProcessWindows_H_
#define liblldb_Plugins_Process_Windows_Common_ProcessWindows_H_

// Other libraries and framework includes
#include "lldb/Core/Error.h"
#include "lldb/Target/Process.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {

class ProcessWindows : public lldb_private::Process {
public:
  //------------------------------------------------------------------
  // Constructors and destructors
  //------------------------------------------------------------------
  ProcessWindows(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp);

  ~ProcessWindows();

  size_t GetSTDOUT(char *buf, size_t buf_size,
                   lldb_private::Error &error) override;
  size_t GetSTDERR(char *buf, size_t buf_size,
                   lldb_private::Error &error) override;
  size_t PutSTDIN(const char *buf, size_t buf_size,
                  lldb_private::Error &error) override;

  lldb::addr_t GetImageInfoAddress() override;

protected:
  // These decode the page protection bits.
  static bool IsPageReadable(uint32_t protect);

  static bool IsPageWritable(uint32_t protect);

  static bool IsPageExecutable(uint32_t protect);
};
}

#endif // liblldb_Plugins_Process_Windows_Common_ProcessWindows_H_
