//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SCRIPTED_FRAME_H
#define LLDB_SOURCE_PLUGINS_SCRIPTED_FRAME_H

#include "ScriptedThread.h"
#include "lldb/Target/DynamicRegisterInfo.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/lldb-forward.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <string>

namespace lldb_private {

class ScriptedFrame : public lldb_private::StackFrame {

public:
  ScriptedFrame(lldb::ThreadSP thread_sp,
                lldb::ScriptedFrameInterfaceSP interface_sp,
                lldb::user_id_t frame_idx, lldb::addr_t pc,
                SymbolContext &sym_ctx, lldb::RegisterContextSP reg_ctx_sp,
                StructuredData::GenericSP script_object_sp = nullptr);

  ~ScriptedFrame() override;

  /// Create a ScriptedFrame from a object instanciated in the script
  /// interpreter.
  ///
  /// \param[in] thread_sp
  ///     The thread this frame belongs to.
  ///
  /// \param[in] scripted_thread_interface_sp
  ///     The scripted thread interface (needed for ScriptedThread
  ///     compatibility). Can be nullptr for frames on real threads.
  ///
  /// \param[in] args_sp
  ///     Arguments to pass to the frame creation.
  ///
  /// \param[in] script_object
  ///     The optional script object representing this frame.
  ///
  /// \return
  ///     An Expected containing the ScriptedFrame shared pointer if successful,
  ///     otherwise an error.
  static llvm::Expected<std::shared_ptr<ScriptedFrame>>
  Create(lldb::ThreadSP thread_sp,
         lldb::ScriptedThreadInterfaceSP scripted_thread_interface_sp,
         StructuredData::DictionarySP args_sp,
         StructuredData::Generic *script_object = nullptr);

  bool IsInlined() override;
  bool IsArtificial() const override;
  bool IsHidden() override;
  const char *GetFunctionName() override;
  const char *GetDisplayFunctionName() override;

  lldb::RegisterContextSP GetRegisterContext() override;

  bool isA(const void *ClassID) const override {
    return ClassID == &ID || StackFrame::isA(ClassID);
  }
  static bool classof(const StackFrame *obj) { return obj->isA(&ID); }

private:
  void CheckInterpreterAndScriptObject() const;
  lldb::ScriptedFrameInterfaceSP GetInterface() const;
  static llvm::Expected<lldb::RegisterContextSP>
  CreateRegisterContext(ScriptedFrameInterface &interface, Thread &thread,
                        lldb::user_id_t frame_id);

  ScriptedFrame(const ScriptedFrame &) = delete;
  const ScriptedFrame &operator=(const ScriptedFrame &) = delete;

  std::shared_ptr<DynamicRegisterInfo> GetDynamicRegisterInfo();

  lldb::ScriptedFrameInterfaceSP m_scripted_frame_interface_sp;
  lldb_private::StructuredData::GenericSP m_script_object_sp;

  static char ID;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SCRIPTED_FRAME_H
