//===-- JITLoaderROAR.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_JITLOADER_ROAR_JITLOADERROAR_H
#define LLDB_SOURCE_PLUGINS_JITLOADER_ROAR_JITLOADERROAR_H

#include "lldb/Target/JITLoader.h"
#include "lldb/Target/Process.h"
#include "roar/Debug/ROARLLDBInterface.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/DynamicLibrary.h"

struct SymbolTableEntryModuleStorage;

namespace lldb_private {
class BreakpointResolverFileLine;
class BreakpointResolverAddress;
} // namespace lldb_private

namespace lldb_roar_private {
class JITLoaderROARSB {
  friend class JITLoaderROAR;

public:
  JITLoaderROARSB();

  ~JITLoaderROARSB();

  // JITLoader interface
  void DidAttach();

  void DidLaunch();

  void ModulesDidLoad(lldb_private::ModuleList &module_list);

  bool ResolveLoadAddress(lldb::addr_t load_addr, lldb_private::Address &addr);

  void HandleBreakpointEvent(
      lldb::BreakpointEventType sub_type, lldb_private::Breakpoint &breakpoint,
      const lldb_private::BreakpointLocationCollection *locations);

  /// Notify a JIT that debug information needs to be loaded for this
  /// trampoline address.
  void NotifyJITToLoadDebugInformation(lldb_private::Symbol &symbol);

  void SetProcess(lldb_private::Process *process) { m_process = process; }

private:
  using BreakpointOrLocationPtr =
      llvm::PointerUnion<lldb_private::Breakpoint *,
                         lldb_private::BreakpointLocation *>;

  void SetJITBreakpoint(lldb_private::ModuleList &module_list);

  bool DidSetJITBreakpoint() const;

  void ReadJITEntries();

  /// Notifies JIT of breakpoints that were set before symbol shared memory was
  /// initialized.
  void NotifyJITWithInitSymbols();

  void HandleBreakpointEventImpl(
      lldb_private::Log *log, bool add_bp_locs, bool loc_event,
      const lldb_private::BreakpointLocationCollection *locations,
      lldb_private::Breakpoint &breakpoint);

  void HandleNameBreakpointEvent(
      lldb_private::Log *log, bool add_bp_locs,
      const lldb_private::BreakpointLocationCollection *locations,
      lldb_private::Breakpoint &breakpoint);

  void HandleFileLineBreakpointEvent(
      lldb_private::Log *log, bool add_bp, lldb_private::Breakpoint &breakpoint,
      const lldb_private::BreakpointResolverFileLine &resolver);

  /// Handle address breakpoint event.
  void HandleAddressBreakpointEvent(
      lldb_private::Log *log, bool add_bp_locs,
      const lldb_private::BreakpointLocationCollection *locations,
      lldb_private::Breakpoint &breakpoint);

  void Reset();

  void Init(lldb_private::ModuleList &module_list);

  static bool
  JITDebugBreakpointHit(void *baton,
                        lldb_private::StoppointCallbackContext *context,
                        lldb::user_id_t break_id, lldb::user_id_t break_loc_id);

  /// Callback function for an internal breakpoint that is set on the on the
  /// function that is called when symbol shared memory was initialized.
  static bool JITDebugDynamicSymbolArenaAddrBreakpointHit(
      void *baton, lldb_private::StoppointCallbackContext *context,
      lldb::user_id_t break_id, lldb::user_id_t break_loc_id);

  // This function is called when the breakpoint in trampoline function is hit.
  static bool JITDebugTrampolineBreakpointHit(
      void *baton, lldb_private::StoppointCallbackContext *context,
      lldb::user_id_t break_id, lldb::user_id_t break_loc_id);

  lldb::user_id_t m_jit_break_id;

  // Breakpoint ID for function that gets triggered after shared memory is
  // initialized.
  lldb::user_id_t m_jit_dynamic_symbol_arena_addr_break_id;

  std::unique_ptr<roar_lldb::ROARDebugInterface> m_roar_di;

  // Flag used to guard against reentrying in ReadJITEntries.
  bool m_reading_jit_entries;

  lldb_private::Process *m_process;
  // Used to detect cycles in ResolveLoadAddress.
  lldb::addr_t m_load_address = 0;
};
} // namespace lldb_roar_private

namespace lldb_private {
class JITLoaderROAR : public JITLoader {
public:
  JITLoaderROAR(Process *process) : JITLoader(process) {
    jitLoader.SetProcess(process);
  };

  ~JITLoaderROAR() override {};

  // Static Functions
  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "roar"; }

  static llvm::StringRef GetPluginDescriptionStatic();

  static lldb::JITLoaderSP CreateInstance(Process *process, bool force);

  static void DebuggerInitialize(Debugger &debugger);

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  // JITLoader interface
  void DidAttach() override { jitLoader.DidAttach(); };

  void DidLaunch() override { jitLoader.DidLaunch(); };

  void ModulesDidLoad(ModuleList &module_list) override {
    jitLoader.ModulesDidLoad(module_list);
  };

  bool ResolveLoadAddress(lldb::addr_t load_addr, Address &addr) override {
    return jitLoader.ResolveLoadAddress(load_addr, addr);
  };

  void HandleBreakpointEvent(
      lldb::BreakpointEventType sub_type, Breakpoint &breakpoint,
      const BreakpointLocationCollection *locations) override {
    jitLoader.HandleBreakpointEvent(sub_type, breakpoint, locations);
  };

  /// Notify a JIT that debug information needs to be loaded for this
  /// trampoline address.
  void NotifyJITToLoadDebugInformation(Symbol &symbol) override {
    jitLoader.NotifyJITToLoadDebugInformation(symbol);
  };

private:
  lldb_roar_private::JITLoaderROARSB jitLoader;
};
} // namespace lldb_private
#endif // LLDB_SOURCE_PLUGINS_JITLOADER_ROAR_JITLOADERROAR_H
