//===-- Dpu.h ------------------------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Dpu_H_
#define liblldb_Dpu_H_

// C Includes
#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

// C++ Includes
#include <mutex>
#include <string>

// Other libraries and framework includes
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/ObjectFile.h"

#include "DpuContext.h"

extern "C" {
#include <dpu.h>
}

namespace lldb_private {
namespace dpu {

class DpuRank;

class Dpu {
public:
  Dpu(DpuRank *rank, dpu_t *dpu, FILE *stdout_fd);
  ~Dpu();

  bool GetPrintfSequenceAddrs();

  bool LoadElf(const FileSpec &elf_file_path);
  bool Boot();

  lldb::StateType
  StepOverPrintfSequenceAndContinue(lldb::StateType result_state,
                                    unsigned int *exit_status);
  lldb::StateType PollStatus(unsigned int *exit_status);
  bool ResumeThreads(llvm::SmallVector<uint32_t, 8> *resume_list,
                     bool allowed_polling = true);
  bool StopThreads(bool force = false);

  bool
  PrepareStepOverPrintfBkp(const uint32_t thread_index,
                           dpuinstruction_t &inst_to_restore_if_printf_enable,
                           dpuinstruction_t &inst_to_replace_with,
                           const uint32_t current_pc);
  lldb::StateType StepThread(uint32_t thread_index, unsigned int *exit_status);

  bool WriteWRAM(uint32_t offset, const void *buf, size_t size);
  bool ReadWRAM(uint32_t offset, void *buf, size_t size);
  bool WriteIRAM(uint32_t offset, const void *buf, size_t size);
  bool ReadIRAM(uint32_t offset, void *buf, size_t size);
  bool WriteMRAM(uint32_t offset, const void *buf, size_t size);
  bool ReadMRAM(uint32_t offset, void *buf, size_t size);

  bool AllocIRAMBuffer(uint8_t **iram, uint32_t *iram_size);
  bool FreeIRAMBuffer(uint8_t *iram);
  bool GenerateSaveCore(const char *exe_path, const char *core_file_path,
                        uint8_t *iram, uint32_t iram_size);

  int GetNrThreads() { return nr_threads; }

  uint32_t *ThreadContextRegs(int thread_index);
  uint16_t *ThreadContextPC(int thread_index);
  bool *ThreadContextZF(int thread_index);
  bool *ThreadContextCF(int thread_index);
  bool *ThreadRegistersHasBeenModified();

  lldb::StateType GetThreadState(uint32_t thread_index,
                                 std::string &description,
                                 lldb::StopReason &stop_reason, bool stepping);

  unsigned int GetSliceID();
  unsigned int GetDpuID();

  bool SaveSliceContext(uint64_t structure_value, uint64_t slice_target,
                        dpu_bitfield_t host_mux_mram_state);
  bool RestoreSliceContext();

  void SetAttachSession();
  bool AttachSession();

  bool PrintfEnable();
  lldb::addr_t GetOpenPrintfSequenceAddr();
  lldb::addr_t GetClosePrintfSequenceAddr();

private:
  DpuRank *m_rank;
  dpu_t *m_dpu;
  int nr_threads;
  uint32_t nr_reg_per_thread;
  DpuContext *m_context;
  bool dpu_is_running = false;
  bool attach_session = false;
  bool registers_has_been_modified = false;
  bool printf_enable = false;
  lldb::addr_t open_print_sequence_addr, close_print_sequence_addr;
  dpuinstruction_t open_print_sequence_inst, close_print_sequence_inst;
  uint32_t printf_buffer_last_idx, printf_buffer_var_addr,
      printf_buffer_address, printf_buffer_size;
  FILE *stdout_file;
};
} // namespace dpu
} // namespace lldb_private

#endif // liblldb_Dpu_H_
