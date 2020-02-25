//===-- Dpu.cpp ----------------------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Dpu.h"
#include "DpuRank.h"

#include "ProcessDpu.h"

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

extern "C" {
#include <dpu.h>
#include <dpu_custom.h>
#include <dpu_debug.h>
#include <dpu_log.h>
#include <dpu_management.h>
#include <dpu_memory.h>
#include <dpu_program.h>
#include <dpu_runner.h>
}

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::dpu;

namespace {
const uint32_t instruction_size_mod = sizeof(dpuinstruction_t) - 1;
const uint32_t instruction_size_mask = ~instruction_size_mod;
const uint32_t dpuword_size_mod = sizeof(dpuword_t) - 1;
const uint32_t dpuword_size_mask = ~dpuword_size_mod;
const uint32_t mram_aligned = 8;
const uint32_t mram_aligned_mod = mram_aligned - 1;
const uint32_t mram_aligned_mask = ~mram_aligned_mod;
} // namespace

// -----------------------------------------------------------------------------
// DPU handling
// -----------------------------------------------------------------------------

Dpu::Dpu(DpuRank *rank, dpu_t *dpu, FILE *stdout_file_)
    : m_rank(rank), m_dpu(dpu), printf_enable(false),
      printf_buffer_last_idx((uint32_t)LLDB_INVALID_ADDRESS),
      printf_buffer_var_addr((uint32_t)LLDB_INVALID_ADDRESS) {
  nr_threads = m_rank->GetNrThreads();
  nr_reg_per_thread = rank->GetDesc()->dpu.nr_of_work_registers_per_thread;

  m_context = new DpuContext(m_dpu, m_rank->AllocContext(), nr_threads);

  stdout_file = stdout_file_;
}

Dpu::~Dpu() {
  fclose(stdout_file);
  delete m_context;
}

bool Dpu::SetPrintfSequenceAddrs(const uint32_t _open_print_sequence_addr,
                                 const uint32_t _close_print_sequence_addr,
                                 const uint32_t _printf_buffer_address,
                                 const uint32_t _printf_buffer_size,
                                 const uint32_t _printf_buffer_var_addr) {
  open_print_sequence_addr =
      _open_print_sequence_addr +
      sizeof(dpuinstruction_t); // we add sizeof(dpuinstruction_t) to avoid to
                                // be on the 'acquire' which could makes us loop
                                // forever
  close_print_sequence_addr = _close_print_sequence_addr;
  printf_buffer_address = _printf_buffer_address;
  printf_buffer_size = _printf_buffer_size;
  printf_buffer_var_addr = _printf_buffer_var_addr;
  if (open_print_sequence_addr != (lldb::addr_t)LLDB_INVALID_ADDRESS &&
      close_print_sequence_addr != (lldb::addr_t)LLDB_INVALID_ADDRESS) {
    printf_enable = true;
    if (!ReadIRAM(open_print_sequence_addr, &open_print_sequence_inst,
                  sizeof(open_print_sequence_inst)))
      return false;
    if (!ReadIRAM(close_print_sequence_addr, &close_print_sequence_inst,
                  sizeof(close_print_sequence_inst)))
      return false;
  }
  return true;
}

bool Dpu::SetPrintfSequenceAddrsFromRuntimeInfo(dpu_program_t *runtime) {
  lldb::addr_t _open_print_sequence_addr = runtime->open_print_sequence_addr;
  lldb::addr_t _close_print_sequence_addr = runtime->close_print_sequence_addr;
  if (_open_print_sequence_addr != (lldb::addr_t)LLDB_INVALID_ADDRESS &&
      _close_print_sequence_addr != (lldb::addr_t)LLDB_INVALID_ADDRESS) {
    _open_print_sequence_addr = InstIdx2InstAddr(_open_print_sequence_addr);
    _close_print_sequence_addr = InstIdx2InstAddr(_close_print_sequence_addr);
    return SetPrintfSequenceAddrs(
        _open_print_sequence_addr, _close_print_sequence_addr,
        runtime->printf_buffer_address, runtime->printf_buffer_size,
        runtime->printf_write_pointer_address);
  }
  return true;
}

bool Dpu::LoadElf(const FileSpec &elf_file_path) {
  ModuleSP elf_mod(new Module(elf_file_path, k_dpu_arch));

  struct dpu_set_t set = dpu_set_from_dpu(m_dpu);
  dpu_error_t status = dpu_load(set, elf_file_path.GetCString(), NULL);
  if (status != DPU_OK)
    return false;

  dpu_program_t *runtime = dpu_get_program(m_dpu);
  if (!runtime)
    return false;

  uint8_t nr_threads_enabled = runtime->nr_threads_enabled;
  if (nr_threads_enabled != (uint8_t)-1)
    nr_threads = nr_threads_enabled;

  if (!SetPrintfSequenceAddrsFromRuntimeInfo(runtime))
    return false;

  return true;
}

bool Dpu::Boot() {
  struct _dpu_context_t *tmp_context;
  // Extract a potential context from the dpu structure (that could have been
  // created by the dpu_loader).
  dpu_pop_debug_context(m_dpu, &tmp_context);

  // If the dpu contains a context, it means that a core file has been loaded.
  // In this case, do not do the boot sequence, the dpu is already in a state
  // ready to be resume with 'dpu_finalize_fault_process_for_dpu'.
  if (tmp_context != NULL) {
    // Free our previously allocated context and get the one created by the
    // dpu_loader.
    m_context->UpdateContext(tmp_context);
    return true;
  }

  dpuinstruction_t first_instruction;
  ReadIRAM(0, (void *)(&first_instruction), sizeof(dpuinstruction_t));

  const dpuinstruction_t breakpoint_instruction = 0x00007e6320000000;
  WriteIRAM(0, (const void *)(&breakpoint_instruction),
            sizeof(dpuinstruction_t));

  int res = dpu_custom_for_dpu(m_dpu, DPU_COMMAND_DPU_PREEXECUTION, NULL);
  if (res != DPU_OK)
    return false;

  bool ignored;
  res = dpu_launch_thread_on_dpu(m_dpu, DPU_BOOT_THREAD, false, &ignored);
  if (res != DPU_OK)
    return false;

  dpu_is_running = true;
  while (1) {
    unsigned int exit_status;
    switch (PollStatus(&exit_status)) {
    case StateType::eStateStopped:
      WriteIRAM(0, (const void *)(&first_instruction),
                sizeof(dpuinstruction_t));
      return true;
    case StateType::eStateRunning:
      break;
    default:
      return false;
    }
  }
}

bool Dpu::StopThreads(bool force) {
  std::lock_guard<std::recursive_mutex> guard(m_rank->GetLock());

  if (!dpu_is_running && !force)
    return true;
  dpu_is_running = false;

  return m_context->StopThreads();
}

StateType Dpu::StepOverPrintfSequenceAndContinue(StateType result_state,
                                                 unsigned int *exit_status) {
  dpu_thread_t thread_in_fault = m_context->Get()->bkp_fault_thread_index;
  lldb::addr_t pc_of_thread_in_fault =
      m_context->GetPcOfThread(thread_in_fault);
  if (pc_of_thread_in_fault == open_print_sequence_addr ||
      pc_of_thread_in_fault == close_print_sequence_addr) {
    result_state = StepThread(thread_in_fault, exit_status);

    *exit_status = m_context->GetExitStatus();

    if (result_state != StateType::eStateStopped)
      return result_state;
    if (ResumeThreads(NULL, true))
      return StateType::eStateRunning;
    else
      return StateType::eStateCrashed;
  }
  return result_state;
}

StateType Dpu::PollStatus(unsigned int *exit_status) {
  std::lock_guard<std::recursive_mutex> guard(m_rank->GetLock());
  bool dpu_is_in_fault;
  StateType result_state = StateType::eStateRunning;

  if (!dpu_is_running)
    return StateType::eStateInvalid;

  dpu_poll_dpu(m_dpu, &dpu_is_running, &dpu_is_in_fault);
  if (dpu_is_in_fault) {
    result_state = StateType::eStateStopped;
  } else if (!dpu_is_running) {
    result_state = m_context->DpuIsRunning(nr_threads)
                       ? StateType::eStateStopped
                       : StateType::eStateExited;
  } else {
    return StateType::eStateRunning;
  }

  if (!StopThreads(true)) {
    return StateType::eStateCrashed;
  }
  // Needs to be after StopThreads to make sure that context is up to
  // date.
  *exit_status = m_context->GetExitStatus();

  if (dpu_is_in_fault && m_context->Get()->bkp_fault && printf_enable) {
    return StepOverPrintfSequenceAndContinue(result_state, exit_status);
  }

  return result_state;
}

bool Dpu::ResumeThreads(llvm::SmallVector<uint32_t, 8> *resume_list,
                        bool allowed_polling) {
  std::lock_guard<std::recursive_mutex> guard(m_rank->GetLock());

  if (!m_context->ContextReadyForResumeOrStep()) {
    m_context->RestoreFaultContext();
    return false;
  }

  int ret = DPU_OK;
  if (registers_has_been_modified) {
    ret = dpu_restore_context_for_dpu(m_dpu, m_context->Get());
    if (ret != DPU_OK)
      return false;
    registers_has_been_modified = false;
  }

  bool status = m_context->ResumeThreads(resume_list);

  if (allowed_polling && status)
    dpu_is_running = true;

  return status;
}

bool Dpu::PrepareStepOverPrintfBkp(
    const uint32_t thread_index,
    dpuinstruction_t &inst_to_restore_if_printf_enable,
    dpuinstruction_t &inst_to_replace_with, const uint32_t current_pc) {
  if (current_pc == open_print_sequence_addr) {
    inst_to_replace_with = open_print_sequence_inst;

    if (printf_buffer_var_addr != (uint32_t)LLDB_INVALID_ADDRESS) {
      if (!ReadWRAM(printf_buffer_var_addr, &printf_buffer_last_idx,
                    sizeof(printf_buffer_last_idx)))
        return false;
    }
  } else if (current_pc == close_print_sequence_addr) {
    inst_to_replace_with = close_print_sequence_inst;
    if (printf_buffer_var_addr != (uint32_t)LLDB_INVALID_ADDRESS &&
        printf_buffer_last_idx != (uint32_t)LLDB_INVALID_ADDRESS) {
      uint32_t printf_buffer_current_idx;
      if (!ReadWRAM(printf_buffer_var_addr, &printf_buffer_current_idx,
                    sizeof(printf_buffer_current_idx)))
        return false;

      size_t mram_buffer_size;
      uint8_t *mram_buffer = (uint8_t *)calloc(1, printf_buffer_size);
      if (mram_buffer == NULL)
        return false;
      if (printf_buffer_current_idx <= printf_buffer_last_idx) {
        mram_buffer_size = printf_buffer_size -
                           (printf_buffer_last_idx - printf_buffer_current_idx);
        if (!ReadMRAM(printf_buffer_last_idx, mram_buffer,
                      printf_buffer_size - printf_buffer_last_idx)) {
          goto PrepareStepOverPrintfBkp_err;
        }
        if (!ReadMRAM(printf_buffer_address & (k_dpu_mram_base - 1),
                      &mram_buffer[printf_buffer_size - printf_buffer_last_idx],
                      printf_buffer_current_idx)) {
          goto PrepareStepOverPrintfBkp_err;
        }
      } else {
        mram_buffer_size = printf_buffer_current_idx - printf_buffer_last_idx;
        if (!ReadMRAM((printf_buffer_last_idx + printf_buffer_address) &
                          (k_dpu_mram_base - 1),
                      mram_buffer, mram_buffer_size)) {
          goto PrepareStepOverPrintfBkp_err;
        }
      }

      if (stdout_file != NULL) {
        if (dpulog_read_and_display_contents_of(mram_buffer, mram_buffer_size,
                                                stdout_file) != DPU_OK) {
          goto PrepareStepOverPrintfBkp_err;
        }

        fflush(stdout_file);
      }

      free(mram_buffer);
      return true;
    PrepareStepOverPrintfBkp_err:
      free(mram_buffer);
      return false;
    }
  }

  return true;
}

#define UNKNOWN_INSTRUCTION (0ULL)

StateType Dpu::StepThread(uint32_t thread_index, unsigned int *exit_status) {
  std::lock_guard<std::recursive_mutex> guard(m_rank->GetLock());

  if (!m_context->ContextReadyForResumeOrStep())
    return StateType::eStateCrashed;

  // If the thread is not in the scheduling list, do not try to step it.
  // This behavior is expected as lldb can ask to step one thread and resume all
  // the other, which result in stepping all the thread contained in the
  // scheduling list.
  if (!m_context->ScheduledThread(thread_index))
    return StateType::eStateStopped;

  int ret = DPU_OK;
  if (registers_has_been_modified) {
    ret = dpu_restore_context_for_dpu(m_dpu, m_context->Get());
    if (ret != DPU_OK)
      return StateType::eStateCrashed;
    registers_has_been_modified = false;
  }

  uint64_t inst_to_restore_if_printf_enable = UNKNOWN_INSTRUCTION;
  uint32_t current_pc;
  if (printf_enable) {
    uint64_t inst_to_replace_with = UNKNOWN_INSTRUCTION;
    current_pc = m_context->GetPcOfThread(thread_index);

    if (!PrepareStepOverPrintfBkp(thread_index,
                                  inst_to_restore_if_printf_enable,
                                  inst_to_replace_with, current_pc))
      return StateType::eStateCrashed;

    if (inst_to_replace_with != UNKNOWN_INSTRUCTION) {
      // Read what should be the bkp instruction and write the expected
      // instruction instead
      if (!ReadIRAM(current_pc, &inst_to_restore_if_printf_enable,
                    sizeof(inst_to_restore_if_printf_enable)))
        return StateType::eStateCrashed;
      if (!WriteIRAM(current_pc, (const void *)&inst_to_replace_with,
                     sizeof(inst_to_replace_with)))
        return StateType::eStateCrashed;
    }
  }

  ret = m_context->StepThread(thread_index);

  // Write back the breakpoint
  if (inst_to_restore_if_printf_enable != UNKNOWN_INSTRUCTION) {
    if (!WriteIRAM(current_pc, (const void *)&inst_to_restore_if_printf_enable,
                   sizeof(inst_to_restore_if_printf_enable)))
      return StateType::eStateCrashed;
  }

  if (ret != DPU_OK)
    return StateType::eStateCrashed;
  if (!m_context->DpuIsRunning(nr_threads)) {
    *exit_status = m_context->GetExitStatus();
    return StateType::eStateExited;
  }
  return StateType::eStateStopped;
}

bool Dpu::WriteWRAM(uint32_t offset, const void *buf, size_t size) {
  std::lock_guard<std::recursive_mutex> guard(m_rank->GetLock());

  dpu_error_t ret;
  // fast path, everything is aligned
  if (((offset & dpuword_size_mod) == 0) && ((size & dpuword_size_mod) == 0)) {
    const dpuword_t *words = static_cast<const dpuword_t *>(buf);
    ret = dpu_copy_to_wram_for_dpu(m_dpu, offset / sizeof(dpuword_t), words,
                                   size / sizeof(dpuword_t));
    return ret == DPU_OK;
  }

  // slow path

  // compute final_offset to start from and the final_size to use
  uint32_t final_offset_in_dpuword =
      (offset & dpuword_size_mask) / sizeof(dpuword_t);
  size_t padding = offset & dpuword_size_mod;
  size_t final_size_in_dpuword =
      ((size + dpuword_size_mod + padding) & dpuword_size_mask) /
      sizeof(dpuword_t);

  // allocating the buffer of dpuwords to read/write from
  dpuword_t *words = new dpuword_t[final_size_in_dpuword];
  if (words == NULL)
    return false;

  // reading the dpuwords
  ret = dpu_copy_from_wram_for_dpu(m_dpu, words, final_offset_in_dpuword,
                                   final_size_in_dpuword);
  if (ret != DPU_OK) {
    delete[] words;
    return false;
  }

  // copy the dpuwords into our buffer
  memcpy(&((uint8_t *)words)[padding], buf, size);

  // writing the buffer
  ret = dpu_copy_to_wram_for_dpu(m_dpu, final_offset_in_dpuword, words,
                                 final_size_in_dpuword);

  delete[] words;
  return ret == DPU_OK;
}

bool Dpu::ReadWRAM(uint32_t offset, void *buf, size_t size) {
  std::lock_guard<std::recursive_mutex> guard(m_rank->GetLock());
  dpuword_t *words = static_cast<dpuword_t *>(buf);

  dpu_error_t ret;
  uint32_t padding = offset & dpuword_size_mod;
  size_t final_size = (size + dpuword_size_mod + padding) & dpuword_size_mask;

  // if an aligned copy copy more than asked by the function, let's read more
  // and then copy only the wanted part into the output buffer.
  if (final_size != size) {
    iram_size_t final_size_in_dpuword = final_size / sizeof(dpuword_t);
    uint32_t final_offset_in_dpuword =
        (offset & dpuword_size_mask) / sizeof(dpuword_t);

    words = new dpuword_t[final_size_in_dpuword];
    if (words == NULL)
      return false;

    ret = dpu_copy_from_wram_for_dpu(m_dpu, words, final_offset_in_dpuword,
                                     final_size_in_dpuword);

    memcpy(buf, &words[padding], size);
    delete[] words;

  } else {
    ret = dpu_copy_from_wram_for_dpu(m_dpu, words, offset / sizeof(dpuword_t),
                                     size / sizeof(dpuword_t));
  }
  return ret == DPU_OK;
}

bool Dpu::WriteIRAM(uint32_t offset, const void *buf, size_t size) {
  std::lock_guard<std::recursive_mutex> guard(m_rank->GetLock());

  dpu_error_t ret;
  // fast path, everything is aligned
  if (((offset & instruction_size_mod) == 0) &&
      ((size & instruction_size_mod) == 0)) {
    const dpuinstruction_t *instrs = static_cast<const dpuinstruction_t *>(buf);
    ret = dpu_copy_to_iram_for_dpu(m_dpu, offset / sizeof(dpuinstruction_t),
                                   instrs, size / sizeof(dpuinstruction_t));
    return ret == DPU_OK;
  }

  // slow path

  // compute final_offset to start from and the final_size to use
  uint32_t final_offset_in_instructions = offset & instruction_size_mask;
  size_t padding = offset & instruction_size_mod;
  size_t final_size_in_instructions =
      ((size + instruction_size_mod + padding) & instruction_size_mask) /
      sizeof(dpuinstruction_t);

  // allocating the buffer of instruction to read/write from
  dpuinstruction_t *instrs = new dpuinstruction_t[final_size_in_instructions];
  if (instrs == NULL)
    return false;

  ret = dpu_copy_from_iram_for_dpu(m_dpu, instrs, final_offset_in_instructions,
                                   final_size_in_instructions);
  if (ret != DPU_OK) {
    delete[] instrs;
    return false;
  }

  memcpy(&((uint8_t *)instrs)[padding], buf, size);

  ret = dpu_copy_to_iram_for_dpu(m_dpu, final_offset_in_instructions, instrs,
                                 final_size_in_instructions);

  delete[] instrs;
  return ret == DPU_OK;
}

bool Dpu::ReadIRAM(uint32_t offset, void *buf, size_t size) {
  std::lock_guard<std::recursive_mutex> guard(m_rank->GetLock());
  dpuinstruction_t *instrs = static_cast<dpuinstruction_t *>(buf);

  dpu_error_t ret;
  uint32_t padding = offset & instruction_size_mod;
  size_t final_size =
      (size + instruction_size_mod + padding) & instruction_size_mask;

  // if an aligned copy copy more than asked by the function, let's read more
  // and then copy only the wanted part into the output buffer.
  if (final_size != size) {
    iram_size_t final_size_in_instructions =
        final_size / sizeof(dpuinstruction_t);
    uint32_t final_offset_in_instructions =
        (offset & instruction_size_mask) / sizeof(dpuinstruction_t);

    instrs = new dpuinstruction_t[final_size_in_instructions];
    if (instrs == NULL)
      return false;

    ret =
        dpu_copy_from_iram_for_dpu(m_dpu, instrs, final_offset_in_instructions,
                                   final_size_in_instructions);

    memcpy(buf, &instrs[padding], size);
    delete[] instrs;
  } else {
    ret = dpu_copy_from_iram_for_dpu(m_dpu, instrs,
                                     offset / sizeof(dpuinstruction_t),
                                     size / sizeof(dpuinstruction_t));
  }
  return ret == DPU_OK;
}

bool Dpu::WriteMRAM(uint32_t offset, const void *buf, size_t size) {
  std::lock_guard<std::recursive_mutex> guard(m_rank->GetLock());

  dpu_error_t ret;
  // fast path, everything is aligned
  if (((offset & mram_aligned_mod) == 0) && ((size & mram_aligned_mod) == 0)) {
    const uint8_t *bytes = static_cast<const uint8_t *>(buf);
    ret = dpu_copy_to_mram(m_dpu, offset, bytes, size, 0);
    return ret == DPU_OK;
  }

  // slow path

  // compute final_offset to start from and the final_size to use
  uint32_t final_offset = offset & mram_aligned_mask;
  size_t padding = offset & mram_aligned_mod;
  size_t final_size = (size + padding + mram_aligned_mod) & mram_aligned_mask;

  // allocation the buffer of mram to read/write from
  uint8_t *bytes = new uint8_t[final_size];
  if (bytes == NULL)
    return false;

  ret = dpu_copy_from_mram(m_dpu, bytes, final_offset, final_size, 0);
  if (ret != DPU_OK) {
    delete[] bytes;
    return false;
  }

  memcpy(&((uint8_t *)bytes)[padding], buf, size);

  ret = dpu_copy_to_mram(m_dpu, final_offset, bytes, final_size, 0);

  delete[] bytes;
  return ret == DPU_OK;
}

bool Dpu::ReadMRAM(uint32_t offset, void *buf, size_t size) {
  std::lock_guard<std::recursive_mutex> guard(m_rank->GetLock());
  uint8_t *bytes = static_cast<uint8_t *>(buf);
  dpu_error_t ret;

  const uint32_t padding = offset & mram_aligned_mod;
  size_t final_size = (size + mram_aligned_mod + padding) & mram_aligned_mask;

  if (final_size != size) {
    const uint32_t final_offset = offset & mram_aligned_mask;
    bytes = new uint8_t[final_size];
    if (bytes == NULL)
      return false;

    ret = dpu_copy_from_mram(m_dpu, bytes, final_offset, final_size, 0);
    memcpy(buf, &bytes[padding], size);
    delete[] bytes;
  } else {
    ret = dpu_copy_from_mram(m_dpu, bytes, offset, size, 0);
  }

  return ret == DPU_OK;
}

bool Dpu::AllocIRAMBuffer(uint8_t **iram, uint32_t *iram_size) {
  dpu_description_t description = dpu_get_description(dpu_get_rank(m_dpu));
  uint32_t nb_instructions = description->memories.iram_size;
  *iram_size = nb_instructions * sizeof(dpuinstruction_t);
  *iram = new uint8_t[*iram_size];
  return *iram != NULL;
}

bool Dpu::FreeIRAMBuffer(uint8_t *iram) {
  delete[] iram;
  return true;
}

bool Dpu::GenerateSaveCore(const char *exe_path, const char *core_file_path,
                           uint8_t *iram, uint32_t iram_size) {
  struct dpu_rank_t *rank = dpu_get_rank(m_dpu);
  dpu_description_t description = dpu_get_description(rank);
  uint32_t nb_word_in_wram = description->memories.wram_size;
  uint32_t wram_size = nb_word_in_wram * sizeof(dpuword_t);
  uint32_t mram_size = description->memories.mram_size;
  uint8_t *wram = new uint8_t[wram_size];
  uint8_t *mram = new uint8_t[mram_size];

  dpu_error_t status;
  if (wram != NULL && mram != NULL) {
    status = dpu_copy_from_wram_for_dpu(m_dpu, (dpuword_t *)wram, 0,
                                        nb_word_in_wram);
    if (status != DPU_OK)
      goto dpu_generate_save_core_exit;
    status = dpu_copy_from_mram(m_dpu, mram, 0, mram_size, DPU_PRIMARY_MRAM);
    if (status != DPU_OK)
      goto dpu_generate_save_core_exit;

    status =
        dpu_create_core_dump(rank, exe_path, core_file_path, m_context->Get(),
                             wram, mram, iram, wram_size, mram_size, iram_size);
    if (status != DPU_OK)
      goto dpu_generate_save_core_exit;

  } else {
    status = DPU_ERR_SYSTEM;
  }

dpu_generate_save_core_exit:
  delete[] wram;
  delete[] mram;

  return status;
}

uint32_t *Dpu::ThreadContextRegs(int thread_index) {
  return m_context->Get()->registers + thread_index * nr_reg_per_thread;
}

uint16_t *Dpu::ThreadContextPC(int thread_index) {
  return m_context->Get()->pcs + thread_index;
}

bool *Dpu::ThreadContextZF(int thread_index) {
  return m_context->Get()->zero_flags + thread_index;
}

bool *Dpu::ThreadContextCF(int thread_index) {
  return m_context->Get()->carry_flags + thread_index;
}

bool *Dpu::ThreadRegistersHasBeenModified() {
  return &registers_has_been_modified;
}

lldb::StateType Dpu::GetThreadState(uint32_t thread_index,
                                    std::string &description,
                                    lldb::StopReason &stop_reason,
                                    bool stepping) {
  stop_reason = eStopReasonNone;
  struct _dpu_context_t *context = m_context->Get();
  bool bkp_fault = context->bkp_fault;
  bool dma_fault = context->dma_fault;
  bool mem_fault = context->mem_fault;
  uint32_t bkp_fault_thread_index = context->bkp_fault_thread_index;
  uint32_t dma_fault_thread_index = context->dma_fault_thread_index;
  uint32_t mem_fault_thread_index = context->mem_fault_thread_index;
  uint32_t bkp_fault_id = context->bkp_fault_id;
  if (bkp_fault && bkp_fault_thread_index == thread_index) {
    if (bkp_fault_id == 0) {
      stop_reason = eStopReasonBreakpoint;
      return eStateStopped;
    } else {
      description = "fault " + std::to_string(bkp_fault_id);
      stop_reason = eStopReasonException;
      return eStateCrashed;
    }
  } else if (dma_fault && dma_fault_thread_index == thread_index) {
    description = "dma fault";
    stop_reason = eStopReasonException;
    return eStateCrashed;
  } else if (mem_fault && mem_fault_thread_index == thread_index) {
    description = "memory fault";
    stop_reason = eStopReasonException;
    return eStateCrashed;
  } else if (m_context->GetPcOfThread(thread_index) != 0 &&
             m_context->ScheduledThread(thread_index) && stepping) {
    description = "stepping";
    stop_reason = eStopReasonTrace;
  } else if (m_context->GetPcOfThread(thread_index) != 0 && stepping) {
    description = "stopped";
    stop_reason = eStopReasonTrace;
  }

  if (!m_context->ScheduledThread(thread_index)) {
    return eStateSuspended;
  } else {
    return eStateStopped;
  }
}

unsigned int Dpu::GetSliceID() { return dpu_get_slice_id(m_dpu); }

unsigned int Dpu::GetDpuID() { return dpu_get_member_id(m_dpu); }

bool Dpu::SaveSliceContext(uint64_t structure_value, uint64_t slice_target,
                           dpu_bitfield_t host_mux_mram_state) {
  bool success = dpu_save_slice_context_for_dpu(m_dpu) == DPU_OK;
  if (!success)
    return false;

  m_rank->SetSliceInfo(dpu_get_slice_id(m_dpu), structure_value, slice_target,
                       host_mux_mram_state);
  return true;
}

bool Dpu::RestoreSliceContext() {
  return dpu_restore_slice_context_for_dpu(m_dpu) == DPU_OK;
}

void Dpu::SetAttachSession() { attach_session = true; }

bool Dpu::AttachSession() { return attach_session; }

bool Dpu::PrintfEnable() { return printf_enable; }

lldb::addr_t Dpu::GetOpenPrintfSequenceAddr() {
  return open_print_sequence_addr;
}
lldb::addr_t Dpu::GetClosePrintfSequenceAddr() {
  return close_print_sequence_addr;
}
