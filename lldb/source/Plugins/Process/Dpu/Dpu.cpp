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
}

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::dpu;

namespace {

const ArchSpec k_dpu_arch("dpu-upmem-dpurte");

const uint32_t instruction_size_mod = sizeof(dpuinstruction_t) - 1;
const uint32_t instruction_size_mask = ~instruction_size_mod;
const uint32_t dpuword_size_mod = sizeof(dpuword_t) - 1;
const uint32_t dpuword_size_mask = ~dpuword_size_mod;
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

bool Dpu::GetPrintfSequenceAddrs() {
  dpu_runtime_context_t *runtime = dpu_get_runtime_context(m_dpu);
  open_print_sequence_addr = runtime->open_print_sequence_addr;
  close_print_sequence_addr = runtime->close_print_sequence_addr;
  printf_buffer_address = runtime->printf_buffer_address;
  printf_buffer_size = runtime->printf_buffer_size;
  printf_buffer_var_addr = runtime->printf_write_pointer_address;
  if (open_print_sequence_addr != (lldb::addr_t)LLDB_INVALID_ADDRESS &&
      close_print_sequence_addr != (lldb::addr_t)LLDB_INVALID_ADDRESS) {
    open_print_sequence_addr++; // we add 1 to avoid to be on the 'acquire'
                                // which could makes us loop forever
    open_print_sequence_addr = InstIdx2InstAddr(open_print_sequence_addr);
    close_print_sequence_addr = InstIdx2InstAddr(close_print_sequence_addr);
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

bool Dpu::LoadElf(const FileSpec &elf_file_path) {
  ModuleSP elf_mod(new Module(elf_file_path, k_dpu_arch));

  dpu_api_status_t status =
      dpu_load_individual(m_dpu, elf_file_path.GetCString());
  if (status != DPU_API_SUCCESS)
    return false;

  if (!GetPrintfSequenceAddrs())
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
  if (res != DPU_API_SUCCESS)
    return false;

  bool ignored;
  res = dpu_launch_thread_on_dpu(m_dpu, DPU_BOOT_THREAD, false, &ignored);
  if (res != DPU_API_SUCCESS)
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
    result_state = StateType::eStateExited;
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

  if (!m_context->ContextReadyForResumeOrStep())
    return false;

  int ret = DPU_API_SUCCESS;
  if (registers_has_been_modified) {
    ret = dpu_restore_context_for_dpu(m_dpu, m_context->Get());
    if (ret != DPU_API_SUCCESS)
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
      if (printf_buffer_current_idx <= printf_buffer_last_idx) {
        mram_buffer_size = printf_buffer_size;
      } else {
        mram_buffer_size = printf_buffer_current_idx - printf_buffer_last_idx;
      }
      uint8_t *mram_buffer = (uint8_t *)malloc(mram_buffer_size);
      if (mram_buffer == NULL)
        return false;
      if (mram_buffer_size == printf_buffer_size) {
        if (!ReadMRAM(printf_buffer_last_idx, mram_buffer,
                      printf_buffer_size - printf_buffer_last_idx))
          return false;
        if (!ReadMRAM(printf_buffer_address,
                      &mram_buffer[printf_buffer_last_idx],
                      printf_buffer_current_idx))
          return false;

      } else {
        if (!ReadMRAM(printf_buffer_last_idx + printf_buffer_address,
                      mram_buffer, mram_buffer_size))
          return false;
      }

      if (stdout_file != NULL) {
        if (dpulog_read_and_display_contents_of(mram_buffer, mram_buffer_size,
                                                stdout_file) != DPU_API_SUCCESS)
          return false;

        fflush(stdout_file);
      }

      free(mram_buffer);
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

  int ret = DPU_API_SUCCESS;
  if (registers_has_been_modified) {
    ret = dpu_restore_context_for_dpu(m_dpu, m_context->Get());
    if (ret != DPU_API_SUCCESS)
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

  if (ret != DPU_API_SUCCESS)
    return StateType::eStateCrashed;
  if (m_context->Get()->nr_of_running_threads == 0) {
    *exit_status = m_context->GetExitStatus();
    return StateType::eStateExited;
  }
  return StateType::eStateStopped;
}

bool Dpu::WriteWRAM(uint32_t offset, const void *buf, size_t size) {
  std::lock_guard<std::recursive_mutex> guard(m_rank->GetLock());

  dpu_api_status_t ret;
  // fast path, everything is aligned
  if (((offset & dpuword_size_mod) == 0) && ((size & dpuword_size_mod) == 0)) {
    const dpuword_t *words = static_cast<const dpuword_t *>(buf);
    ret = dpu_copy_to_wram_for_dpu(m_dpu, offset / sizeof(dpuword_t), words,
                                   size / sizeof(dpuword_t));
    return ret == DPU_API_SUCCESS;
  }

  // slow path

  // compute final_offset to start from and the final_size to use
  uint32_t final_offset = offset & dpuword_size_mask;
  size_t size_with_start_padding = final_offset + size;
  size_t final_size;
  if ((size_with_start_padding & dpuword_size_mod) != 0)
    final_size = size_with_start_padding + sizeof(dpuword_t) -
                 (final_size & dpuword_size_mask);
  else
    final_size = size_with_start_padding;

  // allocating the buffer of dpuwords to read/write from
  wram_size_t nb_dpuwords = final_size / sizeof(dpuword_t);
  dpuword_t *words = new dpuword_t[nb_dpuwords];
  if (words == NULL)
    return false;

  // reading the dpuwords
  ret =
      dpu_copy_from_wram_for_dpu(m_dpu, words, final_offset / sizeof(dpuword_t),
                                 final_size / sizeof(dpuword_t));
  if (ret != DPU_API_SUCCESS) {
    delete[] words;
    return false;
  }

  // copy the dpuwords into our buffer
  memcpy(&((uint8_t *)words)[offset - final_offset], buf, size);

  // writing the buffer
  ret = dpu_copy_to_wram_for_dpu(m_dpu, final_offset / sizeof(dpuword_t), words,
                                 final_size / sizeof(dpuword_t));

  delete[] words;
  return ret == DPU_API_SUCCESS;
}

bool Dpu::ReadWRAM(uint32_t offset, void *buf, size_t size) {
  std::lock_guard<std::recursive_mutex> guard(m_rank->GetLock());
  dpuword_t *words = static_cast<dpuword_t *>(buf);

  dpu_api_status_t ret;
  size_t final_size =
      size + sizeof(dpuword_t) - 1 + (offset & dpuword_size_mod);

  // if an aligned copy copy more than asked by the function, let's read more
  // and then copy only the wanted part into the output buffer.
  if (final_size != size) {
    iram_size_t final_size_in_dpuword = final_size / sizeof(dpuword_t);
    uint32_t final_offset = offset & dpuword_size_mask;

    words = new dpuword_t[final_size_in_dpuword];
    if (words == NULL)
      return false;

    ret = dpu_copy_from_wram_for_dpu(
        m_dpu, words, final_offset / sizeof(dpuword_t), final_size_in_dpuword);

    memcpy(buf, &words[offset - final_offset], size);
    delete[] words;

  } else {
    ret = dpu_copy_from_wram_for_dpu(m_dpu, words, offset / sizeof(dpuword_t),
                                     size / sizeof(dpuword_t));
  }
  return ret == DPU_API_SUCCESS;
}

bool Dpu::WriteIRAM(uint32_t offset, const void *buf, size_t size) {
  std::lock_guard<std::recursive_mutex> guard(m_rank->GetLock());

  dpu_api_status_t ret;
  // fast path, everything is aligned
  if (((offset & instruction_size_mod) == 0) &&
      ((size & instruction_size_mod) == 0)) {
    const dpuinstruction_t *instrs = static_cast<const dpuinstruction_t *>(buf);
    ret = dpu_copy_to_iram_for_dpu(m_dpu, offset / sizeof(dpuinstruction_t),
                                   instrs, size / sizeof(dpuinstruction_t));
    return ret == DPU_API_SUCCESS;
  }

  // slow path

  // compute final_offset to start from and the final_size to use
  uint32_t final_offset = offset & instruction_size_mask;
  size_t size_with_start_padding = final_offset + size;
  size_t final_size;
  if ((size_with_start_padding & instruction_size_mod) != 0)
    final_size = size_with_start_padding + sizeof(dpuinstruction_t) -
                 (final_size & instruction_size_mask);
  else
    final_size = size_with_start_padding;

  // allocating the buffer of instruction to read/write from
  iram_size_t nb_instructions = final_size / sizeof(dpuinstruction_t);
  dpuinstruction_t *instrs = new dpuinstruction_t[nb_instructions];
  if (instrs == NULL)
    return false;

  // reading the instructions
  ret = dpu_copy_from_iram_for_dpu(m_dpu, instrs,
                                   final_offset / sizeof(dpuinstruction_t),
                                   final_size / sizeof(dpuinstruction_t));
  if (ret != DPU_API_SUCCESS) {
    delete[] instrs;
    return false;
  }

  // copy the instructions into our buffer
  memcpy(&((uint8_t *)instrs)[offset - final_offset], buf, size);

  // writing the buffer
  ret = dpu_copy_to_iram_for_dpu(m_dpu, final_offset / sizeof(dpuinstruction_t),
                                 instrs, final_size / sizeof(dpuinstruction_t));

  delete[] instrs;
  return ret == DPU_API_SUCCESS;
}

bool Dpu::ReadIRAM(uint32_t offset, void *buf, size_t size) {
  std::lock_guard<std::recursive_mutex> guard(m_rank->GetLock());
  dpuinstruction_t *instrs = static_cast<dpuinstruction_t *>(buf);

  dpu_api_status_t ret;
  size_t final_size =
      size + sizeof(dpuinstruction_t) - 1 + (offset & instruction_size_mod);

  // if an aligned copy copy more than asked by the function, let's read more
  // and then copy only the wanted part into the output buffer.
  if (final_size != size) {
    iram_size_t final_size_in_instructions =
        final_size / sizeof(dpuinstruction_t);
    uint32_t final_offset = offset & instruction_size_mask;

    instrs = new dpuinstruction_t[final_size_in_instructions];
    if (instrs == NULL)
      return false;

    ret = dpu_copy_from_iram_for_dpu(m_dpu, instrs,
                                     final_offset / sizeof(dpuinstruction_t),
                                     final_size_in_instructions);

    memcpy(buf, &instrs[offset - final_offset], size);
    delete[] instrs;
  } else {
    ret = dpu_copy_from_iram_for_dpu(m_dpu, instrs,
                                     offset / sizeof(dpuinstruction_t),
                                     size / sizeof(dpuinstruction_t));
  }
  return ret == DPU_API_SUCCESS;
}

bool Dpu::WriteMRAM(uint32_t offset, const void *buf, size_t size) {
  std::lock_guard<std::recursive_mutex> guard(m_rank->GetLock());
  const uint8_t *bytes = static_cast<const uint8_t *>(buf);

  dpu_api_status_t ret = dpu_copy_to_mram(m_dpu, offset, bytes, size, 0);
  return ret == DPU_API_SUCCESS;
}

bool Dpu::ReadMRAM(uint32_t offset, void *buf, size_t size) {
  std::lock_guard<std::recursive_mutex> guard(m_rank->GetLock());
  uint8_t *bytes = static_cast<uint8_t *>(buf);

  dpu_api_status_t ret = dpu_copy_from_mram(m_dpu, bytes, offset, size, 0);
  return ret == DPU_API_SUCCESS;
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

  dpu_api_status_t status;
  if (wram != NULL && mram != NULL) {
    status = dpu_copy_from_wram_for_dpu(m_dpu, (dpuword_t *)wram, 0,
                                        nb_word_in_wram);
    if (status != DPU_API_SUCCESS)
      goto dpu_generate_save_core_exit;
    status = dpu_copy_from_mram(m_dpu, mram, 0, mram_size, DPU_PRIMARY_MRAM);
    if (status != DPU_API_SUCCESS)
      goto dpu_generate_save_core_exit;

    status =
        dpu_create_core_dump(rank, exe_path, core_file_path, m_context->Get(),
                             wram, mram, iram, wram_size, mram_size, iram_size);
    if (status != DPU_API_SUCCESS)
      goto dpu_generate_save_core_exit;

  } else {
    status = DPU_API_SYSTEM_ERROR;
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
  } else if (m_context->ScheduledThread(thread_index) && stepping) {
    description = "stepping";
    stop_reason = eStopReasonTrace;
  } else if (m_context->GetPcOfThread(thread_index) != 0 && stepping) {
    description = "stopped";
    stop_reason = eStopReasonTrace;
  }
  return eStateStopped;
}

unsigned int Dpu::GetSliceID() { return dpu_get_slice_id(m_dpu); }

unsigned int Dpu::GetDpuID() { return dpu_get_member_id(m_dpu); }

bool Dpu::SaveSliceContext(uint64_t structure_value, uint64_t slice_target) {
  bool success = dpu_save_slice_context_for_dpu(m_dpu) == DPU_API_SUCCESS;
  if (!success)
    return false;

  m_rank->SetSliceInfo(dpu_get_slice_id(m_dpu), structure_value, slice_target);
  return true;
}

bool Dpu::RestoreSliceContext() {
  return dpu_restore_slice_context_for_dpu(m_dpu) == DPU_API_SUCCESS;
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
