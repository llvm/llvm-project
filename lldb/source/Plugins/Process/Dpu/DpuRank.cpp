//===-- Dpu.cpp ----------------------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

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

#include "RegisterContextDpu.h"

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
// DPU rank handling
// -----------------------------------------------------------------------------

DpuRank::DpuRank() : nr_threads(0), nr_dpus(0), m_lock() { m_rank = NULL; }

bool DpuRank::Open(char *profile) {
  std::lock_guard<std::mutex> guard(m_lock);

  int ret = dpu_get_rank_of_type(profile, &m_rank);
  if (ret != DPU_API_SUCCESS)
    return false;
  m_desc = dpu_get_description(m_rank);

  nr_threads = m_desc->dpu.nr_of_threads;
  nr_dpus = m_desc->topology.nr_of_control_interfaces *
            m_desc->topology.nr_of_dpus_per_control_interface;

  m_dpus.reserve(nr_dpus);
  for (int id = 0; id < nr_dpus; id++) {
    dpu_slice_id_t slice_id = (dpu_slice_id_t)(
        id / m_desc->topology.nr_of_dpus_per_control_interface);
    dpu_id_t dpu_id =
        (dpu_id_t)(id % m_desc->topology.nr_of_dpus_per_control_interface);
    struct dpu_t *dpu = dpu_get(m_rank, slice_id, dpu_id);
    if (dpu != NULL && dpu_is_enabled(dpu)) {
      m_dpus.push_back(new Dpu(this, dpu));
    }
  }

  return true;
}

bool DpuRank::IsValid() { return m_rank ? true : false; }

bool DpuRank::Reset() {
  std::lock_guard<std::mutex> guard(m_lock);
  return dpu_reset_rank(m_rank) == DPU_API_SUCCESS;
}

Dpu *DpuRank::GetDpuFromSliceIdAndDpuId(unsigned int slice_id,
                                        unsigned int dpu_id) {
  for (Dpu *dpu : m_dpus) {
    if (dpu->GetSliceID() == slice_id && dpu->GetDpuID() == dpu_id) {
      return dpu;
    }
  }
  return nullptr;
}

bool DpuRank::StopDpus() {
  for (Dpu *dpu : m_dpus) {
    bool success = dpu->StopThreadsUnlock(true);
    if (!success)
      return false;
  }
  return true;
}

bool DpuRank::ResumeDpus() {
  for (Dpu *dpu : m_dpus) {
    bool success = dpu->ResumeThreads(false);
    if (!success)
      return false;
  }
  return true;
}

Dpu *DpuRank::GetDpu(size_t index) {
  return index < m_dpus.size() ? m_dpus[index] : nullptr;
}

void DpuRank::SetSliceInfo(uint32_t slice_id, uint64_t structure_value,
                           uint64_t slice_target) {
  dpu_set_structure_value_and_slice_target(m_rank, slice_id, structure_value,
                                           slice_target);
}

Dpu::Dpu(DpuRank *rank, dpu_t *dpu) : m_rank(rank), m_dpu(dpu) {
  nr_threads = m_rank->GetNrThreads();
  nr_of_work_registers_per_thread =
      rank->GetDesc()->dpu.nr_of_work_registers_per_thread;

  m_context = dpu_alloc_dpu_context(dpu_get_rank(dpu));
}

Dpu::~Dpu() { dpu_free_dpu_context(m_context); }

bool Dpu::LoadElf(const FileSpec &elf_file_path) {
  ModuleSP elf_mod(new Module(elf_file_path, k_dpu_arch));

  dpu_api_status_t status =
      dpu_load_individual(m_dpu, elf_file_path.GetCString());
  return status == DPU_API_SUCCESS;
}

bool Dpu::Boot() {
  dpu_context_t tmp_context;
  // Extract a potential context from the dpu structure (that could have been
  // created by the dpu_loader).
  dpu_pop_debug_context(m_dpu, &tmp_context);

  // If the dpu contains a context, it means that a core file has been loaded.
  // In this case, do not do the boot sequence, the dpu is already in a state
  // ready to be resume with 'dpu_finalize_fault_process_for_dpu'.
  if (tmp_context != NULL) {
    // Free our previously allocated context and get the one created by the
    // dpu_loader.
    dpu_free_dpu_context(m_context);
    m_context = tmp_context;
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

bool Dpu::StopThreadsUnlock(bool force) {
  if (!dpu_is_running && !force)
    return true;
  dpu_is_running = false;

  for (dpu_thread_t each_thread = 0; each_thread < nr_threads; ++each_thread) {
    m_context->scheduling[each_thread] = 0xFF;
  }
  m_context->nr_of_running_threads = 0;
  m_context->bkp_fault = false;
  m_context->dma_fault = false;
  m_context->mem_fault = false;

  int ret = DPU_API_SUCCESS;
  ret |= dpu_initialize_fault_process_for_dpu(m_dpu, m_context);
  ret |= dpu_extract_context_for_dpu(m_dpu, m_context);
  return ret == DPU_API_SUCCESS;
}

static void SetExitStatus(unsigned int *exit_status,
                          struct _dpu_context_t *context) {
  *exit_status = context->registers[lldb_private::r0_dpu];
}

StateType Dpu::PollStatus(unsigned int *exit_status) {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());
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

  if (!StopThreadsUnlock(true)) {
    return StateType::eStateCrashed;
  }
  // Needs to be after StopThreadsUnlock to make sure that context is up to
  // date.
  SetExitStatus(exit_status, m_context);

  return result_state;
}

bool Dpu::StopThreads() {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());

  return StopThreadsUnlock();
}

static bool IsContextReadyForResumeOrStep(struct _dpu_context_t *context) {
  context->bkp_fault = false;
  return !(context->dma_fault || context->mem_fault);
}

bool Dpu::ResumeThreads(bool allowed_polling) {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());

  if (!IsContextReadyForResumeOrStep(m_context))
    return false;

  int ret = DPU_API_SUCCESS;
  if (registers_has_been_modified) {
    ret |= dpu_restore_context_for_dpu(m_dpu, m_context);
    registers_has_been_modified = false;
  }
  ret |= dpu_finalize_fault_process_for_dpu(m_dpu, m_context);

  if (allowed_polling)
    dpu_is_running = true;

  return ret == DPU_API_SUCCESS;
}

StateType Dpu::StepThread(uint32_t thread_index, unsigned int *exit_status) {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());

  if (!IsContextReadyForResumeOrStep(m_context))
    return StateType::eStateCrashed;

  // If the thread is not in the scheduling list, do not try to step it.
  // This behavior is expected as lldb can ask to step one thread and resume all
  // the other, which result in stepping all the thread contained in the
  // scheduling list.
  if (m_context->scheduling[thread_index] == 0xff)
    return StateType::eStateStopped;

  int ret = DPU_API_SUCCESS;
  if (registers_has_been_modified) {
    ret |= dpu_restore_context_for_dpu(m_dpu, m_context);
    registers_has_been_modified = false;
  }
  ret |=
      dpu_execute_thread_step_in_fault_for_dpu(m_dpu, thread_index, m_context);
  ret |= dpu_extract_context_for_dpu(m_dpu, m_context);

  if (ret != DPU_API_SUCCESS)
    return StateType::eStateCrashed;
  if (m_context->nr_of_running_threads == 0) {
    SetExitStatus(exit_status, m_context);
    return StateType::eStateExited;
  }
  return StateType::eStateStopped;
}

bool Dpu::WriteWRAM(uint32_t offset, const void *buf, size_t size) {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());

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
  std::lock_guard<std::mutex> guard(m_rank->GetLock());
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
  std::lock_guard<std::mutex> guard(m_rank->GetLock());

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
  std::lock_guard<std::mutex> guard(m_rank->GetLock());
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
  std::lock_guard<std::mutex> guard(m_rank->GetLock());
  const uint8_t *bytes = static_cast<const uint8_t *>(buf);

  dpu_api_status_t ret = dpu_copy_to_mram(m_dpu, offset, bytes, size, 0);
  return ret == DPU_API_SUCCESS;
}

bool Dpu::ReadMRAM(uint32_t offset, void *buf, size_t size) {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());
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
        dpu_create_core_dump(rank, exe_path, core_file_path, m_context, wram,
                             mram, iram, wram_size, mram_size, iram_size);
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
  return m_context->registers + thread_index * nr_of_work_registers_per_thread;
}

uint16_t *Dpu::ThreadContextPC(int thread_index) {
  return m_context->pcs + thread_index;
}

bool *Dpu::ThreadContextZF(int thread_index) {
  return m_context->zero_flags + thread_index;
}

bool *Dpu::ThreadContextCF(int thread_index) {
  return m_context->carry_flags + thread_index;
}

bool *Dpu::ThreadRegistersHasBeenModified() {
  return &registers_has_been_modified;
}

lldb::StateType Dpu::GetThreadState(int thread_index, std::string &description,
                                    lldb::StopReason &stop_reason,
                                    bool stepping) {
  stop_reason = eStopReasonNone;
  if (m_context->bkp_fault &&
      m_context->bkp_fault_thread_index == thread_index) {
    stop_reason = eStopReasonBreakpoint;
    return eStateStopped;
  } else if (m_context->dma_fault &&
             m_context->dma_fault_thread_index == thread_index) {
    description = "dma fault";
    stop_reason = eStopReasonException;
    return eStateCrashed;
  } else if (m_context->mem_fault &&
             m_context->mem_fault_thread_index == thread_index) {
    description = "memory fault";
    stop_reason = eStopReasonException;
    return eStateCrashed;
  } else if (m_context->scheduling[thread_index] != 0xff && stepping) {
    description = "stepping";
    stop_reason = eStopReasonTrace;
  } else if (m_context->pcs[thread_index] != 0 && stepping) {
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
