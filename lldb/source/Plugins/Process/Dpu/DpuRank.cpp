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

}

// -----------------------------------------------------------------------------
// DPU rank handling
// -----------------------------------------------------------------------------

DpuRank::DpuRank(dpu_type_t backend_type, const char *profile)
    : m_type(backend_type), m_profile(profile), nr_threads(0), nr_dpus(0),
      m_lock() {
  m_rank = NULL;
}

bool DpuRank::Open() {
  std::lock_guard<std::mutex> guard(m_lock);

  int ret = dpu_get_rank_of_type(m_type, m_profile, &m_rank);
  if (ret != DPU_API_SUCCESS)
    return false;
  m_desc = dpu_get_description(m_rank);

  nr_threads = m_desc->dpu.nr_of_threads;
  nr_dpus = m_desc->topology.nr_of_control_interfaces *
            m_desc->topology.nr_of_dpus_per_control_interface;

  m_dpus.reserve(nr_dpus);
  for (int id = 0; id < nr_dpus; id++) {
    dpu_slice_id_t slice_id =
        (dpu_slice_id_t)(id / m_desc->topology.nr_of_dpus_per_control_interface);
    dpu_id_t dpu_id =
        (dpu_id_t)(id % m_desc->topology.nr_of_dpus_per_control_interface);
    m_dpus.push_back(llvm::make_unique<Dpu>(this, dpu_get(m_rank, slice_id, dpu_id)));
  }

  return true;
}

bool DpuRank::IsValid() { return m_rank ? true : false; }

bool DpuRank::Reset() {
  std::lock_guard<std::mutex> guard(m_lock);
  return dpu_reset_rank(m_rank) == DPU_API_SUCCESS;
}

Dpu *DpuRank::GetDpu(size_t index) {
  return index < m_dpus.size() ? m_dpus[index].get() : nullptr;
}

Dpu::Dpu(DpuRank *rank, dpu_t *dpu) : m_rank(rank), m_dpu(dpu) {
  nr_threads = m_rank->GetNrThreads();
  nr_of_work_registers_per_thread =
      rank->GetDesc()->dpu.nr_of_work_registers_per_thread;

  uint32_t nr_of_atomic_bits = rank->GetDesc()->dpu.nr_of_atomic_bits;
  m_context.registers =
      new uint32_t[nr_of_work_registers_per_thread * nr_threads];
  m_context.scheduling = new uint8_t[nr_threads];
  m_context.pcs = new iram_addr_t[nr_threads];
  m_context.zero_flags = new bool[nr_threads];
  m_context.carry_flags = new bool[nr_threads];
  m_context.atomic_register = new bool[nr_of_atomic_bits];
}

Dpu::~Dpu() {
  delete m_context.registers;
  delete m_context.scheduling;
  delete m_context.pcs;
  delete m_context.zero_flags;
  delete m_context.carry_flags;
  delete m_context.atomic_register;
}

bool Dpu::LoadElf(const FileSpec &elf_file_path) {
  ModuleSP elf_mod(new Module(elf_file_path, k_dpu_arch));

  dpu_api_status_t status =
      dpu_load_individual(m_dpu, elf_file_path.GetCString());
  return status == DPU_API_SUCCESS;
}

bool Dpu::Boot() {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());

  int res = dpu_custom_for_dpu(m_dpu, DPU_COMMAND_DPU_PREEXECUTION, NULL);
  if (res != DPU_API_SUCCESS)
    return false;

  bool ignored;
  res = dpu_launch_thread_on_dpu(m_dpu, DPU_BOOT_THREAD, false, &ignored);
  if (res != DPU_API_SUCCESS)
    return false;

  return true;
}

StateType Dpu::PollStatus(unsigned int *exit_status) {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());
  bool dpu_is_in_fault;
  StateType result_state = StateType::eStateRunning;

  if (!dpu_is_running)
    return StateType::eStateInvalid;

  dpu_poll_dpu(m_dpu, &dpu_is_running, &dpu_is_in_fault);
  if (dpu_is_in_fault) {
    dpu_is_running = false;
    result_state = StateType::eStateStopped;
    dpu_initialize_fault_process_for_dpu(m_dpu, &m_context);
  } else if (!dpu_is_running) {
    result_state = StateType::eStateExited;
  } else {
    return StateType::eStateRunning;
  }

  dpu_extract_context_for_dpu(m_dpu, &m_context);
  *exit_status = m_context.registers[lldb_private::r21_dpu];

  return result_state;
}

bool Dpu::StopThreads() {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());

  dpu_is_running = false;

  for (dpu_thread_t each_thread = 0; each_thread < nr_threads; ++each_thread) {
    m_context.scheduling[each_thread] = 0xFF;
  }
  m_context.nr_of_running_threads = 0;
  m_context.bkp_fault = false;
  m_context.dma_fault = false;
  m_context.mem_fault = false;

  int ret = DPU_API_SUCCESS;
  ret |= dpu_initialize_fault_process_for_dpu(m_dpu, &m_context);
  ret |= dpu_extract_context_for_dpu(m_dpu, &m_context);
  return ret == DPU_API_SUCCESS;
}

bool Dpu::ResumeThreads() {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());

  if (m_context.dma_fault || m_context.mem_fault) {
    return false;
  }
  m_context.bkp_fault = false;

  int ret = DPU_API_SUCCESS;
  ret |= dpu_clear_fault_on_dpu(m_dpu);
  ret |= dpu_finalize_fault_process_for_dpu(m_dpu, &m_context);

  dpu_is_running = true;
  return ret == DPU_API_SUCCESS;
}

bool Dpu::StepThread(uint32_t thread_index) {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());
  m_context.bkp_fault = false;

  int ret = DPU_API_SUCCESS;
  ret |= dpu_execute_thread_step_in_fault_for_dpu(m_dpu, thread_index, &m_context);

  return ret == DPU_API_SUCCESS;
}

bool Dpu::WriteWRAM(uint32_t offset, const void *buf, size_t size) {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());
  const dpuword_t *words = static_cast<const dpuword_t *>(buf);

  dpu_api_status_t ret = dpu_copy_to_wram_for_dpu(
      m_dpu, offset / sizeof(dpuword_t), words, size / sizeof(dpuword_t));
  return ret == DPU_API_SUCCESS;
}

bool Dpu::ReadWRAM(uint32_t offset, void *buf, size_t size) {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());
  dpuword_t *words = static_cast<dpuword_t *>(buf);

  dpu_api_status_t ret = dpu_copy_from_wram_for_dpu(
      m_dpu, words, offset / sizeof(dpuword_t), size / sizeof(dpuword_t));
  return ret == DPU_API_SUCCESS;
}

bool Dpu::WriteIRAM(uint32_t offset, const void *buf, size_t size) {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());
  const dpuinstruction_t *instrs = static_cast<const dpuinstruction_t *>(buf);

  dpu_api_status_t ret =
      dpu_copy_to_iram_for_dpu(m_dpu, offset / sizeof(dpuinstruction_t), instrs,
                               size / sizeof(dpuinstruction_t));
  return ret == DPU_API_SUCCESS;
}

bool Dpu::ReadIRAM(uint32_t offset, void *buf, size_t size) {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());
  dpuinstruction_t *instrs = static_cast<dpuinstruction_t *>(buf);

  dpu_api_status_t ret = dpu_copy_from_iram_for_dpu(
      m_dpu, instrs, offset / sizeof(dpuinstruction_t),
      size / sizeof(dpuinstruction_t));
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

uint32_t *Dpu::ThreadContextRegs(int thread_index) {
  return m_context.registers + thread_index * nr_of_work_registers_per_thread;
}

uint16_t *Dpu::ThreadContextPC(int thread_index) {
  return m_context.pcs + thread_index;
}

bool *Dpu::ThreadContextZF(int thread_index) {
  return m_context.zero_flags + thread_index;
}

bool *Dpu::ThreadContextCF(int thread_index) {
  return m_context.carry_flags + thread_index;
}

lldb::StateType Dpu::GetThreadState(int thread_index, std::string &description,
                                    lldb::StopReason &stop_reason) {
  if (m_context.bkp_fault && m_context.bkp_fault_thread_index == thread_index) {
    description = "breakpoint hit";
  } else if (m_context.dma_fault && m_context.dma_fault_thread_index == thread_index) {
    description = "dma fault";
  } else if (m_context.mem_fault && m_context.mem_fault_thread_index == thread_index) {
    description = "memory fault";
  } else if (m_context.scheduling[thread_index] != 0xff) {
    description = "suspended";
  } else if (m_context.pcs[thread_index] != 0) {
    description = "stopped";
  }
  stop_reason = eStopReasonNone;
  return eStateStopped;
}
