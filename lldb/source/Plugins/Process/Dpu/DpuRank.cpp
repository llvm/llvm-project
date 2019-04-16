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
#include <cni.h>
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
  m_link = nullptr;
}

bool DpuRank::Open() {
  std::lock_guard<std::mutex> guard(m_lock);

  int ret = dpu_cni_get_rank_of_type(m_type, m_profile, &m_link);
  fprintf(stderr, "get_rank=>%d\n", ret);
  if (ret != DPU_CNI_SUCCESS)
    return false;
  dpu_cni_get_target_description(m_link, &m_desc);

  fprintf(stderr, "rank desc threads %d dpus %d\n", m_desc.dpu.nr_of_threads,
          m_desc.topology.nr_of_control_interfaces *
              m_desc.topology.nr_of_dpus_per_control_interface);
  nr_threads = m_desc.dpu.nr_of_threads;
  nr_dpus = m_desc.topology.nr_of_control_interfaces *
            m_desc.topology.nr_of_dpus_per_control_interface;

  m_dpus.reserve(nr_dpus);
  for (int id = 0; id < nr_dpus; id++) {
    dpu_slice_id_t slice_id =
        (dpu_slice_id_t)(id / m_desc.topology.nr_of_dpus_per_control_interface);
    dpu_id_t dpu_id =
        (dpu_id_t)(id % m_desc.topology.nr_of_dpus_per_control_interface);
    m_dpus.push_back(llvm::make_unique<Dpu>(this, slice_id, dpu_id));
  }

  return true;
}

bool DpuRank::IsValid() { return m_link ? true : false; }

bool DpuRank::Reset() {
  std::lock_guard<std::mutex> guard(m_lock);
  return dpu_cni_reset_for_all(m_link) == DPU_CNI_SUCCESS;
}

Dpu *DpuRank::GetDpu(size_t index) {
  return index < m_dpus.size() ? m_dpus[index].get() : nullptr;
}

Dpu::Dpu(DpuRank *rank, dpu_slice_id_t slice_id, dpu_id_t dpu_id)
    : m_rank(rank), m_slice_id(slice_id), m_dpu_id(dpu_id) {
  nr_threads = m_rank->GetNrThreads();
  nr_of_work_registers_per_thread =
      rank->GetDesc()->dpu.nr_of_work_registers_per_thread;
  m_link = m_rank->GetLink();

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

  ObjectFile *exe = elf_mod->GetObjectFile();
  if (exe == nullptr)
    return false;

  const SectionList *sections = exe->GetSectionList();
  Section *section_text =
      sections->FindSectionByName(ConstString(".text")).get();
  Section *section_data =
      sections->FindSectionByName(ConstString(".data")).get();
  if (!section_text || !section_data)
    return false;

  DataExtractor text_content;
  exe->ReadSectionData(section_text, text_content);
  fprintf(stderr, "text +%ld\n", text_content.GetByteSize());
  DataExtractor data_content;
  exe->ReadSectionData(section_data, data_content);
  fprintf(stderr, "data +%ld\n", data_content.GetByteSize());

  {
    std::lock_guard<std::mutex> guard(m_rank->GetLock());

    int res = dpu_cni_copy_to_iram_for_dpu(
        m_link, m_slice_id, m_dpu_id, 0,
        reinterpret_cast<const dpuinstruction_t *>(text_content.GetDataStart()),
        text_content.GetByteSize() / sizeof(dpuinstruction_t));
    if (res != DPU_CNI_SUCCESS)
      return false;
    res = dpu_cni_copy_to_wram_for_dpu(
        m_link, m_slice_id, m_dpu_id, 0,
        reinterpret_cast<const dpuword_t *>(data_content.GetDataStart()),
        data_content.GetByteSize() / sizeof(dpuword_t));
    if (res != DPU_CNI_SUCCESS)
      return false;
  }

  return true;
}

bool Dpu::Boot() {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());

  int res = dpu_cni_preexecution_for_dpu(m_link, m_slice_id, m_dpu_id);
  if (res != DPU_CNI_SUCCESS)
    return false;

  bool ignored;
  res = dpu_cni_launch_thread_for_dpu(m_link, m_slice_id, m_dpu_id, 0, false,
                                      &ignored);
  if (res != DPU_CNI_SUCCESS)
    return false;

  return true;
}

StateType Dpu::PollStatus(unsigned int *exit_status) {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());
  StateType result_state = StateType::eStateRunning;

  if (!dpu_is_running)
    return StateType::eStateInvalid;

  int res = dpu_cni_poll_for_dpu(m_link, m_slice_id, m_dpu_id, &dpu_is_running,
                                 &dpu_is_in_fault);
  if (dpu_is_in_fault) {
    dpu_is_running = false;
    result_state = StateType::eStateStopped;
  } else if (!dpu_is_running) {
    result_state = StateType::eStateExited;
  } else {
    return StateType::eStateRunning;
  }

  dpu_cni_initialize_fault_process_for_dpu(m_link, m_slice_id, m_dpu_id,
                                           &m_context);
  dpu_cni_extract_context_for_dpu(m_link, m_slice_id, m_dpu_id, &m_context);
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

  int ret = DPU_CNI_SUCCESS;
  ret |= dpu_cni_initialize_fault_process_for_dpu(m_link, m_slice_id, m_dpu_id,
                                                  &m_context);
  ret |=
      dpu_cni_extract_context_for_dpu(m_link, m_slice_id, m_dpu_id, &m_context);
  return ret == DPU_CNI_SUCCESS;
}

bool Dpu::ResumeThreads() {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());

  int ret = DPU_CNI_SUCCESS;
  ret |= dpu_cni_clear_fault_on_all(m_link);
  ret |= dpu_cni_finalize_fault_process_for_dpu(m_link, m_slice_id, m_dpu_id,
                                                &m_context);

  dpu_is_running = true;
  return ret == DPU_CNI_SUCCESS;
}

bool Dpu::WriteWRAM(uint32_t offset, const void *buf, size_t size) {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());
  const dpuword_t *words = static_cast<const dpuword_t *>(buf);

  dpu_cni_status_t ret = dpu_cni_copy_to_wram_for_dpu(
      m_link, m_slice_id, m_dpu_id, offset / sizeof(dpuword_t), words,
      size / sizeof(dpuword_t));
  return ret == DPU_CNI_SUCCESS;
}

bool Dpu::ReadWRAM(uint32_t offset, void *buf, size_t size) {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());
  dpuword_t *words = static_cast<dpuword_t *>(buf);

  dpu_cni_status_t ret = dpu_cni_copy_from_wram_for_dpu(
      m_link, m_slice_id, m_dpu_id, words, offset / sizeof(dpuword_t),
      size / sizeof(dpuword_t));
  return ret == DPU_CNI_SUCCESS;
}

bool Dpu::WriteIRAM(uint32_t offset, const void *buf, size_t size) {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());
  const dpuinstruction_t *instrs = static_cast<const dpuinstruction_t *>(buf);

  dpu_cni_status_t ret = dpu_cni_copy_to_iram_for_dpu(
      m_link, m_slice_id, m_dpu_id, offset / sizeof(dpuinstruction_t), instrs,
      size / sizeof(dpuinstruction_t));
  return ret == DPU_CNI_SUCCESS;
}

bool Dpu::ReadIRAM(uint32_t offset, void *buf, size_t size) {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());
  dpuinstruction_t *instrs = static_cast<dpuinstruction_t *>(buf);

  dpu_cni_status_t ret = dpu_cni_copy_from_iram_for_dpu(
      m_link, m_slice_id, m_dpu_id, instrs, offset / sizeof(dpuinstruction_t),
      size / sizeof(dpuinstruction_t));
  return ret == DPU_CNI_SUCCESS;
}

bool Dpu::WriteMRAM(uint32_t offset, const void *buf, size_t size) {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());
  const uint8_t *bytes = static_cast<const uint8_t *>(buf);

  dpu_cni_status_t ret = dpu_cni_copy_to_mram_number_for_dpu(
      m_link, m_slice_id, m_dpu_id, offset, bytes, size, 0);
  return ret == DPU_CNI_SUCCESS;
}

bool Dpu::ReadMRAM(uint32_t offset, void *buf, size_t size) {
  std::lock_guard<std::mutex> guard(m_rank->GetLock());
  uint8_t *bytes = static_cast<uint8_t *>(buf);

  dpu_cni_status_t ret = dpu_cni_copy_from_mram_number_for_dpu(
      m_link, m_slice_id, m_dpu_id, bytes, offset, size, 0);
  return ret == DPU_CNI_SUCCESS;
}

uint32_t *Dpu::ThreadContextRegs(int thread_index) {
  return m_context.registers + thread_index * nr_of_work_registers_per_thread;
}

uint16_t *Dpu::ThreadContextPC(int thread_index) {
  return m_context.pcs + thread_index;
}

lldb::StateType Dpu::GetThreadState(int thread_index, std::string &description,
                                    lldb::StopReason &stop_reason) {
  if (m_context.bkp_fault && m_context.bkp_fault_thread_index == thread_index) {
    description = "breakpoint hit";
    stop_reason = eStopReasonBreakpoint;
    return eStateStopped;
  } else if (m_context.dma_fault && m_context.dma_fault_thread_index == thread_index) {
    description = "dma fault";
    stop_reason = eStopReasonBreakpoint;
    return eStateStopped;
  } else if (m_context.mem_fault && m_context.mem_fault_thread_index == thread_index) {
    description = "memory fault";
    stop_reason = eStopReasonBreakpoint;
    return eStateStopped;
  } else if (m_context.scheduling[thread_index] != 0xff) {
    description = "suspended";
    stop_reason = eStopReasonSignal;
    return eStateStopped;
  } else if (m_context.pcs[thread_index] != 0) {
    description = "stopped";
    stop_reason = eStopReasonNone;
    return eStateStopped;
  } else {
    stop_reason = eStopReasonNone;
    return eStateExited;
  }
}
