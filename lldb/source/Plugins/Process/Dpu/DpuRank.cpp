//===-- DpuRank.cpp ------------------------------------------- -*- C++ -*-===//
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
#include <dpu_config.h>
#include <dpu_debug.h>
#include <dpu_description.h>
#include <dpu_error.h>
#include <dpu_management.h>
}

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::dpu;

// -----------------------------------------------------------------------------
// DPU rank handling
// -----------------------------------------------------------------------------

DpuRank::DpuRank() : nr_threads(0), m_lock() { m_rank = NULL; }

bool DpuRank::Open(char *profile, FILE *stdout_file, bool valid) {
  std::lock_guard<std::recursive_mutex> guard(m_lock);

  int ret = dpu_get_rank_of_type(profile, &m_rank);
  if (ret != DPU_OK)
    return false;
  m_desc = dpu_get_description(m_rank);

  nr_threads = m_desc->hw.dpu.nr_of_threads;

  struct dpu_set_t rank = dpu_set_from_rank(&m_rank);
  struct dpu_set_t dpu;

  DPU_FOREACH(rank, dpu) {
    m_dpus.push_back(new Dpu(this, dpu.dpu, stdout_file, valid));
  }

  return true;
}

bool DpuRank::IsValid() { return m_rank ? true : false; }

bool DpuRank::Reset() {
  std::lock_guard<std::recursive_mutex> guard(m_lock);
  return dpu_reset_rank(m_rank) == DPU_OK;
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

Dpu *DpuRank::GetDpu(size_t index) {
  return index < m_dpus.size() ? m_dpus[index] : nullptr;
}

bool DpuRank::SaveContext() {
  return dpu_save_context_for_rank(m_rank) == DPU_OK;
}

bool DpuRank::RestoreContext() {
  return dpu_restore_context_for_rank(m_rank) == DPU_OK;
}

bool DpuRank::RestoreMuxContext() {
  return dpu_restore_mux_context_for_rank(m_rank) == DPU_OK;
}

void DpuRank::SetSliceInfo(uint32_t slice_id, uint64_t structure_value,
                           uint64_t slice_target,
                           dpu_bitfield_t host_mux_mram_state) {
  dpu_set_debug_slice_info(m_rank, slice_id, structure_value, slice_target,
                           host_mux_mram_state);
}

struct dpu_context_t *DpuRank::AllocContext() {
  struct dpu_context_t *context = new struct dpu_context_t;
  if (dpu_context_fill_from_rank(context, m_rank) != DPU_OK) {
    delete context;
    return NULL;
  }
  return context;
}

bool DpuRank::ResumeDpus() {
  for (Dpu *dpu : m_dpus) {
    if (!dpu->ResumeThreads(NULL, false))
      return false;
  }
  return true;
}

bool DpuRank::StopDpus() {
  for (Dpu *dpu : m_dpus) {
    if (!dpu->StopThreads(true))
      return false;
  }
  return true;
}

uint8_t DpuRank::GetNrCis() {
  return dpu_get_description(m_rank)->hw.topology.nr_of_control_interfaces;
}
