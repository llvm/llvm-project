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
}

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::dpu;

// -----------------------------------------------------------------------------
// DPU rank handling
// -----------------------------------------------------------------------------

DpuRank::DpuRank() : nr_threads(0), m_lock() { m_rank = NULL; }

bool DpuRank::Open(char *profile, FILE *stdout_file) {
  std::lock_guard<std::recursive_mutex> guard(m_lock);

  int ret = dpu_get_rank_of_type(profile, &m_rank);
  if (ret != DPU_API_SUCCESS)
    return false;
  m_desc = dpu_get_description(m_rank);

  nr_threads = m_desc->dpu.nr_of_threads;

  struct dpu_t *dpu;
  DPU_FOREACH(m_rank, dpu) {
    m_dpus.push_back(new Dpu(this, dpu, stdout_file));
  }

  return true;
}

bool DpuRank::IsValid() { return m_rank ? true : false; }

bool DpuRank::Reset() {
  std::lock_guard<std::recursive_mutex> guard(m_lock);
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

Dpu *DpuRank::GetDpu(size_t index) {
  return index < m_dpus.size() ? m_dpus[index] : nullptr;
}

void DpuRank::SetSliceInfo(uint32_t slice_id, uint64_t structure_value,
                           uint64_t slice_target,
                           dpu_bitfield_t host_mux_mram_state) {
  dpu_set_debug_slice_info(m_rank, slice_id, structure_value, slice_target,
                           host_mux_mram_state);
}

struct _dpu_context_t *DpuRank::AllocContext() {
  return dpu_alloc_dpu_context(m_rank);
}
