//===-- DpuRank.h -------------------------------------------- -*- C++ -*- ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DpuRank_H_
#define liblldb_DpuRank_H_

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

#include "Dpu.h"

namespace lldb_private {
namespace dpu {

class DpuRank {
public:
  DpuRank();
  bool Open(char *profile, FILE *stdout_fd);
  bool IsValid();
  bool Reset();
  Dpu *GetDpu(size_t index);

  dpu_description_t GetDesc() { return m_desc; }
  int GetNrThreads() { return nr_threads; }
  std::recursive_mutex &GetLock() { return m_lock; }

  Dpu *GetDpuFromSliceIdAndDpuId(unsigned int slice_id, unsigned int dpu_id);

  void SetSliceInfo(uint32_t slice_id, uint64_t structure_value,
                    uint64_t slice_target);

  struct _dpu_context_t *AllocContext();

private:
  dpu_rank_t *m_rank;
  dpu_description_t m_desc;
  int nr_threads;
  std::recursive_mutex
      m_lock; /* protect rank resources including the comm channel */
  std::vector<Dpu *> m_dpus;
};

} // namespace dpu
} // namespace lldb_private

#endif // liblldb_DpuRank_H_
