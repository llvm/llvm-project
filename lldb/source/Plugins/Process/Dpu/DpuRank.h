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

extern "C" {
#include <cni.h>
#include <dpu.h>
}

namespace lldb_private {
namespace dpu {

class Dpu;

class DpuRank {
public:
  DpuRank(dpu_type_t backend_type = FUNCTIONAL_SIMULATOR,
          const char *profile = "talkalot=false");
  bool Open();
  bool IsValid();
  bool Reset();
  Dpu *GetDpu(size_t index);

  dpu_description_t GetDesc() { return &m_desc; }
  int GetNrThreads() { return nr_threads; }
  dpulink_t GetLink() { return m_link; }
  std::mutex &GetLock() { return m_lock; }

private:
  dpu_type_t m_type;
  const char *m_profile;
  dpulink_t m_link;
  struct _dpu_description_t m_desc;
  int nr_threads;
  int nr_dpus;
  std::mutex m_lock; /* protect rank resources including the comm channel */
  std::vector<std::unique_ptr<Dpu>> m_dpus;
};

class Dpu {
public:
  Dpu(DpuRank *rank, dpu_slice_id_t slice_id, dpu_id_t dpu_id);
  ~Dpu();

  bool LoadElf(const FileSpec &elf_file_path);
  bool Boot();
  lldb::StateType PollStatus(unsigned int *exit_status);
  bool ResumeThreads();
  bool StopThreads();

  bool WriteWRAM(uint32_t offset, const void *buf, size_t size);
  bool ReadWRAM(uint32_t offset, void *buf, size_t size);
  bool WriteIRAM(uint32_t offset, const void *buf, size_t size);
  bool ReadIRAM(uint32_t offset, void *buf, size_t size);
  bool WriteMRAM(uint32_t offset, const void *buf, size_t size);
  bool ReadMRAM(uint32_t offset, void *buf, size_t size);

  int GetNrThreads() { return nr_threads; }

  uint32_t *ThreadContextRegs(int thread_index);
  uint16_t *ThreadContextPC(int thread_index);

  lldb::StateType GetThreadState(int thread_index, std::string &description,
                                 lldb::StopReason &stop_reason);

private:
  DpuRank *m_rank;
  dpu_slice_id_t m_slice_id;
  dpu_id_t m_dpu_id;
  int nr_threads;
  int nr_of_work_registers_per_thread;
  dpulink_t m_link;
  struct _dpu_context_t m_context;
  bool dpu_is_running = false;
  bool dpu_is_in_fault = false;
};

} // namespace dpu
} // namespace lldb_private

#endif // liblldb_Dpu_H_
