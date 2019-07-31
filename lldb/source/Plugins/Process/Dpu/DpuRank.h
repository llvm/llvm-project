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

  dpu_description_t GetDesc() { return m_desc; }
  int GetNrThreads() { return nr_threads; }
  std::mutex &GetLock() { return m_lock; }

private:
  dpu_type_t m_type;
  const char *m_profile;
  dpu_rank_t *m_rank;;
  dpu_description_t m_desc;
  int nr_threads;
  int nr_dpus;
  std::mutex m_lock; /* protect rank resources including the comm channel */
  std::vector<std::unique_ptr<Dpu>> m_dpus;
};

class Dpu {
public:
  Dpu(DpuRank *rank, dpu_t *dpu);
  ~Dpu();

  bool LoadElf(const FileSpec &elf_file_path);
  bool Boot();
  lldb::StateType PollStatus(unsigned int *exit_status);
  bool ResumeThreads();
  bool StopThreads();
  bool StopThreadsUnlock();

  lldb::StateType StepThread(uint32_t thread_index, unsigned int *exit_status);

  bool WriteWRAM(uint32_t offset, const void *buf, size_t size);
  bool ReadWRAM(uint32_t offset, void *buf, size_t size);
  bool WriteIRAM(uint32_t offset, const void *buf, size_t size);
  bool ReadIRAM(uint32_t offset, void *buf, size_t size);
  bool WriteMRAM(uint32_t offset, const void *buf, size_t size);
  bool ReadMRAM(uint32_t offset, void *buf, size_t size);

  int GetNrThreads() { return nr_threads; }

  uint32_t *ThreadContextRegs(int thread_index);
  uint16_t *ThreadContextPC(int thread_index);
  bool *ThreadContextZF(int thread_index);
  bool *ThreadContextCF(int thread_index);

  lldb::StateType GetThreadState(int thread_index, std::string &description,
                                 lldb::StopReason &stop_reason, bool stepping);

private:
  DpuRank *m_rank;
  dpu_t *m_dpu;
  int nr_threads;
  int nr_of_work_registers_per_thread;
  struct _dpu_context_t m_context;
  bool dpu_is_running = false;
};

} // namespace dpu
} // namespace lldb_private

#endif // liblldb_Dpu_H_
