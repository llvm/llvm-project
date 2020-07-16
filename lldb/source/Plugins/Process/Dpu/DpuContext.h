//===-- DpuContext.h ----------------------------------------- -*- C++ -*- ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DpuContext_H_
#define liblldb_DpuContext_H_

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

#include "ProcessDpu.h"

extern "C" {
#include <dpu.h>
#include <dpu_debug.h>
}

namespace lldb_private {
namespace dpu {

static inline lldb::addr_t InstIdx2InstAddr(lldb::addr_t nb_of_inst) {
  return (nb_of_inst * sizeof(dpuinstruction_t)) | k_dpu_iram_base;
}

class DpuContext {
public:
  DpuContext(dpu_t *dpu, struct dpu_context_t *context, uint32_t nr_threads);
  ~DpuContext();

  struct dpu_context_t *Get();

  void UpdateContext(struct dpu_context_t *new_context);

  bool StopThreads();
  bool ResumeThreads(llvm::SmallVector<uint32_t, 8> *resume_list);
  dpu_error_t StepThread(uint32_t thread_index);

  unsigned int GetExitStatus();
  lldb::addr_t GetPcOfThread(dpu_thread_t thread);

  bool ScheduledThread(uint32_t thread);
  bool ContextReadyForResumeOrStep();

  bool DpuIsRunning(int nr_running_threads);

  bool RestoreFaultContext();

private:
  bool AddThreadInScheduling(unsigned int thread);
  void ResetScheduling();
  void ResetLastResumeThreads();
  void UpdateRunningThreads();

  dpu_t *m_dpu;
  struct dpu_context_t *m_context;
  uint32_t nr_threads;
  uint8_t *running_threads;
  uint8_t *last_resume_threads;
};

} // namespace dpu
} // namespace lldb_private

#endif // liblldb_DpuContext_H_
