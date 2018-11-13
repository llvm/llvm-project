//===-- MachTask.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//----------------------------------------------------------------------
//
//  MachTask.h
//  debugserver
//
//  Created by Greg Clayton on 12/5/08.
//
//===----------------------------------------------------------------------===//

#ifndef __MachTask_h__
#define __MachTask_h__

#include <mach/mach.h>
#include <sys/socket.h>
#include <map>
#include <string>
#include "DNBDefs.h"
#include "MachException.h"
#include "MachVMMemory.h"
#include "PThreadMutex.h"

class MachProcess;

typedef uint64_t MachMallocEventId;

enum MachMallocEventType {
  eMachMallocEventTypeAlloc = 2,
  eMachMallocEventTypeDealloc = 4,
  eMachMallocEventTypeOther = 1
};

struct MachMallocEvent {
  mach_vm_address_t m_base_address;
  uint64_t m_size;
  MachMallocEventType m_event_type;
  MachMallocEventId m_event_id;
};

class MachTask {
public:
  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  MachTask(MachProcess *process);
  virtual ~MachTask();

  void Clear();

  kern_return_t Suspend();
  kern_return_t Resume();

  nub_size_t ReadMemory(nub_addr_t addr, nub_size_t size, void *buf);
  nub_size_t WriteMemory(nub_addr_t addr, nub_size_t size, const void *buf);
  int GetMemoryRegionInfo(nub_addr_t addr, DNBRegionInfo *region_info);
  std::string GetProfileData(DNBProfileDataScanType scanType);

  nub_addr_t AllocateMemory(nub_size_t size, uint32_t permissions);
  nub_bool_t DeallocateMemory(nub_addr_t addr);

  mach_port_t ExceptionPort() const;
  bool ExceptionPortIsValid() const;
  kern_return_t SaveExceptionPortInfo();
  kern_return_t RestoreExceptionPortInfo();
  kern_return_t ShutDownExcecptionThread();

  bool StartExceptionThread(DNBError &err);
  nub_addr_t GetDYLDAllImageInfosAddress(DNBError &err);
  kern_return_t BasicInfo(struct task_basic_info *info);
  static kern_return_t BasicInfo(task_t task, struct task_basic_info *info);
  bool IsValid() const;
  static bool IsValid(task_t task);
  static void *ExceptionThread(void *arg);
  void TaskPortChanged(task_t task);
  task_t TaskPort() const { return m_task; }
  task_t TaskPortForProcessID(DNBError &err, bool force = false);
  static task_t TaskPortForProcessID(pid_t pid, DNBError &err,
                                     uint32_t num_retries = 10,
                                     uint32_t usec_interval = 10000);

  MachProcess *Process() { return m_process; }
  const MachProcess *Process() const { return m_process; }

  nub_size_t PageSize();

protected:
  MachProcess *m_process; // The mach process that owns this MachTask
  task_t m_task;
  MachVMMemory m_vm_memory; // Special mach memory reading class that will take
                            // care of watching for page and region boundaries
  MachException::PortInfo
      m_exc_port_info;          // Saved settings for all exception ports
  pthread_t m_exception_thread; // Thread ID for the exception thread in case we
                                // need it
  mach_port_t m_exception_port; // Exception port on which we will receive child
                                // exceptions

  typedef std::map<mach_vm_address_t, size_t> allocation_collection;
  allocation_collection m_allocations;

private:
  MachTask(const MachTask &);               // Outlaw
  MachTask &operator=(const MachTask &rhs); // Outlaw
};

#endif // __MachTask_h__
