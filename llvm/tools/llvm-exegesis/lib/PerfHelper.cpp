//===-- PerfHelper.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PerfHelper.h"
#include "llvm/Config/config.h"
#include "llvm/Support/raw_ostream.h"
#ifdef HAVE_LIBPFM
#include "perfmon/perf_event.h"
#include "perfmon/pfmlib.h"
#include "perfmon/pfmlib_perf_event.h"
#endif
#include <cassert>

namespace exegesis {
namespace pfm {

#ifdef HAVE_LIBPFM
static bool isPfmError(int Code) { return Code != PFM_SUCCESS; }
#endif

bool pfmInitialize() {
#ifdef HAVE_LIBPFM
  return isPfmError(pfm_initialize());
#else
  return true;
#endif
}

void pfmTerminate() {
#ifdef HAVE_LIBPFM
  pfm_terminate();
#endif
}

PerfEvent::~PerfEvent() {
#ifdef HAVE_LIBPFM
  delete Attr;
  ;
#endif
}

PerfEvent::PerfEvent(PerfEvent &&Other)
    : EventString(std::move(Other.EventString)),
      FullQualifiedEventString(std::move(Other.FullQualifiedEventString)),
      Attr(Other.Attr) {
  Other.Attr = nullptr;
}

PerfEvent::PerfEvent(llvm::StringRef PfmEventString)
    : EventString(PfmEventString.str()), Attr(nullptr) {
#ifdef HAVE_LIBPFM
  char *Fstr = nullptr;
  pfm_perf_encode_arg_t Arg = {};
  Attr = new perf_event_attr();
  Arg.attr = Attr;
  Arg.fstr = &Fstr;
  Arg.size = sizeof(pfm_perf_encode_arg_t);
  const int Result = pfm_get_os_event_encoding(EventString.c_str(), PFM_PLM3,
                                               PFM_OS_PERF_EVENT, &Arg);
  if (isPfmError(Result)) {
    // We don't know beforehand which counters are available (e.g. 6 uops ports
    // on Sandybridge but 8 on Haswell) so we report the missing counter without
    // crashing.
    llvm::errs() << pfm_strerror(Result) << " - cannot create event "
                 << EventString << "\n";
  }
  if (Fstr) {
    FullQualifiedEventString = Fstr;
    free(Fstr);
  }
#endif
}

llvm::StringRef PerfEvent::name() const { return EventString; }

bool PerfEvent::valid() const { return !FullQualifiedEventString.empty(); }

const perf_event_attr *PerfEvent::attribute() const { return Attr; }

llvm::StringRef PerfEvent::getPfmEventString() const {
  return FullQualifiedEventString;
}

#ifdef HAVE_LIBPFM
Counter::Counter(const PerfEvent &Event) {
  assert(Event.valid());
  const pid_t Pid = 0;    // measure current process/thread.
  const int Cpu = -1;     // measure any processor.
  const int GroupFd = -1; // no grouping of counters.
  const uint32_t Flags = 0;
  perf_event_attr AttrCopy = *Event.attribute();
  FileDescriptor = perf_event_open(&AttrCopy, Pid, Cpu, GroupFd, Flags);
  if (FileDescriptor == -1) {
    llvm::errs() << "Unable to open event, make sure your kernel allows user "
                    "space perf monitoring.\nYou may want to try:\n$ sudo sh "
                    "-c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'\n";
  }
  assert(FileDescriptor != -1 && "Unable to open event");
}

Counter::~Counter() { close(FileDescriptor); }

void Counter::start() { ioctl(FileDescriptor, PERF_EVENT_IOC_RESET, 0); }

void Counter::stop() { ioctl(FileDescriptor, PERF_EVENT_IOC_DISABLE, 0); }

int64_t Counter::read() const {
  int64_t Count = 0;
  ssize_t ReadSize = ::read(FileDescriptor, &Count, sizeof(Count));
  if (ReadSize != sizeof(Count)) {
    Count = -1;
    llvm::errs() << "Failed to read event counter\n";
  }
  return Count;
}

#else

Counter::Counter(const PerfEvent &Event) {}

Counter::~Counter() = default;

void Counter::start() {}

void Counter::stop() {}

int64_t Counter::read() const { return 42; }

#endif

} // namespace pfm
} // namespace exegesis
