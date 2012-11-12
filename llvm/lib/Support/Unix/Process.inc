//===- Unix/Process.cpp - Unix Process Implementation --------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the generic Unix implementation of the Process class.
//
//===----------------------------------------------------------------------===//

#include "Unix.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/TimeValue.h"
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif
#ifdef HAVE_SYS_RESOURCE_H
#include <sys/resource.h>
#endif
// DragonFlyBSD, OpenBSD, and Bitrig have deprecated <malloc.h> for
// <stdlib.h> instead. Unix.h includes this for us already.
#if defined(HAVE_MALLOC_H) && !defined(__DragonFly__) && \
    !defined(__OpenBSD__) && !defined(__Bitrig__)
#include <malloc.h>
#endif
#ifdef HAVE_MALLOC_MALLOC_H
#include <malloc/malloc.h>
#endif
#ifdef HAVE_SYS_IOCTL_H
#  include <sys/ioctl.h>
#endif
#ifdef HAVE_TERMIOS_H
#  include <termios.h>
#endif

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

using namespace llvm;
using namespace sys;

unsigned
Process::GetPageSize()
{
#if defined(__CYGWIN__)
  // On Cygwin, getpagesize() returns 64k but the page size for the purposes of
  // memory protection and mmap() is 4k.
  // See http://www.cygwin.com/ml/cygwin/2009-01/threads.html#00492
  const int page_size = 0x1000;
#elif defined(HAVE_GETPAGESIZE)
  const int page_size = ::getpagesize();
#elif defined(HAVE_SYSCONF)
  long page_size = ::sysconf(_SC_PAGE_SIZE);
#else
#warning Cannot get the page size on this machine
#endif
  return static_cast<unsigned>(page_size);
}

size_t Process::GetMallocUsage() {
#if defined(HAVE_MALLINFO)
  struct mallinfo mi;
  mi = ::mallinfo();
  return mi.uordblks;
#elif defined(HAVE_MALLOC_ZONE_STATISTICS) && defined(HAVE_MALLOC_MALLOC_H)
  malloc_statistics_t Stats;
  malloc_zone_statistics(malloc_default_zone(), &Stats);
  return Stats.size_in_use;   // darwin
#elif defined(HAVE_SBRK)
  // Note this is only an approximation and more closely resembles
  // the value returned by mallinfo in the arena field.
  static char *StartOfMemory = reinterpret_cast<char*>(::sbrk(0));
  char *EndOfMemory = (char*)sbrk(0);
  if (EndOfMemory != ((char*)-1) && StartOfMemory != ((char*)-1))
    return EndOfMemory - StartOfMemory;
  else
    return 0;
#else
#warning Cannot get malloc info on this platform
  return 0;
#endif
}

size_t
Process::GetTotalMemoryUsage()
{
#if defined(HAVE_MALLINFO)
  struct mallinfo mi = ::mallinfo();
  return mi.uordblks + mi.hblkhd;
#elif defined(HAVE_MALLOC_ZONE_STATISTICS) && defined(HAVE_MALLOC_MALLOC_H)
  malloc_statistics_t Stats;
  malloc_zone_statistics(malloc_default_zone(), &Stats);
  return Stats.size_allocated;   // darwin
#elif defined(HAVE_GETRUSAGE) && !defined(__HAIKU__)
  struct rusage usage;
  ::getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss;
#else
#warning Cannot get total memory size on this platform
  return 0;
#endif
}

void
Process::GetTimeUsage(TimeValue& elapsed, TimeValue& user_time,
                      TimeValue& sys_time)
{
  elapsed = TimeValue::now();
#if defined(HAVE_GETRUSAGE)
  struct rusage usage;
  ::getrusage(RUSAGE_SELF, &usage);
  user_time = TimeValue(
    static_cast<TimeValue::SecondsType>( usage.ru_utime.tv_sec ),
    static_cast<TimeValue::NanoSecondsType>( usage.ru_utime.tv_usec *
      TimeValue::NANOSECONDS_PER_MICROSECOND ) );
  sys_time = TimeValue(
    static_cast<TimeValue::SecondsType>( usage.ru_stime.tv_sec ),
    static_cast<TimeValue::NanoSecondsType>( usage.ru_stime.tv_usec *
      TimeValue::NANOSECONDS_PER_MICROSECOND ) );
#else
#warning Cannot get usage times on this platform
  user_time.seconds(0);
  user_time.microseconds(0);
  sys_time.seconds(0);
  sys_time.microseconds(0);
#endif
}

int Process::GetCurrentUserId() {
  return getuid();
}

int Process::GetCurrentGroupId() {
  return getgid();
}

#if defined(HAVE_MACH_MACH_H) && !defined(__GNU__)
#include <mach/mach.h>
#endif

// Some LLVM programs such as bugpoint produce core files as a normal part of
// their operation. To prevent the disk from filling up, this function
// does what's necessary to prevent their generation.
void Process::PreventCoreFiles() {
#if HAVE_SETRLIMIT
  struct rlimit rlim;
  rlim.rlim_cur = rlim.rlim_max = 0;
  setrlimit(RLIMIT_CORE, &rlim);
#endif

#if defined(HAVE_MACH_MACH_H) && !defined(__GNU__)
  // Disable crash reporting on Mac OS X 10.0-10.4

  // get information about the original set of exception ports for the task
  mach_msg_type_number_t Count = 0;
  exception_mask_t OriginalMasks[EXC_TYPES_COUNT];
  exception_port_t OriginalPorts[EXC_TYPES_COUNT];
  exception_behavior_t OriginalBehaviors[EXC_TYPES_COUNT];
  thread_state_flavor_t OriginalFlavors[EXC_TYPES_COUNT];
  kern_return_t err =
    task_get_exception_ports(mach_task_self(), EXC_MASK_ALL, OriginalMasks,
                             &Count, OriginalPorts, OriginalBehaviors,
                             OriginalFlavors);
  if (err == KERN_SUCCESS) {
    // replace each with MACH_PORT_NULL.
    for (unsigned i = 0; i != Count; ++i)
      task_set_exception_ports(mach_task_self(), OriginalMasks[i],
                               MACH_PORT_NULL, OriginalBehaviors[i],
                               OriginalFlavors[i]);
  }

  // Disable crash reporting on Mac OS X 10.5
  signal(SIGABRT, _exit);
  signal(SIGILL,  _exit);
  signal(SIGFPE,  _exit);
  signal(SIGSEGV, _exit);
  signal(SIGBUS,  _exit);
#endif
}

bool Process::StandardInIsUserInput() {
  return FileDescriptorIsDisplayed(STDIN_FILENO);
}

bool Process::StandardOutIsDisplayed() {
  return FileDescriptorIsDisplayed(STDOUT_FILENO);
}

bool Process::StandardErrIsDisplayed() {
  return FileDescriptorIsDisplayed(STDERR_FILENO);
}

bool Process::FileDescriptorIsDisplayed(int fd) {
#if HAVE_ISATTY
  return isatty(fd);
#else
  // If we don't have isatty, just return false.
  return false;
#endif
}

static unsigned getColumns(int FileID) {
  // If COLUMNS is defined in the environment, wrap to that many columns.
  if (const char *ColumnsStr = std::getenv("COLUMNS")) {
    int Columns = std::atoi(ColumnsStr);
    if (Columns > 0)
      return Columns;
  }

  unsigned Columns = 0;

#if defined(HAVE_SYS_IOCTL_H) && defined(HAVE_TERMIOS_H)
  // Try to determine the width of the terminal.
  struct winsize ws;
  if (ioctl(FileID, TIOCGWINSZ, &ws) == 0)
    Columns = ws.ws_col;
#endif

  return Columns;
}

unsigned Process::StandardOutColumns() {
  if (!StandardOutIsDisplayed())
    return 0;

  return getColumns(1);
}

unsigned Process::StandardErrColumns() {
  if (!StandardErrIsDisplayed())
    return 0;

  return getColumns(2);
}

static bool terminalHasColors() {
  if (const char *term = std::getenv("TERM")) {
    // Most modern terminals support ANSI escape sequences for colors.
    // We could check terminfo, or have a list of known terms that support
    // colors, but that would be overkill.
    // The user can always ask for no colors by setting TERM to dumb, or
    // using a commandline flag.
    return strcmp(term, "dumb") != 0;
  }
  return false;
}

bool Process::FileDescriptorHasColors(int fd) {
  // A file descriptor has colors if it is displayed and the terminal has
  // colors.
  return FileDescriptorIsDisplayed(fd) && terminalHasColors();
}

bool Process::StandardOutHasColors() {
  return FileDescriptorHasColors(STDOUT_FILENO);
}

bool Process::StandardErrHasColors() {
  return FileDescriptorHasColors(STDERR_FILENO);
}

bool Process::ColorNeedsFlush() {
  // No, we use ANSI escape sequences.
  return false;
}

#define COLOR(FGBG, CODE, BOLD) "\033[0;" BOLD FGBG CODE "m"

#define ALLCOLORS(FGBG,BOLD) {\
    COLOR(FGBG, "0", BOLD),\
    COLOR(FGBG, "1", BOLD),\
    COLOR(FGBG, "2", BOLD),\
    COLOR(FGBG, "3", BOLD),\
    COLOR(FGBG, "4", BOLD),\
    COLOR(FGBG, "5", BOLD),\
    COLOR(FGBG, "6", BOLD),\
    COLOR(FGBG, "7", BOLD)\
  }

static const char colorcodes[2][2][8][10] = {
 { ALLCOLORS("3",""), ALLCOLORS("3","1;") },
 { ALLCOLORS("4",""), ALLCOLORS("4","1;") }
};

const char *Process::OutputColor(char code, bool bold, bool bg) {
  return colorcodes[bg?1:0][bold?1:0][code&7];
}

const char *Process::OutputBold(bool bg) {
  return "\033[1m";
}

const char *Process::OutputReverse() {
  return "\033[7m";
}

const char *Process::ResetColor() {
  return "\033[0m";
}

#if !defined(HAVE_ARC4RANDOM)
static unsigned GetRandomNumberSeed() {
  // Attempt to get the initial seed from /dev/urandom, if possible.
  if (FILE *RandomSource = ::fopen("/dev/urandom", "r")) {
    unsigned seed;
    int count = ::fread((void *)&seed, sizeof(seed), 1, RandomSource);
    ::fclose(RandomSource);

    // Return the seed if the read was successful.
    if (count == 1)
      return seed;
  }

  // Otherwise, swizzle the current time and the process ID to form a reasonable
  // seed.
  TimeValue Now = llvm::TimeValue::now();
  return hash_combine(Now.seconds(), Now.nanoseconds(), ::getpid());
}
#endif

unsigned llvm::sys::Process::GetRandomNumber() {
#if defined(HAVE_ARC4RANDOM)
  return arc4random();
#else
  static int x = (::srand(GetRandomNumberSeed()), 0);
  (void)x;
  return ::rand();
#endif
}
