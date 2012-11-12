//===- Win32/Process.cpp - Win32 Process Implementation ------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of the Process class.
//
//===----------------------------------------------------------------------===//

#include "Windows.h"
#include <psapi.h>
#include <malloc.h>
#include <io.h>
#include <direct.h>

#ifdef __MINGW32__
 #if (HAVE_LIBPSAPI != 1)
  #error "libpsapi.a should be present"
 #endif
#else
 #pragma comment(lib, "psapi.lib")
#endif

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code
//===          and must not be UNIX code
//===----------------------------------------------------------------------===//

#ifdef __MINGW32__
// This ban should be lifted when MinGW 1.0+ has defined this value.
#  define _HEAPOK (-2)
#endif

namespace llvm {
using namespace sys;

// This function retrieves the page size using GetSystemInfo and is present
// solely so it can be called once in Process::GetPageSize to initialize the
// static variable PageSize.
inline unsigned GetPageSizeOnce() {
  // NOTE: A 32-bit application running under WOW64 is supposed to use
  // GetNativeSystemInfo.  However, this interface is not present prior
  // to Windows XP so to use it requires dynamic linking.  It is not clear
  // how this affects the reported page size, if at all.  One could argue
  // that LLVM ought to run as 64-bits on a 64-bit system, anyway.
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return static_cast<unsigned>(info.dwPageSize);
}

unsigned
Process::GetPageSize() {
  static const unsigned PageSize = GetPageSizeOnce();
  return PageSize;
}

size_t
Process::GetMallocUsage()
{
  _HEAPINFO hinfo;
  hinfo._pentry = NULL;

  size_t size = 0;

  while (_heapwalk(&hinfo) == _HEAPOK)
    size += hinfo._size;

  return size;
}

size_t
Process::GetTotalMemoryUsage()
{
  PROCESS_MEMORY_COUNTERS pmc;
  GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
  return pmc.PagefileUsage;
}

void
Process::GetTimeUsage(
  TimeValue& elapsed, TimeValue& user_time, TimeValue& sys_time)
{
  elapsed = TimeValue::now();

  uint64_t ProcCreate, ProcExit, KernelTime, UserTime;
  GetProcessTimes(GetCurrentProcess(), (FILETIME*)&ProcCreate,
                  (FILETIME*)&ProcExit, (FILETIME*)&KernelTime,
                  (FILETIME*)&UserTime);

  // FILETIME's are # of 100 nanosecond ticks (1/10th of a microsecond)
  user_time.seconds( UserTime / 10000000 );
  user_time.nanoseconds( unsigned(UserTime % 10000000) * 100 );
  sys_time.seconds( KernelTime / 10000000 );
  sys_time.nanoseconds( unsigned(KernelTime % 10000000) * 100 );
}

int Process::GetCurrentUserId()
{
  return 65536;
}

int Process::GetCurrentGroupId()
{
  return 65536;
}

// Some LLVM programs such as bugpoint produce core files as a normal part of
// their operation. To prevent the disk from filling up, this configuration item
// does what's necessary to prevent their generation.
void Process::PreventCoreFiles() {
  // Windows doesn't do core files, but it does do modal pop-up message
  // boxes.  As this method is used by bugpoint, preventing these pop-ups
  // is the moral equivalent of suppressing core files.
  SetErrorMode(SEM_FAILCRITICALERRORS |
               SEM_NOGPFAULTERRORBOX |
               SEM_NOOPENFILEERRORBOX);
}

bool Process::StandardInIsUserInput() {
  return FileDescriptorIsDisplayed(0);
}

bool Process::StandardOutIsDisplayed() {
  return FileDescriptorIsDisplayed(1);
}

bool Process::StandardErrIsDisplayed() {
  return FileDescriptorIsDisplayed(2);
}

bool Process::FileDescriptorIsDisplayed(int fd) {
  DWORD Mode;  // Unused
  return (GetConsoleMode((HANDLE)_get_osfhandle(fd), &Mode) != 0);
}

unsigned Process::StandardOutColumns() {
  unsigned Columns = 0;
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi))
    Columns = csbi.dwSize.X;
  return Columns;
}

unsigned Process::StandardErrColumns() {
  unsigned Columns = 0;
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  if (GetConsoleScreenBufferInfo(GetStdHandle(STD_ERROR_HANDLE), &csbi))
    Columns = csbi.dwSize.X;
  return Columns;
}

// The terminal always has colors.
bool Process::FileDescriptorHasColors(int fd) {
  return FileDescriptorIsDisplayed(fd);
}

bool Process::StandardOutHasColors() {
  return FileDescriptorHasColors(1);
}

bool Process::StandardErrHasColors() {
  return FileDescriptorHasColors(2);
}

namespace {
class DefaultColors
{
  private:
    WORD defaultColor;
  public:
    DefaultColors()
     :defaultColor(GetCurrentColor()) {}
    static unsigned GetCurrentColor() {
      CONSOLE_SCREEN_BUFFER_INFO csbi;
      if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi))
        return csbi.wAttributes;
      return 0;
    }
    WORD operator()() const { return defaultColor; }
};

DefaultColors defaultColors;
}

bool Process::ColorNeedsFlush() {
  return true;
}

const char *Process::OutputBold(bool bg) {
  WORD colors = DefaultColors::GetCurrentColor();
  if (bg)
    colors |= BACKGROUND_INTENSITY;
  else
    colors |= FOREGROUND_INTENSITY;
  SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), colors);
  return 0;
}

const char *Process::OutputColor(char code, bool bold, bool bg) {
  WORD colors;
  if (bg) {
    colors = ((code&1) ? BACKGROUND_RED : 0) |
      ((code&2) ? BACKGROUND_GREEN : 0 ) |
      ((code&4) ? BACKGROUND_BLUE : 0);
    if (bold)
      colors |= BACKGROUND_INTENSITY;
  } else {
    colors = ((code&1) ? FOREGROUND_RED : 0) |
      ((code&2) ? FOREGROUND_GREEN : 0 ) |
      ((code&4) ? FOREGROUND_BLUE : 0);
    if (bold)
      colors |= FOREGROUND_INTENSITY;
  }
  SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), colors);
  return 0;
}

static WORD GetConsoleTextAttribute(HANDLE hConsoleOutput) {
  CONSOLE_SCREEN_BUFFER_INFO info;
  GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info);
  return info.wAttributes;
}

const char *Process::OutputReverse() {
  const WORD attributes
   = GetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE));

  const WORD foreground_mask = FOREGROUND_BLUE | FOREGROUND_GREEN |
    FOREGROUND_RED | FOREGROUND_INTENSITY;
  const WORD background_mask = BACKGROUND_BLUE | BACKGROUND_GREEN |
    BACKGROUND_RED | BACKGROUND_INTENSITY;
  const WORD color_mask = foreground_mask | background_mask;

  WORD new_attributes =
    ((attributes & FOREGROUND_BLUE     )?BACKGROUND_BLUE     :0) |
    ((attributes & FOREGROUND_GREEN    )?BACKGROUND_GREEN    :0) |
    ((attributes & FOREGROUND_RED      )?BACKGROUND_RED      :0) |
    ((attributes & FOREGROUND_INTENSITY)?BACKGROUND_INTENSITY:0) |
    ((attributes & BACKGROUND_BLUE     )?FOREGROUND_BLUE     :0) |
    ((attributes & BACKGROUND_GREEN    )?FOREGROUND_GREEN    :0) |
    ((attributes & BACKGROUND_RED      )?FOREGROUND_RED      :0) |
    ((attributes & BACKGROUND_INTENSITY)?FOREGROUND_INTENSITY:0) |
    0;
  new_attributes = (attributes & ~color_mask) | (new_attributes & color_mask);

  SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), new_attributes);
  return 0;
}

const char *Process::ResetColor() {
  SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), defaultColors());
  return 0;
}

}
