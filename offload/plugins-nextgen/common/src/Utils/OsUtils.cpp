#ifdef _WIN32
#include <windows.h>
#else
#include <limits.h>
#include <unistd.h>
#endif

#include "llvm/Support/Path.h"

namespace utils::os {

std::string getExecName() {
#if defined(_WIN32)
  char Buffer[MAX_PATH];
  GetModuleFileNameA(nullptr, Buffer, MAX_PATH);
#else
  char Buffer[PATH_MAX];
  ssize_t Len = readlink("/proc/self/exe", Buffer, sizeof(Buffer) - 1);
  if (Len == -1)
    return "unknown";
  Buffer[Len] = '\0';
#endif
  llvm::StringRef Path(Buffer);

  if (!Path.empty())
    return llvm::sys::path::filename(Path).str();

  return "unknown";
}

} // namespace utils::os