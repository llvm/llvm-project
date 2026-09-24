#ifndef ORC_RT_INTERNAL_SUPPORT_SYS_WINERRORTOORCERROR_H
#define ORC_RT_INTERNAL_SUPPORT_SYS_WINERRORTOORCERROR_H

#include "orc-rt/support/Error.h"

#include <string>
#include <string_view>

#include <windows.h>

namespace orc_rt::sys {

inline std::string getWindowsErrorMessage(DWORD ErrorCode) {
  LPSTR Message = nullptr;

  const DWORD Length = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, ErrorCode, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      reinterpret_cast<LPSTR>(&Message), 0, nullptr);

  if (!Length)
    return "Windows error " + std::to_string(ErrorCode);

  std::string Result(Message, Length);
  LocalFree(Message);

  while (!Result.empty() && (Result.back() == '\r' || Result.back() == '\n'))
    Result.pop_back();

  return Result;
}

inline Error generateErrorFromGetLastError(std::string_view Prefix) {
  const DWORD ErrorCode = GetLastError();

  return make_error<StringError>(std::string(Prefix) + ": " +
                                 getWindowsErrorMessage(ErrorCode));
}

} // namespace orc_rt::sys

#endif
