//===- Error.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of APIs in the orc-rt/support/Error.h and
// orc-rt-c/support/Error.h headers.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/support/Error.h"
#include "orc-rt-c/support/Error.h"

#include <cstring>
#include <system_error>

namespace orc_rt {

#if ORC_RT_ENABLE_EXCEPTIONS

std::string ExceptionError::toString() const noexcept {
  std::string Result;
  try {
    std::rethrow_exception(E);
  } catch (std::exception &SE) {
    Result = SE.what();
    E = std::current_exception();
  } catch (std::error_code &EC) {
    try {
      // Technically 'message' itself can throw.
      Result = EC.message();
    } catch (...) {
      Result = "std::error_code (.message() call failed)";
    }
    E = std::current_exception();
  } catch (std::string &ErrMsg) {
    Result = ErrMsg;
    E = std::current_exception();
  } catch (...) {
    Result = "C++ exception of unknown type";
    E = std::current_exception();
  }
  return Result;
}

#endif // ORC_RT_ENABLE_EXCEPTIONS

// --- C API Implementation ---

extern "C" {

ORC_RT_C_RTTI_IMPL(StringError)

void orc_rt_Error_consume(orc_rt_ErrorRef Err) noexcept {
  consumeError(unwrap(Err));
}

void orc_rt_Error_cantFail(orc_rt_ErrorRef Err) noexcept {
  cantFail(unwrap(Err));
}

char *orc_rt_Error_toString(orc_rt_ErrorRef Err) noexcept {
  return strdup(toString(unwrap(Err)).c_str());
}

void orc_rt_Error_freeErrorMessage(char *ErrMsg) noexcept { free(ErrMsg); }

orc_rt_ErrorRef orc_rt_StringError_create(const char *ErrMsg) noexcept {
  return wrap(make_error<StringError>(ErrMsg));
}

} // extern "C"

} // namespace orc_rt
