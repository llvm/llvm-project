//===- Error.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of APIs in the orc-rt/Error.h header.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/Error.h"

#include <system_error>

namespace orc_rt {

#if ORC_RT_ENABLE_EXCEPTIONS

std::string ExceptionError::toString() const noexcept {
  std::string Result;
  try {
    try {
      std::rethrow_exception(E);
    } catch (std::exception &E) {
      Result = E.what();
      throw;
    } catch (std::error_code &EC) {
      try {
        // Technically 'message' itself can throw.
        Result = EC.message();
      } catch (...) {
        Result = "std::error_code (.message() call failed)";
        throw EC;
      }
      throw;
    } catch (std::string &ErrMsg) {
      Result = ErrMsg;
      throw;
    } catch (...) {
      Result = "c++ exception of unknown type";
      throw;
    }
  } catch (...) {
    E = std::current_exception();
  }
  return Result;
}

#endif // ORC_RT_ENABLE_EXCEPTIONS

} // namespace orc_rt
