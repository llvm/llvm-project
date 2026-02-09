//===- Error.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of APIs in the orc-rt/Error.h and
// orc-rt-c/Error.h headers.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/Error.h"
#include "orc-rt-c/Error.h"

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

extern "C" orc_rt_Error_TypeId orc_rt_Error_getTypeId(orc_rt_ErrorRef Err) {
  assert(Err && "Err must not be null");
  return reinterpret_cast<ErrorInfoBase *>(Err)->dynamicClassID();
}

extern "C" void orc_rt_Error_consume(orc_rt_ErrorRef Err) {
  consumeError(unwrap(Err));
}

extern "C" void orc_rt_Error_cantFail(orc_rt_ErrorRef Err) {
  cantFail(unwrap(Err));
}

extern "C" char *orc_rt_Error_toString(orc_rt_ErrorRef Err) {
  return strdup(toString(unwrap(Err)).c_str());
}

extern "C" void orc_rt_Error_freeErrorMessage(char *ErrMsg) { free(ErrMsg); }

extern "C" orc_rt_Error_TypeId orc_rt_StringError_getTypeId(void) {
  return StringError::classID();
}

extern "C" orc_rt_ErrorRef orc_rt_StringError_create(const char *ErrMsg) {
  return wrap(make_error<StringError>(ErrMsg));
}

} // namespace orc_rt
