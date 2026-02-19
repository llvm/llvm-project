//===-- Utils/OsUtils.h - Common OS utilities ------------------- C++ -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Useful utilites to interact with the OS environment in a platform independent
// way.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_UTILS_OSUTILS_H
#define OMPTARGET_UTILS_OSUTILS_H

namespace utils::os {

/// Get the name of the current executable, without the path
std::string getExecName();

} // namespace utils::os

#endif // OMPTARGET_UTILS_OSUTILS_H
