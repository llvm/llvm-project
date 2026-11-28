//===-- Platform.cpp - liboffload platform wrapper ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Platform.h"

PlatformTy PlatformTy::create(ol_platform_handle_t Handle) {
  size_t Size = 0;
  std::string Name;
  if (!olGetPlatformInfoSize(Handle, OL_PLATFORM_INFO_NAME, &Size) &&
      Size > 0) {
    Name.resize(Size);
    if (olGetPlatformInfo(Handle, OL_PLATFORM_INFO_NAME, Size, Name.data()))
      Name.clear();
  }

  // The reported size includes the null terminator.
  while (!Name.empty() && Name.back() == '\0')
    Name.pop_back();
  if (Name.empty())
    Name = "Unknown";

  return PlatformTy(Handle, std::move(Name));
}
