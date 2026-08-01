//===- Logging.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of APIs in the orc-rt-c/Logging.h header.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-c/Logging.h"

#include <array>
#include <cassert>
#include <cctype>
#include <cstring>

static const char *CategoryNames[] = {"General", "ControllerAccess"};
static_assert(std::size(CategoryNames) == orc_rt_log_Category_Count,
              "CategoryNames array is the wrong size");

static const char *LevelNames[] = {
    "DEBUG", "INFO", "WARNING", "ERROR", "OFF",
};
static_assert(std::size(LevelNames) == ORC_RT_LOG_LEVEL_COUNT,
              "LevelNames array is the wrong size");

const char *orc_rt_log_Category_getName(orc_rt_log_Category Cat) noexcept {
  if (Cat < 0 || Cat >= orc_rt_log_Category_Count)
    return nullptr;
  return CategoryNames[Cat];
}

const char *orc_rt_log_Level_getName(orc_rt_log_Level L) noexcept {
  if (L < 0 || L >= ORC_RT_LOG_LEVEL_COUNT)
    return nullptr;
  return LevelNames[L];
}

orc_rt_log_Level orc_rt_log_Level_parse(const char *Str) noexcept {

  auto StrMatches = [&](const char *LevelName) {
    size_t Size = strlen(LevelName) + 1; // include null terminator.

    for (size_t I = 0; I != Size; ++I) {
      unsigned char P = LevelName[I];
      unsigned char Q = Str[I];
      assert((!P || std::isupper(P)) && "Level name is not all uppercase");
      if (std::toupper(Q) != P)
        return false;
    }
    return true;
  };

  for (int I = 0; I != ORC_RT_LOG_LEVEL_COUNT; ++I)
    if (StrMatches(LevelNames[I]))
      return I;

  return -1;
}
