//===-- llvm/PrecompiledHeaders.h - Precompiled Headers ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains common headers used within the LLVM library.
/// It is intended to be used as a precompiled header.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_PRECOMPILEDHEADERS_H
#define LLVM_PRECOMPILEDHEADERS_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/DebugInfoMetadata.h"

#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/ADL.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/bit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/EpochTracker.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMapEntry.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ilist_node_base.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/ilist_node_options.h"
#include "llvm/ADT/BitmaskEnum.h"

#include <algorithm>
#include <utility>
#include <optional>
#include <vector>
#include <string>
#include <memory>
#include <iterator>
#include <map>
#include <tuple>
#include <limits>
#include <set>
#include <system_error>
#include <functional>
#include <numeric>
#include <deque>
#include <sstream>
#include <queue>
#include <type_traits>
#include <mutex>
#include <array>
#include <list>
#include <atomic>
#include <unordered_map>
#include <bitset>
#include <new>
#include <unordered_set>
#include <iomanip>
#include <thread>
#include <variant>
#include <future>
#include <stack>
#include <chrono>
#include <initializer_list>
#include <random>

#include <cctype>
#include <cerrno>
#include <cfloat>
#include <cinttypes>
#include <climits>
#include <clocale>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cwchar>

#endif // LLVM_PRECOMPILEDHEADERS_H
