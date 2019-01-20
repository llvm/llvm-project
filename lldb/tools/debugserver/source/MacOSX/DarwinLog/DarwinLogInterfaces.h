//===-- DarwinLogInterfaces.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DarwinLogInterfaces_h
#define DarwinLogInterfaces_h

#include <memory>

class ActivityStore;

class LogFilter;
using LogFilterSP = std::shared_ptr<LogFilter>;

class LogFilterChain;
using LogFilterChainSP = std::shared_ptr<LogFilterChain>;

class LogMessage;

#endif /* DarwinLogInterfaces_h */
