//===-- DarwinLogInterfaces.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
