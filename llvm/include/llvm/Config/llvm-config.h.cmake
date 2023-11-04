/*===------- llvm/Config/llvm-config.h - llvm configuration -------*- C -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

#ifndef LLVM_CONFIG_H
#define LLVM_CONFIG_H

// This file is here for backward compatibility: please don't add to it.
//
// The configuration is sharded in a few files, it is recommended
// to include only the one you need to minimize the impact on
// incremental build and caching.
#include "llvm-config-build-llvm-dylib.h"
#include "llvm-config-build-shared-libs.h"
#include "llvm-config-enable-curl.h"
#include "llvm-config-enable-dia-sdk.h"
#include "llvm-config-enable-dump.h"
#include "llvm-config-enable-httplib.h"
#include "llvm-config-enable-plugins.h"
#include "llvm-config-enable-threads.h"
#include "llvm-config-enable-zlib.h"
#include "llvm-config-enable-zstd.h"
#include "llvm-config-force-enable-stats.h"
#include "llvm-config-force-use-old-toolchain.h"
#include "llvm-config-has-atomics.h"
#include "llvm-config-have-sysexits.h"
#include "llvm-config-have-tflite.h"
#include "llvm-config-on-unix.h"
#include "llvm-config-unreachable-optimize.h"
#include "llvm-config-use-intel-jit-events.h"
#include "llvm-config-use-oprofile.h"
#include "llvm-config-use-perf.h"
#include "llvm-config-with-z3.h"

#include "llvm-config-target-native.h"
#include "llvm-config-target-triple.h"
#include "llvm-config-version.h"

#endif
