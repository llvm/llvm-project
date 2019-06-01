//===- polly/ScheduleTreeTransform.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Make changes to isl's schedule tree data structure.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCHEDULETREETRANSFORM_H
#define POLLY_SCHEDULETREETRANSFORM_H

#include "isl/isl-noexceptions.h"

namespace polly {
/// Hoist all domains from extension into the root domain node, such that there
/// are no more extension nodes (which isl does not support for some
/// operations). This assumes that domains added by to extension nodes do not
/// overlap.
isl::schedule hoistExtensionNodes(isl::schedule Sched);
} // namespace polly

#endif // POLLY_SCHEDULETREETRANSFORM_H
