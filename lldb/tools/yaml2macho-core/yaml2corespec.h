//===-- yaml2corespec.h ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef YAML2MACHOCOREFILE_YAML2CORESPEC_H
#define YAML2MACHOCOREFILE_YAML2CORESPEC_H

#include "CoreSpec.h"

CoreSpec from_yaml(char *buf, size_t len);

#endif
