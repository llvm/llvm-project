//===-- SwiftFrameRecognizers.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftFrameRecognizers_h_
#define liblldb_SwiftFrameRecognizers_h_

#include "lldb/Target/StackFrameRecognizer.h"

namespace lldb_private {

void RegisterSwiftFrameRecognizers(Process &process);
 
} // namespace lldb_private

#endif
