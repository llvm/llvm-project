//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def ARCOptions : OptionsStruct<prefix = "arc-">;

// ARCFrameLowering.cpp
defm : BoolField<"arc-save-restore-funclet", "true",
  "Use arc callee save/restore functions">;

// ARCOptAddrMode.cpp
defm : ValueField<"arc-kill-addr-mode", "unsigned", "0", "">;
