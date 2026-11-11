//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def XCoreOptions : OptionsStruct<prefix = "xcore-">;

defm : ValueField<"xcore-max-threads", "unsigned", "8", "Maximum number of threads (for emulation thread-local storage)">;
