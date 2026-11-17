//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def BitReaderOptions : OptionsStruct;

// BitcodeReader.cpp
defm : BoolField<"expand-constant-exprs", "false",
  "Expand constant expressions to instructions for testing purposes">;

// MetadataLoader.cpp
// Flag whether we need to import full type definitions for ThinLTO. Currently
// needed for Darwin and LLDB.
defm : BoolField<"import-full-type-definitions", "false",
  "Import full type definitions for ThinLTO">;
defm : BoolField<"ondemand-mds-loading", "true",
  "Lazy-load metadata on demand when loading bitcode for importing">;

// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"disable-ondemand-mds-loading", "ondemand-mds-loading=false">;
