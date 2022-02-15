//===- RecordContext.h - RecordContext implementation ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions for interacting with the tablegen record
// context.
//
//===----------------------------------------------------------------------===//

namespace llvm {
namespace detail {

/// Resets the Tablegen record context and all currently parsed record data.
/// Tablegen currently relies on a lot of static data to keep track of parsed
/// records, which accumulates into static fields. This method resets all of
/// that data to enable successive executions of the tablegen parser.
/// FIXME: Ideally tablegen would use a properly scoped (non-static) context,
/// which would remove any need for managing the context in this way. In that
/// case, this method could be removed.
void resetTablegenRecordContext();

} // end namespace detail
} // end namespace llvm
