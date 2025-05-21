//===-- BreakpointLocationsHandler..cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "JSONUtils.h"
#include "RequestHandler.h"
#include <vector>

namespace lldb_dap {

/// The `breakpointLocations` request returns all possible locations for source
/// breakpoints in a given range. Clients should only call this request if the
/// corresponding capability `supportsBreakpointLocationsRequest` is true.
llvm::Expected<protocol::BreakpointLocationsResponseBody>
BreakpointLocationsRequestHandler::Run(
    const protocol::BreakpointLocationsArguments &args) const {
  std::string path = args.source.path.value_or("");
  uint32_t start_line = args.line;
  uint32_t start_column = args.column.value_or(LLDB_INVALID_COLUMN_NUMBER);
  uint32_t end_line = args.endLine.value_or(start_line);
  uint32_t end_column =
      args.endColumn.value_or(std::numeric_limits<uint32_t>::max());

  lldb::SBFileSpec file_spec(path.c_str(), true);
  lldb::SBSymbolContextList compile_units =
      dap.target.FindCompileUnits(file_spec);

  // Find all relevant lines & columns
  llvm::SmallVector<std::pair<uint32_t, uint32_t>, 8> locations;
  for (uint32_t c_idx = 0, c_limit = compile_units.GetSize(); c_idx < c_limit;
       ++c_idx) {
    const lldb::SBCompileUnit &compile_unit =
        compile_units.GetContextAtIndex(c_idx).GetCompileUnit();
    if (!compile_unit.IsValid())
      continue;
    lldb::SBFileSpec primary_file_spec = compile_unit.GetFileSpec();

    // Go through the line table and find all matching lines / columns
    for (uint32_t l_idx = 0, l_limit = compile_unit.GetNumLineEntries();
         l_idx < l_limit; ++l_idx) {
      lldb::SBLineEntry line_entry = compile_unit.GetLineEntryAtIndex(l_idx);

      // Filter by line / column
      uint32_t line = line_entry.GetLine();
      if (line < start_line || line > end_line)
        continue;
      uint32_t column = line_entry.GetColumn();
      if (column == LLDB_INVALID_COLUMN_NUMBER)
        continue;
      if (line == start_line && column < start_column)
        continue;
      if (line == end_line && column > end_column)
        continue;

      // Make sure we are in the right file.
      // We might have a match on line & column range and still
      // be in the wrong file, e.g. for included files.
      // Given that the involved pointers point into LLDB's string pool,
      // we can directly compare the `const char*` pointers.
      if (line_entry.GetFileSpec().GetFilename() !=
              primary_file_spec.GetFilename() ||
          line_entry.GetFileSpec().GetDirectory() !=
              primary_file_spec.GetDirectory())
        continue;

      locations.emplace_back(line, column);
    }
  }

  // The line entries are sorted by addresses, but we must return the list
  // ordered by line / column position.
  std::sort(locations.begin(), locations.end());
  locations.erase(llvm::unique(locations), locations.end());

  std::vector<protocol::BreakpointLocation> breakpoint_locations;
  for (auto &l : locations) {
    protocol::BreakpointLocation lc;
    lc.line = l.first;
    lc.column = l.second;
    breakpoint_locations.push_back(std::move(lc));
  }

  return protocol::BreakpointLocationsResponseBody{
      /*breakpoints=*/std::move(breakpoint_locations)};
}

} // namespace lldb_dap
