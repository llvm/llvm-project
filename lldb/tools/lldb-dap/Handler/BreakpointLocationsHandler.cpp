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

namespace lldb_dap {

// "BreakpointLocationsRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "The `breakpointLocations` request returns all possible
//     locations for source breakpoints in a given range.\nClients should only
//     call this request if the corresponding capability
//     `supportsBreakpointLocationsRequest` is true.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "breakpointLocations" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/BreakpointLocationsArguments"
//       }
//     },
//     "required": [ "command" ]
//   }]
// },
// "BreakpointLocationsArguments": {
//   "type": "object",
//   "description": "Arguments for `breakpointLocations` request.",
//   "properties": {
//     "source": {
//       "$ref": "#/definitions/Source",
//       "description": "The source location of the breakpoints; either
//       `source.path` or `source.sourceReference` must be specified."
//     },
//     "line": {
//       "type": "integer",
//       "description": "Start line of range to search possible breakpoint
//       locations in. If only the line is specified, the request returns all
//       possible locations in that line."
//     },
//     "column": {
//       "type": "integer",
//       "description": "Start position within `line` to search possible
//       breakpoint locations in. It is measured in UTF-16 code units and the
//       client capability `columnsStartAt1` determines whether it is 0- or
//       1-based. If no column is given, the first position in the start line is
//       assumed."
//     },
//     "endLine": {
//       "type": "integer",
//       "description": "End line of range to search possible breakpoint
//       locations in. If no end line is given, then the end line is assumed to
//       be the start line."
//     },
//     "endColumn": {
//       "type": "integer",
//       "description": "End position within `endLine` to search possible
//       breakpoint locations in. It is measured in UTF-16 code units and the
//       client capability `columnsStartAt1` determines whether it is 0- or
//       1-based. If no end column is given, the last position in the end line
//       is assumed."
//     }
//   },
//   "required": [ "source", "line" ]
// },
// "BreakpointLocationsResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to `breakpointLocations` request.\nContains
//     possible locations for source breakpoints.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "breakpoints": {
//             "type": "array",
//             "items": {
//               "$ref": "#/definitions/BreakpointLocation"
//             },
//             "description": "Sorted set of possible breakpoint locations."
//           }
//         },
//         "required": [ "breakpoints" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// },
// "BreakpointLocation": {
//   "type": "object",
//   "description": "Properties of a breakpoint location returned from the
//   `breakpointLocations` request.",
//   "properties": {
//     "line": {
//       "type": "integer",
//       "description": "Start line of breakpoint location."
//     },
//     "column": {
//       "type": "integer",
//       "description": "The start position of a breakpoint location. Position
//       is measured in UTF-16 code units and the client capability
//       `columnsStartAt1` determines whether it is 0- or 1-based."
//     },
//     "endLine": {
//       "type": "integer",
//       "description": "The end line of breakpoint location if the location
//       covers a range."
//     },
//     "endColumn": {
//       "type": "integer",
//       "description": "The end position of a breakpoint location (if the
//       location covers a range). Position is measured in UTF-16 code units and
//       the client capability `columnsStartAt1` determines whether it is 0- or
//       1-based."
//     }
//   },
//   "required": [ "line" ]
// },
void BreakpointLocationsRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  FillResponse(request, response);
  auto *arguments = request.getObject("arguments");
  auto *source = arguments->getObject("source");
  std::string path = GetString(source, "path").str();
  const auto start_line = GetInteger<uint64_t>(arguments, "line")
                              .value_or(LLDB_INVALID_LINE_NUMBER);
  const auto start_column = GetInteger<uint64_t>(arguments, "column")
                                .value_or(LLDB_INVALID_COLUMN_NUMBER);
  const auto end_line =
      GetInteger<uint64_t>(arguments, "endLine").value_or(start_line);
  const auto end_column = GetInteger<uint64_t>(arguments, "endColumn")
                              .value_or(std::numeric_limits<uint64_t>::max());

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
  locations.erase(std::unique(locations.begin(), locations.end()),
                  locations.end());

  llvm::json::Array locations_json;
  for (auto &l : locations) {
    llvm::json::Object location;
    location.try_emplace("line", l.first);
    location.try_emplace("column", l.second);
    locations_json.emplace_back(std::move(location));
  }

  llvm::json::Object body;
  body.try_emplace("breakpoints", std::move(locations_json));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
