//===-- RegisterFlags.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/RegisterFlags.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/ADT/StringExtras.h"

#include <numeric>
#include <optional>

using namespace lldb_private;

RegisterFlags::Field::Field(std::string name, unsigned start, unsigned end)
    : m_name(std::move(name)), m_start(start), m_end(end) {
  assert(m_start <= m_end && "Start bit must be <= end bit.");
}

void RegisterFlags::Field::log(Log *log) const {
  LLDB_LOG(log, "  Name: \"{0}\" Start: {1} End: {2}", m_name.c_str(), m_start,
           m_end);
}

bool RegisterFlags::Field::Overlaps(const Field &other) const {
  unsigned overlap_start = std::max(GetStart(), other.GetStart());
  unsigned overlap_end = std::min(GetEnd(), other.GetEnd());
  return overlap_start <= overlap_end;
}

unsigned RegisterFlags::Field::PaddingDistance(const Field &other) const {
  assert(!Overlaps(other) &&
         "Cannot get padding distance for overlapping fields.");
  assert((other < (*this)) && "Expected fields in MSB to LSB order.");

  // If they don't overlap they are either next to each other or separated
  // by some number of bits.

  // Where left will be the MSB and right will be the LSB.
  unsigned lhs_start = GetStart();
  unsigned rhs_end = other.GetStart() + other.GetSizeInBits() - 1;

  if (*this < other) {
    lhs_start = other.GetStart();
    rhs_end = GetStart() + GetSizeInBits() - 1;
  }

  return lhs_start - rhs_end - 1;
}

void RegisterFlags::SetFields(const std::vector<Field> &fields) {
  // We expect that the XML processor will discard anything describing flags but
  // with no fields.
  assert(fields.size() && "Some fields must be provided.");

  // We expect that these are unsorted but do not overlap.
  // They could fill the register but may have gaps.
  std::vector<Field> provided_fields = fields;

  m_fields.clear();
  m_fields.reserve(provided_fields.size());

  // ProcessGDBRemote should have sorted these in descending order already.
  assert(std::is_sorted(provided_fields.rbegin(), provided_fields.rend()));

  // Build a new list of fields that includes anonymous (empty name) fields
  // wherever there is a gap. This will simplify processing later.
  std::optional<Field> previous_field;
  unsigned register_msb = (m_size * 8) - 1;
  for (auto field : provided_fields) {
    if (previous_field) {
      unsigned padding = previous_field->PaddingDistance(field);
      if (padding) {
        // -1 to end just before the previous field.
        unsigned end = previous_field->GetStart() - 1;
        // +1 because if you want to pad 1 bit you want to start and end
        // on the same bit.
        m_fields.push_back(Field("", field.GetEnd() + 1, end));
      }
    } else {
      // This is the first field. Check that it starts at the register's MSB.
      if (field.GetEnd() != register_msb)
        m_fields.push_back(Field("", field.GetEnd() + 1, register_msb));
    }
    m_fields.push_back(field);
    previous_field = field;
  }

  // The last field may not extend all the way to bit 0.
  if (previous_field && previous_field->GetStart() != 0)
    m_fields.push_back(Field("", 0, previous_field->GetStart() - 1));
}

RegisterFlags::RegisterFlags(std::string id, unsigned size,
                             const std::vector<Field> &fields)
    : m_id(std::move(id)), m_size(size) {
  SetFields(fields);
}

void RegisterFlags::log(Log *log) const {
  LLDB_LOG(log, "ID: \"{0}\" Size: {1}", m_id.c_str(), m_size);
  for (const Field &field : m_fields)
    field.log(log);
}

static StreamString FormatCell(const StreamString &content,
                               unsigned column_width) {
  unsigned pad = column_width - content.GetString().size();
  std::string pad_l;
  std::string pad_r;
  if (pad) {
    pad_l = std::string(pad / 2, ' ');
    pad_r = std::string((pad / 2) + (pad % 2), ' ');
  }

  StreamString aligned;
  aligned.Printf("|%s%s%s", pad_l.c_str(), content.GetString().data(),
                 pad_r.c_str());
  return aligned;
}

static void EmitTable(std::string &out, std::array<std::string, 3> &table) {
  // Close the table.
  for (std::string &line : table)
    line += '|';

  out += std::accumulate(table.begin() + 1, table.end(), table.front(),
                         [](std::string lhs, const auto &rhs) {
                           return std::move(lhs) + "\n" + rhs;
                         });
}

std::string RegisterFlags::AsTable(uint32_t max_width) const {
  std::string table;
  // position / gridline / name
  std::array<std::string, 3> lines;
  uint32_t current_width = 0;

  for (const RegisterFlags::Field &field : m_fields) {
    StreamString position;
    if (field.GetEnd() == field.GetStart())
      position.Printf(" %d ", field.GetEnd());
    else
      position.Printf(" %d-%d ", field.GetEnd(), field.GetStart());

    StreamString name;
    name.Printf(" %s ", field.GetName().c_str());

    unsigned column_width = position.GetString().size();
    unsigned name_width = name.GetString().size();
    if (name_width > column_width)
      column_width = name_width;

    // If the next column would overflow and we have already formatted at least
    // one column, put out what we have and move to a new table on the next line
    // (+1 here because we need to cap the ends with '|'). If this is the first
    // column, just let it overflow and we'll wrap next time around. There's not
    // much we can do with a very small terminal.
    if (current_width && ((current_width + column_width + 1) >= max_width)) {
      EmitTable(table, lines);
      // Blank line between each.
      table += "\n\n";

      for (std::string &line : lines)
        line.clear();
      current_width = 0;
    }

    StreamString aligned_position = FormatCell(position, column_width);
    lines[0] += aligned_position.GetString();
    StreamString grid;
    grid << '|' << std::string(column_width, '-');
    lines[1] += grid.GetString();
    StreamString aligned_name = FormatCell(name, column_width);
    lines[2] += aligned_name.GetString();

    // +1 for the left side '|'.
    current_width += column_width + 1;
  }

  // If we didn't overflow and still have table to print out.
  if (lines[0].size())
    EmitTable(table, lines);

  return table;
}

void RegisterFlags::ToXML(StreamString &strm) const {
  // Example XML:
  // <flags id="cpsr_flags" size="4">
  //   <field name="incorrect" start="0" end="0"/>
  // </flags>
  strm.Indent();
  strm << "<flags id=\"" << GetID() << "\" ";
  strm.Printf("size=\"%d\"", GetSize());
  strm << ">";
  for (const Field &field : m_fields) {
    // Skip padding fields.
    if (field.GetName().empty())
      continue;

    strm << "\n";
    strm.IndentMore();
    field.ToXML(strm);
    strm.IndentLess();
  }
  strm.PutChar('\n');
  strm.Indent("</flags>\n");
}

void RegisterFlags::Field::ToXML(StreamString &strm) const {
  // Example XML:
  // <field name="correct" start="0" end="0"/>
  strm.Indent();
  strm << "<field name=\"";

  std::string escaped_name;
  llvm::raw_string_ostream escape_strm(escaped_name);
  llvm::printHTMLEscaped(GetName(), escape_strm);
  strm << escaped_name << "\" ";

  strm.Printf("start=\"%d\" end=\"%d\"", GetStart(), GetEnd());
  strm << "/>";
}
