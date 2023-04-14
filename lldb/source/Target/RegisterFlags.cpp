//===-- RegisterFlags.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/RegisterFlags.h"

#include <optional>

using namespace lldb_private;

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

RegisterFlags::RegisterFlags(std::string id, unsigned size,
                             const std::vector<Field> &fields)
    : m_id(std::move(id)), m_size(size) {
  // We expect that the XML processor will discard anything describing flags but
  // with no fields.
  assert(fields.size() && "Some fields must be provided.");

  // We expect that these are unsorted but do not overlap.
  // They could fill the register but may have gaps.
  std::vector<Field> provided_fields = fields;
  m_fields.reserve(provided_fields.size());

  // ProcessGDBRemote should have sorted these in descending order already.
  assert(std::is_sorted(provided_fields.rbegin(), provided_fields.rend()));

  // Build a new list of fields that includes anonymous (empty name) fields
  // wherever there is a gap. This will simplify processing later.
  std::optional<Field> previous_field;
  unsigned register_msb = (size * 8) - 1;
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

void RegisterFlags::log(Log *log) const {
  LLDB_LOG(log, "ID: \"{0}\" Size: {1}", m_id.c_str(), m_size);
  for (const Field &field : m_fields)
    field.log(log);
}
