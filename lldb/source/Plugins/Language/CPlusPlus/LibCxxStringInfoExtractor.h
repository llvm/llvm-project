
#ifndef LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_LIBCXXSTRINGINFOEXTRACTOR_H
#define LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_LIBCXXSTRINGINFOEXTRACTOR_H

#include "lldb/Core/ValueObject.h"
#include "lldb/lldb-forward.h"

#include <optional>
#include <utility>

using namespace lldb;
using namespace lldb_private;

/// The field layout in a libc++ string (cap, side, data or data, size, cap).
namespace {
enum class StringLayout { CSD, DSC };
}

/// Determine the size in bytes of \p valobj (a libc++ std::string object) and
/// extract its data payload. Return the size + payload pair.
// TODO: Support big-endian architectures.
static std::optional<std::pair<uint64_t, ValueObjectSP>>
ExtractLibcxxStringInfo(ValueObject &valobj) {
  ValueObjectSP valobj_r_sp =
      valobj.GetChildMemberWithName(ConstString("__r_"), /*can_create=*/true);
  if (!valobj_r_sp || !valobj_r_sp->GetError().Success())
    return {};

  // __r_ is a compressed_pair of the actual data and the allocator. The data we
  // want is in the first base class.
  ValueObjectSP valobj_r_base_sp =
      valobj_r_sp->GetChildAtIndex(0, /*can_create=*/true);
  if (!valobj_r_base_sp)
    return {};

  ValueObjectSP valobj_rep_sp = valobj_r_base_sp->GetChildMemberWithName(
      ConstString("__value_"), /*can_create=*/true);
  if (!valobj_rep_sp)
    return {};

  ValueObjectSP l = valobj_rep_sp->GetChildMemberWithName(ConstString("__l"),
                                                          /*can_create=*/true);
  if (!l)
    return {};

  StringLayout layout = l->GetIndexOfChildWithName(ConstString("__data_")) == 0
                            ? StringLayout::DSC
                            : StringLayout::CSD;

  bool short_mode = false;    // this means the string is in short-mode and the
                              // data is stored inline
  bool using_bitmasks = true; // Whether the class uses bitmasks for the mode
                              // flag (pre-D123580).
  uint64_t size;
  uint64_t size_mode_value = 0;

  ValueObjectSP short_sp = valobj_rep_sp->GetChildMemberWithName(
      ConstString("__s"), /*can_create=*/true);
  if (!short_sp)
    return {};

  ValueObjectSP is_long =
      short_sp->GetChildMemberWithName(ConstString("__is_long_"), true);
  ValueObjectSP size_sp =
      short_sp->GetChildAtNamePath({ConstString("__size_")});
  if (!size_sp)
    return {};

  if (is_long) {
    using_bitmasks = false;
    short_mode = !is_long->GetValueAsUnsigned(/*fail_value=*/0);
    size = size_sp->GetValueAsUnsigned(/*fail_value=*/0);
  } else {
    // The string mode is encoded in the size field.
    size_mode_value = size_sp->GetValueAsUnsigned(0);
    uint8_t mode_mask = layout == StringLayout::DSC ? 0x80 : 1;
    short_mode = (size_mode_value & mode_mask) == 0;
  }

  if (short_mode) {
    ValueObjectSP location_sp =
        short_sp->GetChildMemberWithName(ConstString("__data_"), true);
    if (using_bitmasks)
      size = (layout == StringLayout::DSC) ? size_mode_value
                                           : ((size_mode_value >> 1) % 256);

    // When the small-string optimization takes place, the data must fit in the
    // inline string buffer (23 bytes on x86_64/Darwin). If it doesn't, it's
    // likely that the string isn't initialized and we're reading garbage.
    ExecutionContext exe_ctx(location_sp->GetExecutionContextRef());
    const std::optional<uint64_t> max_bytes =
        location_sp->GetCompilerType().GetByteSize(
            exe_ctx.GetBestExecutionContextScope());
    if (!max_bytes || size > *max_bytes || !location_sp)
      return {};

    return std::make_pair(size, location_sp);
  }

  // we can use the layout_decider object as the data pointer
  ValueObjectSP location_sp =
      l->GetChildMemberWithName(ConstString("__data_"), /*can_create=*/true);
  ValueObjectSP size_vo =
      l->GetChildMemberWithName(ConstString("__size_"), /*can_create=*/true);
  ValueObjectSP capacity_vo =
      l->GetChildMemberWithName(ConstString("__cap_"), /*can_create=*/true);
  if (!size_vo || !location_sp || !capacity_vo)
    return {};
  size = size_vo->GetValueAsUnsigned(LLDB_INVALID_OFFSET);
  uint64_t capacity = capacity_vo->GetValueAsUnsigned(LLDB_INVALID_OFFSET);
  if (!using_bitmasks && layout == StringLayout::CSD)
    capacity *= 2;
  if (size == LLDB_INVALID_OFFSET || capacity == LLDB_INVALID_OFFSET ||
      capacity < size)
    return {};
  return std::make_pair(size, location_sp);
}

#endif // LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_LIBCXXSTRINGINFOEXTRACTOR_H
