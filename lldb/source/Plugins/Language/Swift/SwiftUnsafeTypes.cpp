#include "SwiftUnsafeTypes.h"

#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Symbol/SwiftASTContext.h"
#include "lldb/Target/SwiftLanguageRuntime.h"

#include <utility>

using namespace lldb;
using namespace lldb_private;

class SwiftUnsafeBufferPointer {
public:
  SwiftUnsafeBufferPointer(ValueObject &valobj);
  size_t GetCount() const { return m_count; }
  addr_t GetStartAddress() const { return m_start_addr; }
  CompilerType GetElementType() const { return m_elem_type; }
  bool Update();

private:
  ValueObject &m_valobj;
  size_t m_count;
  addr_t m_start_addr;
  CompilerType m_elem_type;
};

SwiftUnsafeBufferPointer::SwiftUnsafeBufferPointer(ValueObject &valobj)
    : m_valobj(*valobj.GetNonSyntheticValue().get()) {}

bool SwiftUnsafeBufferPointer::Update() {
  if (!m_valobj.GetNumChildren())
    return false;

  // Here is the layout of Swift's Unsafe[Mutable]BufferPointer.
  //
  //  ▿ UnsafeBufferPointer
  //    ▿ _position : Optional<UnsafePointer<Int>>
  //      ▿ some : UnsafePointer<Int>
  //        - pointerValue : Int
  //    - count : Int
  //
  // The structure has 2 children:
  //  1. The buffer `count` child stored as a Swift `Int` type. This entry is a
  // "value-providing synthetic children", so lldb need to access to its
  // children in order to get the  actual value.
  //  2. An Optional UnsafePointer to the buffer start address. To access the
  // pointer address, lldb unfolds every value object child until reaching
  // `pointerValue`.

  static ConstString g_count("count");
  ValueObjectSP count_value_sp(m_valobj.GetChildMemberWithName(g_count, true));
  if (!count_value_sp)
    return false;

  ValueObjectSP value_provided_child_sp = nullptr;

  // Implement Swift's 'value-providing synthetic children' workaround.
  // Depending on whether the value object type is a primitive or a structure,
  // lldb should prioritize the synthetic value children.
  // If it has no synthetic children then fallback to non synthetic children.
  ValueObjectSP synthetic = count_value_sp->GetSyntheticValue();
  if (synthetic)
    value_provided_child_sp = synthetic->GetChildAtIndex(0, true);
  if (!value_provided_child_sp)
    value_provided_child_sp = count_value_sp->GetChildAtIndex(0, true);
  // If neither child exists, fail.
  if (!value_provided_child_sp)
    return false;

  size_t count = value_provided_child_sp->GetValueAsUnsigned(UINT64_MAX);

  if (count == UINT64_MAX)
    return false;

  m_count = count;

  static ConstString g_position("_position");
  ValueObjectSP position_value_sp(
      m_valobj.GetChildMemberWithName(g_position, true));
  if (!position_value_sp || !position_value_sp->GetNumChildren())
    return false;

  ValueObjectSP some_value_sp = position_value_sp->GetChildAtIndex(0, true);
  if (!some_value_sp || !some_value_sp->GetNumChildren())
    return false;

  CompilerType argument_type;

  if (CompilerType type = some_value_sp->GetCompilerType())
    argument_type = SwiftASTContext::GetGenericArgumentType(type, 0);

  if (!argument_type.IsValid())
    return nullptr;

  m_elem_type = argument_type;

  ValueObjectSP pointer_value_sp = some_value_sp->GetChildAtIndex(0, true);
  if (!pointer_value_sp)
    return false;

  addr_t addr = pointer_value_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);

  if (!addr || addr == LLDB_INVALID_ADDRESS)
    return false;

  m_start_addr = addr;

  return true;
}

bool lldb_private::formatters::swift::UnsafeBufferPointerSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {

  SwiftUnsafeBufferPointer swift_ubp(valobj);

  if (!swift_ubp.Update())
    return false;

  size_t count = swift_ubp.GetCount();
  addr_t addr = swift_ubp.GetStartAddress();

  stream.Printf("%zu %s (0x%" PRIx64 ")", count,
                (count == 1) ? "value" : "values", addr);

  return true;
}

namespace lldb_private {
namespace formatters {
namespace swift {
class UnsafeBufferPointerSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  UnsafeBufferPointerSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  virtual size_t CalculateNumChildren();

  virtual lldb::ValueObjectSP GetChildAtIndex(size_t idx);

  virtual bool Update();

  virtual bool MightHaveChildren();

  virtual size_t GetIndexOfChildWithName(ConstString name);

  virtual ~UnsafeBufferPointerSyntheticFrontEnd() = default;

private:
  ExecutionContextRef m_exe_ctx_ref;
  uint8_t m_ptr_size;
  lldb::ByteOrder m_order;

  SwiftUnsafeBufferPointer m_unsafe_ptr;
  size_t m_element_stride;
  DataBufferSP m_buffer_sp;
  std::vector<ValueObjectSP> m_children;
};
} // namespace swift
} // namespace formatters
} // namespace lldb_private

lldb_private::formatters::swift::UnsafeBufferPointerSyntheticFrontEnd::
    UnsafeBufferPointerSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp.get()),
      m_unsafe_ptr(*valobj_sp.get()) {

  ProcessSP process_sp = valobj_sp->GetProcessSP();
  if (!process_sp)
    return;

  m_ptr_size = process_sp->GetAddressByteSize();
  m_order = process_sp->GetByteOrder();

  if (valobj_sp)
    Update();
}

size_t lldb_private::formatters::swift::UnsafeBufferPointerSyntheticFrontEnd::
    CalculateNumChildren() {
  return m_unsafe_ptr.GetCount();
}

lldb::ValueObjectSP lldb_private::formatters::swift::
    UnsafeBufferPointerSyntheticFrontEnd::GetChildAtIndex(size_t idx) {
  const size_t num_children = CalculateNumChildren();

  if (idx >= num_children || idx >= m_children.size())
    return lldb::ValueObjectSP();

  return m_children[idx];
}

bool lldb_private::formatters::swift::UnsafeBufferPointerSyntheticFrontEnd::
    Update() {
  m_children.clear();
  ValueObjectSP valobj_sp = m_backend.GetSP();
  if (!valobj_sp)
    return false;
  m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();

  lldb::ProcessSP process_sp(valobj_sp->GetProcessSP());
  if (!process_sp)
    return false;
  if (!m_unsafe_ptr.Update())
    return false;

  const addr_t start_addr = m_unsafe_ptr.GetStartAddress();
  const size_t num_children = CalculateNumChildren();
  const CompilerType element_type = m_unsafe_ptr.GetElementType();

  auto stride = element_type.GetByteStride(process_sp.get());
  if (!stride)
    return false;

  m_element_stride = *stride;

  if (m_children.empty()) {
    size_t buffer_size = num_children * m_element_stride;
    m_buffer_sp.reset(new DataBufferHeap(buffer_size, 0));

    Status error;
    size_t read_bytes = process_sp->ReadMemory(
        start_addr, m_buffer_sp->GetBytes(), buffer_size, error);

    if (!read_bytes || error.Fail())
      return false;

    DataExtractor buffer_data(m_buffer_sp->GetBytes(),
                              m_buffer_sp->GetByteSize(), m_order, m_ptr_size);

    for (size_t i = 0; i < num_children; i++) {
      StreamString idx_name;
      idx_name.Printf("[%" PRIu64 "]", i);
      DataExtractor data(buffer_data, i * m_element_stride, m_element_stride);
      m_children.push_back(CreateValueObjectFromData(
          idx_name.GetString(), data, m_exe_ctx_ref, element_type));
    }
  }

  return m_children.size() == num_children;
}

bool lldb_private::formatters::swift::UnsafeBufferPointerSyntheticFrontEnd::
    MightHaveChildren() {
  return m_unsafe_ptr.GetCount();
}

size_t lldb_private::formatters::swift::UnsafeBufferPointerSyntheticFrontEnd::
    GetIndexOfChildWithName(ConstString name) {
  return UINT32_MAX;
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::swift::UnsafeBufferPointerSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;
  return (new UnsafeBufferPointerSyntheticFrontEnd(valobj_sp));
}
