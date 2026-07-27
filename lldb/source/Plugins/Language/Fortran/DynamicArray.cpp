//===-- DynamicArray.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: Clean this up, fix names, add comments
// TODO: Fix the BitSize...
#include "DynamicArray.h"

#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Expression/DWARFExpressionList.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/ValueObject/ValueObject.h"
#include "Plugins/TypeSystem/Fortran/FortranTypes.h"
#include "Plugins/TypeSystem/Fortran/TypeSystemFortran.h"

#include "llvm/Support/Error.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::plugin::fortran;

namespace lldb_private {
namespace formatters {

class DynamicArraySyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  DynamicArraySyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  ~DynamicArraySyntheticFrontEnd() = default;

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override;

private:
  CompilerType m_element_type;
  CompilerType m_array_type;
  lldb::addr_t m_array_addr;
  bool m_allocated;
  std::shared_ptr<lldb_private::TypeSystemFortran> m_ast_sp;
  // TODO: very bad design, children will overwrite themselves...
  llvm::DenseMap<FortranArray *, CompilerType> m_children_map;
};

DynamicArraySyntheticFrontEnd::DynamicArraySyntheticFrontEnd(
    lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp), m_element_type() {
  if (valobj_sp)
    Update();
}
// TODO: Handle star arrays
lldb::ChildCacheState DynamicArraySyntheticFrontEnd::Update() {
  m_children_map.clear();
  m_allocated = false;

  lldb::opaque_compiler_type_t raw_type =
      m_backend.GetCompilerType().GetOpaqueQualType();
  if (!raw_type)
    return lldb::ChildCacheState::eRefetch;
  auto ast_sp = m_backend.GetCompilerType().GetTypeSystem<TypeSystemFortran>();
  if (!ast_sp)
    return lldb::ChildCacheState::eRefetch;
  m_ast_sp = ast_sp;
  FortranArray *array_type = static_cast<FortranArray *>(raw_type);
  if (!array_type) {
    return lldb::ChildCacheState::eRefetch;
  }
  m_element_type = array_type->GetElementType();
  if (!array_type->IsDynamic()) {
    m_allocated = true;
    m_array_addr = m_backend.GetValueAsUnsigned(LLDB_INVALID_ADDRESS);

    if (m_array_addr == LLDB_INVALID_ADDRESS) {
      m_array_addr = m_backend.GetLoadAddress();
    }

    m_array_type = m_backend.GetCompilerType();
    return lldb::ChildCacheState::eReuse;
  }
  lldb::addr_t loclist_base_load_addr = LLDB_INVALID_ADDRESS;
  ExecutionContext exe_ctx(m_backend.GetExecutionContextRef());
  Target *target = exe_ctx.GetTargetPtr();
  StackFrame *frame = exe_ctx.GetFramePtr();

  if (!target || !frame)
    return lldb::ChildCacheState::eRefetch;
  SymbolContext sc = frame->GetSymbolContext(eSymbolContextFunction);

  if (!sc.function)
    return lldb::ChildCacheState::eRefetch;

  loclist_base_load_addr = sc.function->GetAddress().GetLoadAddress(target);

  lldb::addr_t obj_load_addr = m_backend.GetLoadAddress();

  if (obj_load_addr == LLDB_INVALID_ADDRESS)
    return lldb::ChildCacheState::eRefetch;

  Value object_address_val;
  object_address_val.SetValueType(Value::ValueType::LoadAddress);
  object_address_val.GetScalar() = obj_load_addr;

  DWARFExpressionList allocated_exp = array_type->GetAllocatedExpression();
  DWARFExpressionList data_location_exp =
      array_type->GetDataLocationExpression();

  llvm::Expected<Value> allocated_val_or_err = allocated_exp.Evaluate(
      &exe_ctx, nullptr, loclist_base_load_addr, nullptr, &object_address_val);
  if (!allocated_val_or_err) {
    llvm::consumeError(allocated_val_or_err.takeError());
    return lldb::ChildCacheState::eRefetch;
  }

  Value allocated_val = *allocated_val_or_err;
  if (allocated_val.ResolveValue(&exe_ctx).IsZero())
    return lldb::ChildCacheState::eRefetch;

  m_allocated = true;

  llvm::Expected<Value> array_addr_or_err = data_location_exp.Evaluate(
      &exe_ctx, nullptr, loclist_base_load_addr, nullptr, &object_address_val);
  if (!array_addr_or_err) {
    llvm::consumeError(array_addr_or_err.takeError());
    return lldb::ChildCacheState::eRefetch;
  }

  Value array_addr = *array_addr_or_err;

  m_array_addr =
      array_addr.ResolveValue(&exe_ctx).ULongLong(LLDB_INVALID_ADDRESS);

  llvm::ArrayRef<ArrayShape> dimensions = array_type->GetDimensions();
  FortranArrayMetadata array_info;
  array_info.element_type = m_element_type;
  array_info.is_allocatable = array_type->IsAllocatable();
  array_info.is_dynamic = false;
  array_info.is_star = false;
  for (ArrayShape dimension : dimensions) {
    FortranDimension dimension_info;
    DWARFExpressionList lower_bound_exp = dimension.GetLowerBoundExpression();
    DWARFExpressionList upper_bound_exp = dimension.GetUpperBoundExpression();
    DWARFExpressionList element_count_exp =
        dimension.GetElementCountExpression();
    DWARFExpressionList byte_stride_exp = dimension.GetByteStrideExpression();
    if (lower_bound_exp.IsValid()) {
      llvm::Expected<Value> lower_bound_or_err =
          lower_bound_exp.Evaluate(&exe_ctx, nullptr, loclist_base_load_addr,
                                   nullptr, &object_address_val);
      if (!lower_bound_or_err) {
        llvm::consumeError(lower_bound_or_err.takeError());
        return lldb::ChildCacheState::eRefetch;
      }

      Value lower_bound = *lower_bound_or_err;

      dimension_info.lower_bound =
          lower_bound.ResolveValue(&exe_ctx).SLongLong(0);
    } else
      dimension_info.lower_bound = dimension.GetLowerBound().GetBound();

    // TODO: Handle cases where we only upper and lower bound
    if (element_count_exp.IsValid()) {
      llvm::Expected<Value> element_count_or_err =
          element_count_exp.Evaluate(&exe_ctx, nullptr, loclist_base_load_addr,
                                     nullptr, &object_address_val);
      if (!element_count_or_err) {
        llvm::consumeError(element_count_or_err.takeError());
        return lldb::ChildCacheState::eRefetch;
      }
      Value element_count = *element_count_or_err;
      dimension_info.element_count =
          element_count.ResolveValue(&exe_ctx).ULongLong(0);
    } else
      dimension_info.element_count = dimension.GetElementCount();

    if (upper_bound_exp.IsValid()) {
      llvm::Expected<Value> upper_bound_or_err =
          upper_bound_exp.Evaluate(&exe_ctx, nullptr, loclist_base_load_addr,
                                   nullptr, &object_address_val);
      if (!upper_bound_or_err) {
        llvm::consumeError(upper_bound_or_err.takeError());
        return lldb::ChildCacheState::eRefetch;
      }

      Value upper_bound = *upper_bound_or_err;
      dimension_info.upper_bound =
          upper_bound.ResolveValue(&exe_ctx).SLongLong(0);
    } else if (dimension.GetUpperBound().IsBoundKnown())
      dimension_info.upper_bound = dimension.GetUpperBound().GetBound();
    else
      dimension_info.upper_bound =
          std::get<int64_t>(dimension_info.lower_bound) +
          std::get<uint64_t>(dimension_info.element_count);

    if (byte_stride_exp.IsValid()) {
      llvm::Expected<Value> byte_stride_or_err =
          byte_stride_exp.Evaluate(&exe_ctx, nullptr, loclist_base_load_addr,
                                   nullptr, &object_address_val);
      if (!byte_stride_or_err) {
        llvm::consumeError(byte_stride_or_err.takeError());
        return lldb::ChildCacheState::eRefetch;
      }

      Value byte_stride = *byte_stride_or_err;
      dimension_info.byte_stride =
          byte_stride.ResolveValue(&exe_ctx).ULongLong(0);
    } else
      dimension_info.byte_stride = dimension.GetByteStride();

    array_info.dimensions.push_back(dimension_info);
  }
  uint64_t total_elements = 1;
  uint64_t total_array_size = 0;

  for (auto &dim : array_info.dimensions) {
    uint64_t count = std::get<uint64_t>(dim.element_count);

    int64_t lb = 0;
    int64_t ub = 0;
    if (const auto *l_val = std::get_if<int64_t>(&dim.lower_bound))
      lb = *l_val;
    if (const auto *u_val = std::get_if<int64_t>(&dim.upper_bound))
      ub = *u_val;

    if (count == 0) {
      if (ub >= lb) {
        count = ub - lb + 1;
      } else if (array_info.is_star && &dim == &array_info.dimensions.back()) {
        count = 0;
      }
    }

    // Update the struct so it's correct for later indexing math
    dim.element_count = count;
    total_elements *= count;
  }

  // Calculate total byte size of the array
  llvm::Expected<uint64_t> elem_byte_size_or_err =
      m_element_type.GetByteSize(exe_ctx.GetBestExecutionContextScope());

  if (!elem_byte_size_or_err) {
    llvm::consumeError(elem_byte_size_or_err.takeError());
    total_array_size = 0;
  } else
    // Fallback if the element type is incomplete
    total_array_size = total_elements * (*elem_byte_size_or_err);

  m_array_type =
      ast_sp->CreateArrayType(array_info, total_array_size, total_elements);
  return lldb::ChildCacheState::eRefetch;
}

llvm::Expected<uint32_t> DynamicArraySyntheticFrontEnd::CalculateNumChildren() {
  lldb::opaque_compiler_type_t raw_type = m_array_type.GetOpaqueQualType();
  if (!raw_type)
    return 0;
  FortranArray *array_type = static_cast<FortranArray *>(raw_type);
  if (!array_type)
    return 0;

  ArrayShape first_dimension = array_type->GetDimensions().front();
  return first_dimension.GetNumberOfElements();
}

lldb::ValueObjectSP
DynamicArraySyntheticFrontEnd::GetChildAtIndex(uint32_t idx) {
  if (!m_allocated)
    return ValueObjectSP();
  lldb::opaque_compiler_type_t raw_type = m_array_type.GetOpaqueQualType();

  if (!raw_type)
    return ValueObjectSP();

  FortranType *super_type = static_cast<FortranType *>(raw_type);

  if (super_type->GetKind() != FortranType::KIND_ARRAY)
    return ValueObjectSP();

  FortranArray *fortran_type = static_cast<FortranArray *>(super_type);

  if (!fortran_type)
    return ValueObjectSP();
  ExecutionContext exe_ctx(m_backend.GetExecutionContextRef());
  bool omit_empty_base_classes = true;
  bool ignore_array_bounds = false;
  uint32_t child_byte_size = 0;
  int32_t child_byte_offset = 0;
  uint32_t child_bitfield_bit_size = 0;
  uint32_t child_bitfield_bit_offset = 0;
  bool child_is_base_class = false;
  bool child_is_deref_of_parent = false;
  uint64_t language_flags = 0;
  const bool transparent_pointers = true;
  std::string child_name;
  llvm::ArrayRef<ArrayShape> old_dimensions = fortran_type->GetDimensions();
  int64_t ub = old_dimensions.front().GetUpperBound().GetBound();
  int64_t lb = old_dimensions.front().GetLowerBound().GetBound();
  uint64_t num_elements = old_dimensions.front().GetNumberOfElements();
  if (idx >= num_elements)
    return ValueObjectSP();

  child_name = llvm::formatv("[{0}]", idx);
  llvm::Expected<CompilerType> child_type_orr_err =
      m_ast_sp->GetChildCompilerTypeAtIndex(
          raw_type, &exe_ctx, idx, transparent_pointers,
          omit_empty_base_classes, ignore_array_bounds, child_name,
          child_byte_size, child_byte_offset, child_bitfield_bit_size,
          child_bitfield_bit_offset, child_is_base_class,
          child_is_deref_of_parent, &m_backend, language_flags);
  if (!child_type_orr_err) {
    llvm::consumeError(child_type_orr_err.takeError());
    return ValueObjectSP();
  }
  CompilerType child_type = *child_type_orr_err;
  m_children_map[fortran_type] = child_type;
  uint64_t array_address = m_array_addr + child_byte_offset;
  if (!fortran_type->IsDynamic() && m_array_addr == LLDB_INVALID_ADDRESS) {
    return m_backend.GetSyntheticChildAtOffset(child_byte_offset, child_type,
                                               true, ConstString(child_name));
  }

  lldb::ValueObjectSP child_sp = CreateChildValueObjectFromAddress(child_name, array_address,
                                           m_backend.GetExecutionContextRef(),
                                           child_type, false);
  if (child_sp) {
    child_sp->GetValue().SetValueType(Value::ValueType::LoadAddress);
    child_sp->GetValue().GetScalar() = array_address;
  }
  
  return child_sp;
}

llvm::Expected<size_t> lldb_private::formatters::DynamicArraySyntheticFrontEnd::
    GetIndexOfChildWithName(ConstString name) {
  if (!m_array_type)
    return llvm::createStringErrorV("type has no child named '{0}'", name);
  auto optional_idx = formatters::ExtractIndexFromString(name.GetCString());
  if (!optional_idx) {
    return llvm::createStringErrorV("type has no child named '{0}'", name);
  }
  return *optional_idx;
}

lldb_private::SyntheticChildrenFrontEnd *
FortranDynamicArraySyntheticFrontEndCreator(CXXSyntheticChildren *,
                                            lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;
  return new DynamicArraySyntheticFrontEnd(valobj_sp);
}

} // namespace formatters
} // namespace lldb_private