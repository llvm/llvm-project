//===-- DynamicArray.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: Clean this up, fix names, add comments
#include "DynamicArray.h"

#include "lldb/Expression/DWARFExpressionList.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/SymbolContext.h"

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

  ~DynamicArraySyntheticFrontEnd();

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override;

private:
  CompilerType m_element_type;
  CompilerType m_array_type;
  lldb::addr_t m_array_addr;
  bool m_allocated;
  llvm::DenseMap<FortranArray *, FortranType *> m_children_map;
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
  FortranArray *array_type = static_cast<FortranArray *>(raw_type);
  m_element_type = array_type->GetElementType();
  if (!array_type->IsDynamic())
    return lldb::ChildCacheState::eReuse;
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

  DWARFExpressionList allocated_exp = array_type->GetAllocatedExpression();
  DWARFExpressionList data_location_exp =
      array_type->GetDataLocationExpression();

  llvm::Expected<Value> allocated_val_or_err = allocated_exp.Evaluate(
      &exe_ctx, nullptr, loclist_base_load_addr, nullptr, nullptr);
  if (!allocated_val_or_err)
    return lldb::ChildCacheState::eRefetch;

  Value allocated_val = *allocated_val_or_err;
  if (allocated_val.ResolveValue(&exe_ctx).IsZero())
    return lldb::ChildCacheState::eRefetch;

  m_allocated = true;

  llvm::Expected<Value> array_addr_or_err = data_location_exp.Evaluate(
      &exe_ctx, nullptr, loclist_base_load_addr, nullptr, nullptr);
  if (!array_addr_or_err)
    return lldb::ChildCacheState::eRefetch;

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
      llvm::Expected<Value> lower_bound_or_err = lower_bound_exp.Evaluate(
          &exe_ctx, nullptr, loclist_base_load_addr, nullptr, nullptr);
      if (!lower_bound_or_err)
        return lldb::ChildCacheState::eRefetch;

      Value lower_bound = *lower_bound_or_err;

      dimension_info.lower_bound =
          lower_bound.ResolveValue(&exe_ctx).SLongLong(0);
    } else
      dimension_info.lower_bound = dimension.GetLowerBound().GetBound();

    if (upper_bound_exp.IsValid()) {
      llvm::Expected<Value> upper_bound_or_err = upper_bound_exp.Evaluate(
          &exe_ctx, nullptr, loclist_base_load_addr, nullptr, nullptr);
      if (!upper_bound_or_err)
        return lldb::ChildCacheState::eRefetch;

      Value upper_bound = *upper_bound_or_err;
      dimension_info.upper_bound =
          upper_bound.ResolveValue(&exe_ctx).SLongLong(0);
    } else
      dimension_info.upper_bound = dimension.GetUpperBound().GetBound();
    // TODO: Handle cases where we only upper and lower bound
    if (element_count_exp.IsValid()) {
      llvm::Expected<Value> element_count_or_err = element_count_exp.Evaluate(
          &exe_ctx, nullptr, loclist_base_load_addr, nullptr, nullptr);
      if (!element_count_or_err)
        return lldb::ChildCacheState::eRefetch;

      Value element_count = *element_count_or_err;
      dimension_info.element_count =
          element_count.ResolveValue(&exe_ctx).ULongLong(0);
    } else
      dimension_info.element_count = dimension.GetElementCount();

    if (byte_stride_exp.IsValid()) {
      llvm::Expected<Value> byte_stride_or_err = byte_stride_exp.Evaluate(
          &exe_ctx, nullptr, loclist_base_load_addr, nullptr, nullptr);
      if (!byte_stride_or_err)
        return lldb::ChildCacheState::eRefetch;

      Value byte_stride = *byte_stride_or_err;
      dimension_info.byte_stride =
          byte_stride.ResolveValue(&exe_ctx).SLongLong(0);
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

  if (elem_byte_size_or_err) {
    total_array_size = total_elements * (*elem_byte_size_or_err);
  } else {
    // Fallback if the element type is incomplete
    total_array_size = 0;
  }
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
  uint32_t child_byte_size;
  int32_t child_byte_offset;
  std::string child_name;
  llvm::ArrayRef<ArrayShape> old_dimensions = fortran_type->GetDimensions();
  int64_t ub = old_dimensions.front().GetUpperBound().GetBound();
  int64_t lb = old_dimensions.front().GetLowerBound().GetBound();
  uint64_t num_elements = old_dimensions.front().GetNumberOfElements();
  if (idx >= num_elements)
    return ValueObjectSP();

  if (old_dimensions.size() > 1) {

    llvm::SmallVector<ArrayShape, 2> new_dimensions(old_dimensions.begin() + 1,
                                                    old_dimensions.end());

    ArrayShape old_first_dimension = old_dimensions.front();
    uint64_t new_byte_stride;

    if (old_first_dimension.GetByteStride() != 0)
      new_byte_stride = old_first_dimension.GetNumberOfElements() *
                        old_first_dimension.GetByteStride();
    else
      new_byte_stride = old_first_dimension.GetNumberOfElements() *
                        fortran_type->GetElementByteSize();

    new_dimensions.front().SetByteStride(new_byte_stride);
    bool is_star = new_dimensions.back().GetLowerBound().IsStar();
    bool is_allocatable = fortran_type->IsAllocatable();
    uint64_t new_total_elements = fortran_type->GetTotalElements() /
                                  old_first_dimension.GetNumberOfElements();
    if (old_dimensions.front().GetByteStride() != 0)
      child_byte_offset = idx * old_dimensions.front().GetByteStride();
    else
      child_byte_offset = idx * fortran_type->GetElementByteSize();

    uint64_t last_dim_stride = new_dimensions.back().GetByteStride();

    if (last_dim_stride != 0) {
      child_byte_size =
          new_dimensions.back().GetNumberOfElements() * last_dim_stride;
    } else {
      child_byte_size = new_total_elements * fortran_type->GetElementByteSize();
    }

    child_name = llvm::formatv("[{0}]", lb + idx);
    ConstString type_name =
        CreateArrayTypeName(fortran_type->GetElementType(), new_dimensions,
                            is_allocatable, is_star);
    FortranArray *array_type = new FortranArray(
        fortran_type->GetElementType(), new_dimensions, type_name,
        child_byte_size, is_allocatable, fortran_type->IsDynamic(),
        new_total_elements, fortran_type->GetAllocatedExpression(),
        fortran_type->GetDataLocationExpression());
    CompilerType array_compiler_type(
        TypeSystemWP(m_array_type.GetTypeSystem().GetSharedPointer()),
        (void *)array_type);
    m_children_map[fortran_type] = array_type;
    uint64_t array_address = m_array_addr + child_byte_offset;
    return CreateChildValueObjectFromAddress(child_name, array_address,
                                             m_backend.GetExecutionContextRef(),
                                             array_compiler_type);
  } else {
    if (old_dimensions.front().GetByteStride() != 0)
      child_byte_offset = idx * old_dimensions.front().GetByteStride();
    else
      child_byte_offset = idx * fortran_type->GetElementByteSize();
    child_byte_size = fortran_type->GetElementByteSize();
    StreamString name;
    name.Printf("[%" PRIu64 "]", (uint64_t)idx);
    return CreateChildValueObjectFromAddress(
        name.GetString(), m_array_addr + child_byte_offset,
        m_backend.GetExecutionContextRef(), m_element_type);
  }
}

} // namespace formatters
} // namespace lldb_private