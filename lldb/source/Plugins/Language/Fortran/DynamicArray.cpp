//===-- DynamicArray.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DynamicArray.h"

#include "lldb/Expression/DWARFExpressionList.h"

#include "Plugins/TypeSystem/Fortran/FortranTypes.h"
#include "Plugins/TypeSystem/Fortran/TypeSystemFortran.h"

#include "llvm/Support/Error.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

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
  llvm::DenseMap<FortranArray *, FortranType *> m_children_map;
};

DynamicArraySyntheticFrontEnd::DynamicArraySyntheticFrontEnd(
    lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp), m_element_type() {
  if (valobj_sp)
    Update();
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
  if ((idx + 1) < lb || (idx + 1) > ub)
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
    // Fortran is 1-base indexed, and the offset of element array(1) is 0
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

    child_name = llvm::formatv("[{0}]", idx);
    ConstString type_name =
        CreateArrayTypeName(fortran_type->GetElementType(), new_dimensions,
                            is_allocatable, is_star);
    FortranArray *array_type =
        new FortranArray(fortran_type->GetElementType(), new_dimensions,
                         type_name, child_byte_size, is_allocatable,
                         fortran_type->IsSizeKnown(), new_total_elements);
    CompilerType array_compiler_type(
        TypeSystemWP(m_array_type.GetTypeSystem().GetSharedPointer()),
        (void *)array_type);
    
  } else {
    if (old_dimensions.front().GetByteStride() != 0)
      child_byte_offset = idx * old_dimensions.front().GetByteStride();
    else
      child_byte_offset = idx * fortran_type->GetElementByteSize();
    child_byte_size = fortran_type->GetElementByteSize();
    return fortran_type->GetElementType();
  }
}

} // namespace formatters
} // namespace lldb_private