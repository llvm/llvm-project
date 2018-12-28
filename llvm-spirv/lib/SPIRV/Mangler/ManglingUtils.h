//===------------------------- ManglingUtils.h ---------------------------===//
//
//                              SPIR Tools
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
/*
 * Contributed by: Intel Corporation.
 */

#ifndef SPIRV_MANGLER_MANGLINGUTILS_H
#define SPIRV_MANGLER_MANGLINGUTILS_H

#include "ParameterType.h"

namespace SPIR {

const char *mangledPrimitiveString(TypePrimitiveEnum Primitive);
const char *readablePrimitiveString(TypePrimitiveEnum Primitive);

const char *getMangledAttribute(TypeAttributeEnum Attribute);
const char *getReadableAttribute(TypeAttributeEnum Attribute);

SPIRversion getSupportedVersion(TypePrimitiveEnum T);
const char *getSPIRVersionAsString(SPIRversion Version);

const char *mangledPrimitiveStringfromName(std::string Type);
} // namespace SPIR

#endif // SPIRV_MANGLER_MANGLINGUTILS_H
