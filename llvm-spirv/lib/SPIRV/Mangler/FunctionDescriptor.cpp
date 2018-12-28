//===---------------------- FunctionDescriptor.cpp -----------------------===//
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

#include "FunctionDescriptor.h"
#include "ParameterType.h"
#include <sstream>

namespace SPIR {

std::string FunctionDescriptor::nullString() {
  return std::string("<invalid>");
}

std::string FunctionDescriptor::toString() const {
  std::stringstream Stream;
  if (isNull()) {
    return FunctionDescriptor::nullString();
  }
  Stream << Name << "(";
  size_t ParamCount = Parameters.size();
  if (ParamCount > 0) {
    for (size_t I = 0; I < ParamCount - 1; ++I)
      Stream << Parameters[I]->toString() << ", ";
    Stream << Parameters[ParamCount - 1]->toString();
  }
  Stream << ")";
  return Stream.str();
}

static bool equal(const TypeVector &L, const TypeVector &R) {
  if (&L == &R)
    return true;
  if (L.size() != R.size())
    return false;
  TypeVector::const_iterator Itl = L.begin(), Itr = R.begin(), Endl = L.end();
  while (Itl != Endl) {
    if (!(*Itl)->equals(*Itr))
      return false;
    ++Itl;
    ++Itr;
  }
  return true;
}

//
// FunctionDescriptor
//

bool FunctionDescriptor::operator==(const FunctionDescriptor &That) const {
  if (this == &That)
    return true;
  if (Name != That.Name)
    return false;
  return equal(Parameters, That.Parameters);
}

bool FunctionDescriptor::operator<(const FunctionDescriptor &That) const {
  int StrCmp = Name.compare(That.Name);
  if (StrCmp)
    return (StrCmp < 0);
  size_t Len = Parameters.size(), ThatLen = That.Parameters.size();
  if (Len != ThatLen)
    return Len < ThatLen;
  TypeVector::const_iterator It = Parameters.begin(), E = Parameters.end(),
                             Thatit = That.Parameters.begin();
  while (It != E) {
    int Cmp = (*It)->toString().compare((*Thatit)->toString());
    if (Cmp)
      return (Cmp < 0);
    ++Thatit;
    ++It;
  }
  return false;
}

bool FunctionDescriptor::isNull() const {
  return (Name.empty() && Parameters.empty());
}

FunctionDescriptor FunctionDescriptor::null() {
  FunctionDescriptor Fd;
  Fd.Name = "";
  return Fd;
}

} // namespace SPIR
