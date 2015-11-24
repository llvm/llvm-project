//===- CodeGenDAGPatterns.cpp - Read DAG patterns from .td file -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CodeGenDAGPatterns class, which is used to read and
// represent the patterns present in a .td file for instructions.
//
//===----------------------------------------------------------------------===//

#include "CodeGenDAGPatterns.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <cstdio>
#include <set>
using namespace llvm;

#define DEBUG_TYPE "dag-patterns"

//===----------------------------------------------------------------------===//
//  EEVT::TypeSet Implementation
//===----------------------------------------------------------------------===//

static inline bool isInteger(MVT::SimpleValueType VT) {
  return MVT(VT).isInteger();
}
static inline bool isFloatingPoint(MVT::SimpleValueType VT) {
  return MVT(VT).isFloatingPoint();
}
static inline bool isVector(MVT::SimpleValueType VT) {
  return MVT(VT).isVector();
}
static inline bool isScalar(MVT::SimpleValueType VT) {
  return !MVT(VT).isVector();
}

EEVT::TypeSet::TypeSet(MVT::SimpleValueType VT, TreePattern &TP) {
  if (VT == MVT::iAny)
    EnforceInteger(TP);
  else if (VT == MVT::fAny)
    EnforceFloatingPoint(TP);
  else if (VT == MVT::vAny)
    EnforceVector(TP);
  else {
    assert((VT < MVT::LAST_VALUETYPE || VT == MVT::iPTR ||
            VT == MVT::iPTRAny || VT == MVT::Any) && "Not a concrete type!");
    TypeVec.push_back(VT);
  }
}


EEVT::TypeSet::TypeSet(ArrayRef<MVT::SimpleValueType> VTList) {
  assert(!VTList.empty() && "empty list?");
  TypeVec.append(VTList.begin(), VTList.end());

  if (!VTList.empty())
    assert(VTList[0] != MVT::iAny && VTList[0] != MVT::vAny &&
           VTList[0] != MVT::fAny);

  // Verify no duplicates.
  array_pod_sort(TypeVec.begin(), TypeVec.end());
  assert(std::unique(TypeVec.begin(), TypeVec.end()) == TypeVec.end());
}

/// FillWithPossibleTypes - Set to all legal types and return true, only valid
/// on completely unknown type sets.
bool EEVT::TypeSet::FillWithPossibleTypes(TreePattern &TP,
                                          bool (*Pred)(MVT::SimpleValueType),
                                          const char *PredicateName) {
  assert(isCompletelyUnknown());
  ArrayRef<MVT::SimpleValueType> LegalTypes =
    TP.getDAGPatterns().getTargetInfo().getLegalValueTypes();

  if (TP.hasError())
    return false;

  for (MVT::SimpleValueType VT : LegalTypes)
    if (!Pred || Pred(VT))
      TypeVec.push_back(VT);

  // If we have nothing that matches the predicate, bail out.
  if (TypeVec.empty()) {
    TP.error("Type inference contradiction found, no " +
             std::string(PredicateName) + " types found");
    return false;
  }
  // No need to sort with one element.
  if (TypeVec.size() == 1) return true;

  // Remove duplicates.
  array_pod_sort(TypeVec.begin(), TypeVec.end());
  TypeVec.erase(std::unique(TypeVec.begin(), TypeVec.end()), TypeVec.end());

  return true;
}

/// hasIntegerTypes - Return true if this TypeSet contains iAny or an
/// integer value type.
bool EEVT::TypeSet::hasIntegerTypes() const {
  return std::any_of(TypeVec.begin(), TypeVec.end(), isInteger);
}

/// hasFloatingPointTypes - Return true if this TypeSet contains an fAny or
/// a floating point value type.
bool EEVT::TypeSet::hasFloatingPointTypes() const {
  return std::any_of(TypeVec.begin(), TypeVec.end(), isFloatingPoint);
}

/// hasScalarTypes - Return true if this TypeSet contains a scalar value type.
bool EEVT::TypeSet::hasScalarTypes() const {
  return std::any_of(TypeVec.begin(), TypeVec.end(), isScalar);
}

/// hasVectorTypes - Return true if this TypeSet contains a vAny or a vector
/// value type.
bool EEVT::TypeSet::hasVectorTypes() const {
  return std::any_of(TypeVec.begin(), TypeVec.end(), isVector);
}


std::string EEVT::TypeSet::getName() const {
  if (TypeVec.empty()) return "<empty>";

  std::string Result;

  for (unsigned i = 0, e = TypeVec.size(); i != e; ++i) {
    std::string VTName = llvm::getEnumName(TypeVec[i]);
    // Strip off MVT:: prefix if present.
    if (VTName.substr(0,5) == "MVT::")
      VTName = VTName.substr(5);
    if (i) Result += ':';
    Result += VTName;
  }

  if (TypeVec.size() == 1)
    return Result;
  return "{" + Result + "}";
}

/// MergeInTypeInfo - This merges in type information from the specified
/// argument.  If 'this' changes, it returns true.  If the two types are
/// contradictory (e.g. merge f32 into i32) then this flags an error.
bool EEVT::TypeSet::MergeInTypeInfo(const EEVT::TypeSet &InVT, TreePattern &TP){
  if (InVT.isCompletelyUnknown() || *this == InVT || TP.hasError())
    return false;

  if (isCompletelyUnknown()) {
    *this = InVT;
    return true;
  }

  assert(!TypeVec.empty() && !InVT.TypeVec.empty() && "No unknowns");

  // Handle the abstract cases, seeing if we can resolve them better.
  switch (TypeVec[0]) {
  default: break;
  case MVT::iPTR:
  case MVT::iPTRAny:
    if (InVT.hasIntegerTypes()) {
      EEVT::TypeSet InCopy(InVT);
      InCopy.EnforceInteger(TP);
      InCopy.EnforceScalar(TP);

      if (InCopy.isConcrete()) {
        // If the RHS has one integer type, upgrade iPTR to i32.
        TypeVec[0] = InVT.TypeVec[0];
        return true;
      }

      // If the input has multiple scalar integers, this doesn't add any info.
      if (!InCopy.isCompletelyUnknown())
        return false;
    }
    break;
  }

  // If the input constraint is iAny/iPTR and this is an integer type list,
  // remove non-integer types from the list.
  if ((InVT.TypeVec[0] == MVT::iPTR || InVT.TypeVec[0] == MVT::iPTRAny) &&
      hasIntegerTypes()) {
    bool MadeChange = EnforceInteger(TP);

    // If we're merging in iPTR/iPTRAny and the node currently has a list of
    // multiple different integer types, replace them with a single iPTR.
    if ((InVT.TypeVec[0] == MVT::iPTR || InVT.TypeVec[0] == MVT::iPTRAny) &&
        TypeVec.size() != 1) {
      TypeVec.assign(1, InVT.TypeVec[0]);
      MadeChange = true;
    }

    return MadeChange;
  }

  // If this is a type list and the RHS is a typelist as well, eliminate entries
  // from this list that aren't in the other one.
  TypeSet InputSet(*this);

  TypeVec.clear();
  std::set_intersection(InputSet.TypeVec.begin(), InputSet.TypeVec.end(),
                        InVT.TypeVec.begin(), InVT.TypeVec.end(),
                        std::back_inserter(TypeVec));

  // If the intersection is the same size as the original set then we're done.
  if (TypeVec.size() == InputSet.TypeVec.size())
    return false;

  // If we removed all of our types, we have a type contradiction.
  if (!TypeVec.empty())
    return true;

  // FIXME: Really want an SMLoc here!
  TP.error("Type inference contradiction found, merging '" +
           InVT.getName() + "' into '" + InputSet.getName() + "'");
  return false;
}

/// EnforceInteger - Remove all non-integer types from this set.
bool EEVT::TypeSet::EnforceInteger(TreePattern &TP) {
  if (TP.hasError())
    return false;
  // If we know nothing, then get the full set.
  if (TypeVec.empty())
    return FillWithPossibleTypes(TP, isInteger, "integer");

  if (!hasFloatingPointTypes())
    return false;

  TypeSet InputSet(*this);

  // Filter out all the fp types.
  TypeVec.erase(std::remove_if(TypeVec.begin(), TypeVec.end(),
                               std::not1(std::ptr_fun(isInteger))),
                TypeVec.end());

  if (TypeVec.empty()) {
    TP.error("Type inference contradiction found, '" +
             InputSet.getName() + "' needs to be integer");
    return false;
  }
  return true;
}

/// EnforceFloatingPoint - Remove all integer types from this set.
bool EEVT::TypeSet::EnforceFloatingPoint(TreePattern &TP) {
  if (TP.hasError())
    return false;
  // If we know nothing, then get the full set.
  if (TypeVec.empty())
    return FillWithPossibleTypes(TP, isFloatingPoint, "floating point");

  if (!hasIntegerTypes())
    return false;

  TypeSet InputSet(*this);

  // Filter out all the integer types.
  TypeVec.erase(std::remove_if(TypeVec.begin(), TypeVec.end(),
                               std::not1(std::ptr_fun(isFloatingPoint))),
                TypeVec.end());

  if (TypeVec.empty()) {
    TP.error("Type inference contradiction found, '" +
             InputSet.getName() + "' needs to be floating point");
    return false;
  }
  return true;
}

/// EnforceScalar - Remove all vector types from this.
bool EEVT::TypeSet::EnforceScalar(TreePattern &TP) {
  if (TP.hasError())
    return false;

  // If we know nothing, then get the full set.
  if (TypeVec.empty())
    return FillWithPossibleTypes(TP, isScalar, "scalar");

  if (!hasVectorTypes())
    return false;

  TypeSet InputSet(*this);

  // Filter out all the vector types.
  TypeVec.erase(std::remove_if(TypeVec.begin(), TypeVec.end(),
                               std::not1(std::ptr_fun(isScalar))),
                TypeVec.end());

  if (TypeVec.empty()) {
    TP.error("Type inference contradiction found, '" +
             InputSet.getName() + "' needs to be scalar");
    return false;
  }
  return true;
}

/// EnforceVector - Remove all vector types from this.
bool EEVT::TypeSet::EnforceVector(TreePattern &TP) {
  if (TP.hasError())
    return false;

  // If we know nothing, then get the full set.
  if (TypeVec.empty())
    return FillWithPossibleTypes(TP, isVector, "vector");

  TypeSet InputSet(*this);
  bool MadeChange = false;

  // Filter out all the scalar types.
  TypeVec.erase(std::remove_if(TypeVec.begin(), TypeVec.end(),
                               std::not1(std::ptr_fun(isVector))),
                TypeVec.end());

  if (TypeVec.empty()) {
    TP.error("Type inference contradiction found, '" +
             InputSet.getName() + "' needs to be a vector");
    return false;
  }
  return MadeChange;
}



/// EnforceSmallerThan - 'this' must be a smaller VT than Other. For vectors
/// this should be based on the element type. Update this and other based on
/// this information.
bool EEVT::TypeSet::EnforceSmallerThan(EEVT::TypeSet &Other, TreePattern &TP) {
  if (TP.hasError())
    return false;

  // Both operands must be integer or FP, but we don't care which.
  bool MadeChange = false;

  if (isCompletelyUnknown())
    MadeChange = FillWithPossibleTypes(TP);

  if (Other.isCompletelyUnknown())
    MadeChange = Other.FillWithPossibleTypes(TP);

  // If one side is known to be integer or known to be FP but the other side has
  // no information, get at least the type integrality info in there.
  if (!hasFloatingPointTypes())
    MadeChange |= Other.EnforceInteger(TP);
  else if (!hasIntegerTypes())
    MadeChange |= Other.EnforceFloatingPoint(TP);
  if (!Other.hasFloatingPointTypes())
    MadeChange |= EnforceInteger(TP);
  else if (!Other.hasIntegerTypes())
    MadeChange |= EnforceFloatingPoint(TP);

  assert(!isCompletelyUnknown() && !Other.isCompletelyUnknown() &&
         "Should have a type list now");

  // If one contains vectors but the other doesn't pull vectors out.
  if (!hasVectorTypes())
    MadeChange |= Other.EnforceScalar(TP);
  else if (!hasScalarTypes())
    MadeChange |= Other.EnforceVector(TP);
  if (!Other.hasVectorTypes())
    MadeChange |= EnforceScalar(TP);
  else if (!Other.hasScalarTypes())
    MadeChange |= EnforceVector(TP);

  // This code does not currently handle nodes which have multiple types,
  // where some types are integer, and some are fp.  Assert that this is not
  // the case.
  assert(!(hasIntegerTypes() && hasFloatingPointTypes()) &&
         !(Other.hasIntegerTypes() && Other.hasFloatingPointTypes()) &&
         "SDTCisOpSmallerThanOp does not handle mixed int/fp types!");

  if (TP.hasError())
    return false;

  // Okay, find the smallest type from current set and remove anything the
  // same or smaller from the other set. We need to ensure that the scalar
  // type size is smaller than the scalar size of the smallest type. For
  // vectors, we also need to make sure that the total size is no larger than
  // the size of the smallest type.
  {
    TypeSet InputSet(Other);
    MVT Smallest = TypeVec[0];
    auto I = std::remove_if(Other.TypeVec.begin(), Other.TypeVec.end(),
      [Smallest](MVT OtherVT) {
        // Don't compare vector and non-vector types.
        if (OtherVT.isVector() != Smallest.isVector())
          return false;
        // The getSizeInBits() check here is only needed for vectors, but is
        // a subset of the scalar check for scalars so no need to qualify.
        return OtherVT.getScalarSizeInBits() <= Smallest.getScalarSizeInBits()||
               OtherVT.getSizeInBits() < Smallest.getSizeInBits();
      });
    MadeChange |= I != Other.TypeVec.end(); // If we're about to remove types.
    Other.TypeVec.erase(I, Other.TypeVec.end());

    if (Other.TypeVec.empty()) {
      TP.error("Type inference contradiction found, '" + InputSet.getName() +
               "' has nothing larger than '" + getName() +"'!");
      return false;
    }
  }

  // Okay, find the largest type from the other set and remove anything the
  // same or smaller from the current set. We need to ensure that the scalar
  // type size is larger than the scalar size of the largest type. For
  // vectors, we also need to make sure that the total size is no smaller than
  // the size of the largest type.
  {
    TypeSet InputSet(*this);
    MVT Largest = Other.TypeVec[Other.TypeVec.size()-1];
    auto I = std::remove_if(TypeVec.begin(), TypeVec.end(),
      [Largest](MVT OtherVT) {
        // Don't compare vector and non-vector types.
        if (OtherVT.isVector() != Largest.isVector())
          return false;
        return OtherVT.getScalarSizeInBits() >= Largest.getScalarSizeInBits() ||
               OtherVT.getSizeInBits() > Largest.getSizeInBits();
      });
    MadeChange |= I != TypeVec.end(); // If we're about to remove types.
    TypeVec.erase(I, TypeVec.end());

    if (TypeVec.empty()) {
      TP.error("Type inference contradiction found, '" + InputSet.getName() +
               "' has nothing smaller than '" + Other.getName() +"'!");
      return false;
    }
  }

  return MadeChange;
}

/// EnforceVectorEltTypeIs - 'this' is now constrained to be a vector type
/// whose element is specified by VTOperand.
bool EEVT::TypeSet::EnforceVectorEltTypeIs(MVT::SimpleValueType VT,
                                           TreePattern &TP) {
  bool MadeChange = false;

  MadeChange |= EnforceVector(TP);

  TypeSet InputSet(*this);

  // Filter out all the types which don't have the right element type.
  auto I = std::remove_if(TypeVec.begin(), TypeVec.end(),
    [VT](MVT VVT) {
      return VVT.getVectorElementType().SimpleTy != VT;
    });
  MadeChange |= I != TypeVec.end();
  TypeVec.erase(I, TypeVec.end());

  if (TypeVec.empty()) {  // FIXME: Really want an SMLoc here!
    TP.error("Type inference contradiction found, forcing '" +
             InputSet.getName() + "' to have a vector element of type " +
             getEnumName(VT));
    return false;
  }

  return MadeChange;
}

/// EnforceVectorEltTypeIs - 'this' is now constrained to be a vector type
/// whose element is specified by VTOperand.
bool EEVT::TypeSet::EnforceVectorEltTypeIs(EEVT::TypeSet &VTOperand,
                                           TreePattern &TP) {
  if (TP.hasError())
    return false;

  // "This" must be a vector and "VTOperand" must be a scalar.
  bool MadeChange = false;
  MadeChange |= EnforceVector(TP);
  MadeChange |= VTOperand.EnforceScalar(TP);

  // If we know the vector type, it forces the scalar to agree.
  if (isConcrete()) {
    MVT IVT = getConcrete();
    IVT = IVT.getVectorElementType();
    return MadeChange || VTOperand.MergeInTypeInfo(IVT.SimpleTy, TP);
  }

  // If the scalar type is known, filter out vector types whose element types
  // disagree.
  if (!VTOperand.isConcrete())
    return MadeChange;

  MVT::SimpleValueType VT = VTOperand.getConcrete();

  MadeChange |= EnforceVectorEltTypeIs(VT, TP);

  return MadeChange;
}

/// EnforceVectorSubVectorTypeIs - 'this' is now constrained to be a
/// vector type specified by VTOperand.
bool EEVT::TypeSet::EnforceVectorSubVectorTypeIs(EEVT::TypeSet &VTOperand,
                                                 TreePattern &TP) {
  if (TP.hasError())
    return false;

  // "This" must be a vector and "VTOperand" must be a vector.
  bool MadeChange = false;
  MadeChange |= EnforceVector(TP);
  MadeChange |= VTOperand.EnforceVector(TP);

  // If one side is known to be integer or known to be FP but the other side has
  // no information, get at least the type integrality info in there.
  if (!hasFloatingPointTypes())
    MadeChange |= VTOperand.EnforceInteger(TP);
  else if (!hasIntegerTypes())
    MadeChange |= VTOperand.EnforceFloatingPoint(TP);
  if (!VTOperand.hasFloatingPointTypes())
    MadeChange |= EnforceInteger(TP);
  else if (!VTOperand.hasIntegerTypes())
    MadeChange |= EnforceFloatingPoint(TP);

  assert(!isCompletelyUnknown() && !VTOperand.isCompletelyUnknown() &&
         "Should have a type list now");

  // If we know the vector type, it forces the scalar types to agree.
  // Also force one vector to have more elements than the other.
  if (isConcrete()) {
    MVT IVT = getConcrete();
    unsigned NumElems = IVT.getVectorNumElements();
    IVT = IVT.getVectorElementType();

    EEVT::TypeSet EltTypeSet(IVT.SimpleTy, TP);
    MadeChange |= VTOperand.EnforceVectorEltTypeIs(EltTypeSet, TP);

    // Only keep types that have less elements than VTOperand.
    TypeSet InputSet(VTOperand);

    auto I = std::remove_if(VTOperand.TypeVec.begin(), VTOperand.TypeVec.end(),
                            [NumElems](MVT VVT) {
                              return VVT.getVectorNumElements() >= NumElems;
                            });
    MadeChange |= I != VTOperand.TypeVec.end();
    VTOperand.TypeVec.erase(I, VTOperand.TypeVec.end());

    if (VTOperand.TypeVec.empty()) {  // FIXME: Really want an SMLoc here!
      TP.error("Type inference contradiction found, forcing '" +
               InputSet.getName() + "' to have less vector elements than '" +
               getName() + "'");
      return false;
    }
  } else if (VTOperand.isConcrete()) {
    MVT IVT = VTOperand.getConcrete();
    unsigned NumElems = IVT.getVectorNumElements();
    IVT = IVT.getVectorElementType();

    EEVT::TypeSet EltTypeSet(IVT.SimpleTy, TP);
    MadeChange |= EnforceVectorEltTypeIs(EltTypeSet, TP);

    // Only keep types that have more elements than 'this'.
    TypeSet InputSet(*this);

    auto I = std::remove_if(TypeVec.begin(), TypeVec.end(),
                            [NumElems](MVT VVT) {
                              return VVT.getVectorNumElements() <= NumElems;
                            });
    MadeChange |= I != TypeVec.end();
    TypeVec.erase(I, TypeVec.end());

    if (TypeVec.empty()) {  // FIXME: Really want an SMLoc here!
      TP.error("Type inference contradiction found, forcing '" +
               InputSet.getName() + "' to have more vector elements than '" +
               VTOperand.getName() + "'");
      return false;
    }
  }

  return MadeChange;
}

/// EnforceVectorSameNumElts - 'this' is now constrained to
/// be a vector with same num elements as VTOperand.
bool EEVT::TypeSet::EnforceVectorSameNumElts(EEVT::TypeSet &VTOperand,
                                             TreePattern &TP) {
  if (TP.hasError())
    return false;

  // "This" must be a vector and "VTOperand" must be a vector.
  bool MadeChange = false;
  MadeChange |= EnforceVector(TP);
  MadeChange |= VTOperand.EnforceVector(TP);

  // If we know one of the vector types, it forces the other type to agree.
  if (isConcrete()) {
    MVT IVT = getConcrete();
    unsigned NumElems = IVT.getVectorNumElements();

    // Only keep types that have same elements as VTOperand.
    TypeSet InputSet(VTOperand);

    auto I = std::remove_if(VTOperand.TypeVec.begin(), VTOperand.TypeVec.end(),
                            [NumElems](MVT VVT) {
                              return VVT.getVectorNumElements() != NumElems;
                            });
    MadeChange |= I != VTOperand.TypeVec.end();
    VTOperand.TypeVec.erase(I, VTOperand.TypeVec.end());

    if (VTOperand.TypeVec.empty()) {  // FIXME: Really want an SMLoc here!
      TP.error("Type inference contradiction found, forcing '" +
               InputSet.getName() + "' to have same number elements as '" +
               getName() + "'");
      return false;
    }
  } else if (VTOperand.isConcrete()) {
    MVT IVT = VTOperand.getConcrete();
    unsigned NumElems = IVT.getVectorNumElements();

    // Only keep types that have same elements as 'this'.
    TypeSet InputSet(*this);

    auto I = std::remove_if(TypeVec.begin(), TypeVec.end(),
                            [NumElems](MVT VVT) {
                              return VVT.getVectorNumElements() != NumElems;
                            });
    MadeChange |= I != TypeVec.end();
    TypeVec.erase(I, TypeVec.end());

    if (TypeVec.empty()) {  // FIXME: Really want an SMLoc here!
      TP.error("Type inference contradiction found, forcing '" +
               InputSet.getName() + "' to have same number elements than '" +
               VTOperand.getName() + "'");
      return false;
    }
  }

  return MadeChange;
}

//===----------------------------------------------------------------------===//
// Helpers for working with extended types.

/// Dependent variable map for CodeGenDAGPattern variant generation
typedef std::map<std::string, int> DepVarMap;

static void FindDepVarsOf(TreePatternNode *N, DepVarMap &DepMap) {
  if (N->isLeaf()) {
    if (isa<DefInit>(N->getLeafValue()))
      DepMap[N->getName()]++;
  } else {
    for (size_t i = 0, e = N->getNumChildren(); i != e; ++i)
      FindDepVarsOf(N->getChild(i), DepMap);
  }
}
  
/// Find dependent variables within child patterns
static void FindDepVars(TreePatternNode *N, MultipleUseVarSet &DepVars) {
  DepVarMap depcounts;
  FindDepVarsOf(N, depcounts);
  for (const std::pair<std::string, int> &Pair : depcounts) {
    if (Pair.second > 1)
      DepVars.insert(Pair.first);
  }
}

#ifndef NDEBUG
/// Dump the dependent variable set:
static void DumpDepVars(MultipleUseVarSet &DepVars) {
  if (DepVars.empty()) {
    DEBUG(errs() << "<empty set>");
  } else {
    DEBUG(errs() << "[ ");
    for (const std::string &DepVar : DepVars) {
      DEBUG(errs() << DepVar << " ");
    }
    DEBUG(errs() << "]");
  }
}
#endif


//===----------------------------------------------------------------------===//
// TreePredicateFn Implementation
//===----------------------------------------------------------------------===//

/// TreePredicateFn constructor.  Here 'N' is a subclass of PatFrag.
TreePredicateFn::TreePredicateFn(TreePattern *N) : PatFragRec(N) {
  assert((getPredCode().empty() || getImmCode().empty()) &&
        ".td file corrupt: can't have a node predicate *and* an imm predicate");
}

std::string TreePredicateFn::getPredCode() const {
  return PatFragRec->getRecord()->getValueAsString("PredicateCode");
}

std::string TreePredicateFn::getImmCode() const {
  return PatFragRec->getRecord()->getValueAsString("ImmediateCode");
}


/// isAlwaysTrue - Return true if this is a noop predicate.
bool TreePredicateFn::isAlwaysTrue() const {
  return getPredCode().empty() && getImmCode().empty();
}

/// Return the name to use in the generated code to reference this, this is
/// "Predicate_foo" if from a pattern fragment "foo".
std::string TreePredicateFn::getFnName() const {
  return "Predicate_" + PatFragRec->getRecord()->getName();
}

/// getCodeToRunOnSDNode - Return the code for the function body that
/// evaluates this predicate.  The argument is expected to be in "Node",
/// not N.  This handles casting and conversion to a concrete node type as
/// appropriate.
std::string TreePredicateFn::getCodeToRunOnSDNode() const {
  // Handle immediate predicates first.
  std::string ImmCode = getImmCode();
  if (!ImmCode.empty()) {
    std::string Result =
      "    int64_t Imm = cast<ConstantSDNode>(Node)->getSExtValue();\n";
    return Result + ImmCode;
  }
  
  // Handle arbitrary node predicates.
  assert(!getPredCode().empty() && "Don't have any predicate code!");
  std::string ClassName;
  if (PatFragRec->getOnlyTree()->isLeaf())
    ClassName = "SDNode";
  else {
    Record *Op = PatFragRec->getOnlyTree()->getOperator();
    ClassName = PatFragRec->getDAGPatterns().getSDNodeInfo(Op).getSDClassName();
  }
  std::string Result;
  if (ClassName == "SDNode")
    Result = "    SDNode *N = Node;\n";
  else
    Result = "    auto *N = cast<" + ClassName + ">(Node);\n";
  
  return Result + getPredCode();
}

//===----------------------------------------------------------------------===//
// PatternToMatch implementation
//


/// getPatternSize - Return the 'size' of this pattern.  We want to match large
/// patterns before small ones.  This is used to determine the size of a
/// pattern.
static unsigned getPatternSize(const TreePatternNode *P,
                               const CodeGenDAGPatterns &CGP) {
  unsigned Size = 3;  // The node itself.
  // If the root node is a ConstantSDNode, increases its size.
  // e.g. (set R32:$dst, 0).
  if (P->isLeaf() && isa<IntInit>(P->getLeafValue()))
    Size += 2;

  // FIXME: This is a hack to statically increase the priority of patterns
  // which maps a sub-dag to a complex pattern. e.g. favors LEA over ADD.
  // Later we can allow complexity / cost for each pattern to be (optionally)
  // specified. To get best possible pattern match we'll need to dynamically
  // calculate the complexity of all patterns a dag can potentially map to.
  const ComplexPattern *AM = P->getComplexPatternInfo(CGP);
  if (AM) {
    Size += AM->getNumOperands() * 3;

    // We don't want to count any children twice, so return early.
    return Size;
  }

  // If this node has some predicate function that must match, it adds to the
  // complexity of this node.
  if (!P->getPredicateFns().empty())
    ++Size;

  // Count children in the count if they are also nodes.
  for (unsigned i = 0, e = P->getNumChildren(); i != e; ++i) {
    TreePatternNode *Child = P->getChild(i);
    if (!Child->isLeaf() && Child->getNumTypes() &&
        Child->getType(0) != MVT::Other)
      Size += getPatternSize(Child, CGP);
    else if (Child->isLeaf()) {
      if (isa<IntInit>(Child->getLeafValue()))
        Size += 5;  // Matches a ConstantSDNode (+3) and a specific value (+2).
      else if (Child->getComplexPatternInfo(CGP))
        Size += getPatternSize(Child, CGP);
      else if (!Child->getPredicateFns().empty())
        ++Size;
    }
  }

  return Size;
}

/// Compute the complexity metric for the input pattern.  This roughly
/// corresponds to the number of nodes that are covered.
int PatternToMatch::
getPatternComplexity(const CodeGenDAGPatterns &CGP) const {
  return getPatternSize(getSrcPattern(), CGP) + getAddedComplexity();
}


/// getPredicateCheck - Return a single string containing all of this
/// pattern's predicates concatenated with "&&" operators.
///
std::string PatternToMatch::getPredicateCheck() const {
  std::string PredicateCheck;
  for (Init *I : Predicates->getValues()) {
    if (DefInit *Pred = dyn_cast<DefInit>(I)) {
      Record *Def = Pred->getDef();
      if (!Def->isSubClassOf("Predicate")) {
#ifndef NDEBUG
        Def->dump();
#endif
        llvm_unreachable("Unknown predicate type!");
      }
      if (!PredicateCheck.empty())
        PredicateCheck += " && ";
      PredicateCheck += "(" + Def->getValueAsString("CondString") + ")";
    }
  }

  return PredicateCheck;
}

//===----------------------------------------------------------------------===//
// SDTypeConstraint implementation
//

SDTypeConstraint::SDTypeConstraint(Record *R) {
  OperandNo = R->getValueAsInt("OperandNum");

  if (R->isSubClassOf("SDTCisVT")) {
    ConstraintType = SDTCisVT;
    x.SDTCisVT_Info.VT = getValueType(R->getValueAsDef("VT"));
    if (x.SDTCisVT_Info.VT == MVT::isVoid)
      PrintFatalError(R->getLoc(), "Cannot use 'Void' as type to SDTCisVT");

  } else if (R->isSubClassOf("SDTCisPtrTy")) {
    ConstraintType = SDTCisPtrTy;
  } else if (R->isSubClassOf("SDTCisInt")) {
    ConstraintType = SDTCisInt;
  } else if (R->isSubClassOf("SDTCisFP")) {
    ConstraintType = SDTCisFP;
  } else if (R->isSubClassOf("SDTCisVec")) {
    ConstraintType = SDTCisVec;
  } else if (R->isSubClassOf("SDTCisSameAs")) {
    ConstraintType = SDTCisSameAs;
    x.SDTCisSameAs_Info.OtherOperandNum = R->getValueAsInt("OtherOperandNum");
  } else if (R->isSubClassOf("SDTCisVTSmallerThanOp")) {
    ConstraintType = SDTCisVTSmallerThanOp;
    x.SDTCisVTSmallerThanOp_Info.OtherOperandNum =
      R->getValueAsInt("OtherOperandNum");
  } else if (R->isSubClassOf("SDTCisOpSmallerThanOp")) {
    ConstraintType = SDTCisOpSmallerThanOp;
    x.SDTCisOpSmallerThanOp_Info.BigOperandNum =
      R->getValueAsInt("BigOperandNum");
  } else if (R->isSubClassOf("SDTCisEltOfVec")) {
    ConstraintType = SDTCisEltOfVec;
    x.SDTCisEltOfVec_Info.OtherOperandNum = R->getValueAsInt("OtherOpNum");
  } else if (R->isSubClassOf("SDTCisSubVecOfVec")) {
    ConstraintType = SDTCisSubVecOfVec;
    x.SDTCisSubVecOfVec_Info.OtherOperandNum =
      R->getValueAsInt("OtherOpNum");
  } else if (R->isSubClassOf("SDTCVecEltisVT")) {
    ConstraintType = SDTCVecEltisVT;
    x.SDTCVecEltisVT_Info.VT = getValueType(R->getValueAsDef("VT"));
    if (MVT(x.SDTCVecEltisVT_Info.VT).isVector())
      PrintFatalError(R->getLoc(), "Cannot use vector type as SDTCVecEltisVT");
    if (!MVT(x.SDTCVecEltisVT_Info.VT).isInteger() &&
        !MVT(x.SDTCVecEltisVT_Info.VT).isFloatingPoint())
      PrintFatalError(R->getLoc(), "Must use integer or floating point type "
                                   "as SDTCVecEltisVT");
  } else if (R->isSubClassOf("SDTCisSameNumEltsAs")) {
    ConstraintType = SDTCisSameNumEltsAs;
    x.SDTCisSameNumEltsAs_Info.OtherOperandNum =
      R->getValueAsInt("OtherOperandNum");
  } else {
    PrintFatalError("Unrecognized SDTypeConstraint '" + R->getName() + "'!\n");
  }
}

/// getOperandNum - Return the node corresponding to operand #OpNo in tree
/// N, and the result number in ResNo.
static TreePatternNode *getOperandNum(unsigned OpNo, TreePatternNode *N,
                                      const SDNodeInfo &NodeInfo,
                                      unsigned &ResNo) {
  unsigned NumResults = NodeInfo.getNumResults();
  if (OpNo < NumResults) {
    ResNo = OpNo;
    return N;
  }

  OpNo -= NumResults;

  if (OpNo >= N->getNumChildren()) {
    std::string S;
    raw_string_ostream OS(S);
    OS << "Invalid operand number in type constraint "
           << (OpNo+NumResults) << " ";
    N->print(OS);
    PrintFatalError(OS.str());
  }

  return N->getChild(OpNo);
}

/// ApplyTypeConstraint - Given a node in a pattern, apply this type
/// constraint to the nodes operands.  This returns true if it makes a
/// change, false otherwise.  If a type contradiction is found, flag an error.
bool SDTypeConstraint::ApplyTypeConstraint(TreePatternNode *N,
                                           const SDNodeInfo &NodeInfo,
                                           TreePattern &TP) const {
  if (TP.hasError())
    return false;

  unsigned ResNo = 0; // The result number being referenced.
  TreePatternNode *NodeToApply = getOperandNum(OperandNo, N, NodeInfo, ResNo);

  switch (ConstraintType) {
  case SDTCisVT:
    // Operand must be a particular type.
    return NodeToApply->UpdateNodeType(ResNo, x.SDTCisVT_Info.VT, TP);
  case SDTCisPtrTy:
    // Operand must be same as target pointer type.
    return NodeToApply->UpdateNodeType(ResNo, MVT::iPTR, TP);
  case SDTCisInt:
    // Require it to be one of the legal integer VTs.
    return NodeToApply->getExtType(ResNo).EnforceInteger(TP);
  case SDTCisFP:
    // Require it to be one of the legal fp VTs.
    return NodeToApply->getExtType(ResNo).EnforceFloatingPoint(TP);
  case SDTCisVec:
    // Require it to be one of the legal vector VTs.
    return NodeToApply->getExtType(ResNo).EnforceVector(TP);
  case SDTCisSameAs: {
    unsigned OResNo = 0;
    TreePatternNode *OtherNode =
      getOperandNum(x.SDTCisSameAs_Info.OtherOperandNum, N, NodeInfo, OResNo);
    return NodeToApply->UpdateNodeType(ResNo, OtherNode->getExtType(OResNo),TP)|
           OtherNode->UpdateNodeType(OResNo,NodeToApply->getExtType(ResNo),TP);
  }
  case SDTCisVTSmallerThanOp: {
    // The NodeToApply must be a leaf node that is a VT.  OtherOperandNum must
    // have an integer type that is smaller than the VT.
    if (!NodeToApply->isLeaf() ||
        !isa<DefInit>(NodeToApply->getLeafValue()) ||
        !static_cast<DefInit*>(NodeToApply->getLeafValue())->getDef()
               ->isSubClassOf("ValueType")) {
      TP.error(N->getOperator()->getName() + " expects a VT operand!");
      return false;
    }
    MVT::SimpleValueType VT =
     getValueType(static_cast<DefInit*>(NodeToApply->getLeafValue())->getDef());

    EEVT::TypeSet TypeListTmp(VT, TP);

    unsigned OResNo = 0;
    TreePatternNode *OtherNode =
      getOperandNum(x.SDTCisVTSmallerThanOp_Info.OtherOperandNum, N, NodeInfo,
                    OResNo);

    return TypeListTmp.EnforceSmallerThan(OtherNode->getExtType(OResNo), TP);
  }
  case SDTCisOpSmallerThanOp: {
    unsigned BResNo = 0;
    TreePatternNode *BigOperand =
      getOperandNum(x.SDTCisOpSmallerThanOp_Info.BigOperandNum, N, NodeInfo,
                    BResNo);
    return NodeToApply->getExtType(ResNo).
                  EnforceSmallerThan(BigOperand->getExtType(BResNo), TP);
  }
  case SDTCisEltOfVec: {
    unsigned VResNo = 0;
    TreePatternNode *VecOperand =
      getOperandNum(x.SDTCisEltOfVec_Info.OtherOperandNum, N, NodeInfo,
                    VResNo);

    // Filter vector types out of VecOperand that don't have the right element
    // type.
    return VecOperand->getExtType(VResNo).
      EnforceVectorEltTypeIs(NodeToApply->getExtType(ResNo), TP);
  }
  case SDTCisSubVecOfVec: {
    unsigned VResNo = 0;
    TreePatternNode *BigVecOperand =
      getOperandNum(x.SDTCisSubVecOfVec_Info.OtherOperandNum, N, NodeInfo,
                    VResNo);

    // Filter vector types out of BigVecOperand that don't have the
    // right subvector type.
    return BigVecOperand->getExtType(VResNo).
      EnforceVectorSubVectorTypeIs(NodeToApply->getExtType(ResNo), TP);
  }
  case SDTCVecEltisVT: {
    return NodeToApply->getExtType(ResNo).
      EnforceVectorEltTypeIs(x.SDTCVecEltisVT_Info.VT, TP);
  }
  case SDTCisSameNumEltsAs: {
    unsigned OResNo = 0;
    TreePatternNode *OtherNode =
      getOperandNum(x.SDTCisSameNumEltsAs_Info.OtherOperandNum,
                    N, NodeInfo, OResNo);
    return OtherNode->getExtType(OResNo).
      EnforceVectorSameNumElts(NodeToApply->getExtType(ResNo), TP);
  }
  }
  llvm_unreachable("Invalid ConstraintType!");
}

// Update the node type to match an instruction operand or result as specified
// in the ins or outs lists on the instruction definition. Return true if the
// type was actually changed.
bool TreePatternNode::UpdateNodeTypeFromInst(unsigned ResNo,
                                             Record *Operand,
                                             TreePattern &TP) {
  // The 'unknown' operand indicates that types should be inferred from the
  // context.
  if (Operand->isSubClassOf("unknown_class"))
    return false;

  // The Operand class specifies a type directly.
  if (Operand->isSubClassOf("Operand"))
    return UpdateNodeType(ResNo, getValueType(Operand->getValueAsDef("Type")),
                          TP);

  // PointerLikeRegClass has a type that is determined at runtime.
  if (Operand->isSubClassOf("PointerLikeRegClass"))
    return UpdateNodeType(ResNo, MVT::iPTR, TP);

  // Both RegisterClass and RegisterOperand operands derive their types from a
  // register class def.
  Record *RC = nullptr;
  if (Operand->isSubClassOf("RegisterClass"))
    RC = Operand;
  else if (Operand->isSubClassOf("RegisterOperand"))
    RC = Operand->getValueAsDef("RegClass");

  assert(RC && "Unknown operand type");
  CodeGenTarget &Tgt = TP.getDAGPatterns().getTargetInfo();
  return UpdateNodeType(ResNo, Tgt.getRegisterClass(RC).getValueTypes(), TP);
}


//===----------------------------------------------------------------------===//
// SDNodeInfo implementation
//
SDNodeInfo::SDNodeInfo(Record *R) : Def(R) {
  EnumName    = R->getValueAsString("Opcode");
  SDClassName = R->getValueAsString("SDClass");
  Record *TypeProfile = R->getValueAsDef("TypeProfile");
  NumResults = TypeProfile->getValueAsInt("NumResults");
  NumOperands = TypeProfile->getValueAsInt("NumOperands");

  // Parse the properties.
  Properties = 0;
  for (Record *Property : R->getValueAsListOfDefs("Properties")) {
    if (Property->getName() == "SDNPCommutative") {
      Properties |= 1 << SDNPCommutative;
    } else if (Property->getName() == "SDNPAssociative") {
      Properties |= 1 << SDNPAssociative;
    } else if (Property->getName() == "SDNPHasChain") {
      Properties |= 1 << SDNPHasChain;
    } else if (Property->getName() == "SDNPOutGlue") {
      Properties |= 1 << SDNPOutGlue;
    } else if (Property->getName() == "SDNPInGlue") {
      Properties |= 1 << SDNPInGlue;
    } else if (Property->getName() == "SDNPOptInGlue") {
      Properties |= 1 << SDNPOptInGlue;
    } else if (Property->getName() == "SDNPMayStore") {
      Properties |= 1 << SDNPMayStore;
    } else if (Property->getName() == "SDNPMayLoad") {
      Properties |= 1 << SDNPMayLoad;
    } else if (Property->getName() == "SDNPSideEffect") {
      Properties |= 1 << SDNPSideEffect;
    } else if (Property->getName() == "SDNPMemOperand") {
      Properties |= 1 << SDNPMemOperand;
    } else if (Property->getName() == "SDNPVariadic") {
      Properties |= 1 << SDNPVariadic;
    } else {
      PrintFatalError("Unknown SD Node property '" +
                      Property->getName() + "' on node '" +
                      R->getName() + "'!");
    }
  }


  // Parse the type constraints.
  std::vector<Record*> ConstraintList =
    TypeProfile->getValueAsListOfDefs("Constraints");
  TypeConstraints.assign(ConstraintList.begin(), ConstraintList.end());
}

/// getKnownType - If the type constraints on this node imply a fixed type
/// (e.g. all stores return void, etc), then return it as an
/// MVT::SimpleValueType.  Otherwise, return EEVT::Other.
MVT::SimpleValueType SDNodeInfo::getKnownType(unsigned ResNo) const {
  unsigned NumResults = getNumResults();
  assert(NumResults <= 1 &&
         "We only work with nodes with zero or one result so far!");
  assert(ResNo == 0 && "Only handles single result nodes so far");

  for (const SDTypeConstraint &Constraint : TypeConstraints) {
    // Make sure that this applies to the correct node result.
    if (Constraint.OperandNo >= NumResults)  // FIXME: need value #
      continue;

    switch (Constraint.ConstraintType) {
    default: break;
    case SDTypeConstraint::SDTCisVT:
      return Constraint.x.SDTCisVT_Info.VT;
    case SDTypeConstraint::SDTCisPtrTy:
      return MVT::iPTR;
    }
  }
  return MVT::Other;
}

//===----------------------------------------------------------------------===//
// TreePatternNode implementation
//

TreePatternNode::~TreePatternNode() {
#if 0 // FIXME: implement refcounted tree nodes!
  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    delete getChild(i);
#endif
}

static unsigned GetNumNodeResults(Record *Operator, CodeGenDAGPatterns &CDP) {
  if (Operator->getName() == "set" ||
      Operator->getName() == "implicit")
    return 0;  // All return nothing.

  if (Operator->isSubClassOf("Intrinsic"))
    return CDP.getIntrinsic(Operator).IS.RetVTs.size();

  if (Operator->isSubClassOf("SDNode"))
    return CDP.getSDNodeInfo(Operator).getNumResults();

  if (Operator->isSubClassOf("PatFrag")) {
    // If we've already parsed this pattern fragment, get it.  Otherwise, handle
    // the forward reference case where one pattern fragment references another
    // before it is processed.
    if (TreePattern *PFRec = CDP.getPatternFragmentIfRead(Operator))
      return PFRec->getOnlyTree()->getNumTypes();

    // Get the result tree.
    DagInit *Tree = Operator->getValueAsDag("Fragment");
    Record *Op = nullptr;
    if (Tree)
      if (DefInit *DI = dyn_cast<DefInit>(Tree->getOperator()))
        Op = DI->getDef();
    assert(Op && "Invalid Fragment");
    return GetNumNodeResults(Op, CDP);
  }

  if (Operator->isSubClassOf("Instruction")) {
    CodeGenInstruction &InstInfo = CDP.getTargetInfo().getInstruction(Operator);

    unsigned NumDefsToAdd = InstInfo.Operands.NumDefs;

    // Subtract any defaulted outputs.
    for (unsigned i = 0; i != InstInfo.Operands.NumDefs; ++i) {
      Record *OperandNode = InstInfo.Operands[i].Rec;

      if (OperandNode->isSubClassOf("OperandWithDefaultOps") &&
          !CDP.getDefaultOperand(OperandNode).DefaultOps.empty())
        --NumDefsToAdd;
    }

    // Add on one implicit def if it has a resolvable type.
    if (InstInfo.HasOneImplicitDefWithKnownVT(CDP.getTargetInfo()) !=MVT::Other)
      ++NumDefsToAdd;
    return NumDefsToAdd;
  }

  if (Operator->isSubClassOf("SDNodeXForm"))
    return 1;  // FIXME: Generalize SDNodeXForm

  if (Operator->isSubClassOf("ValueType"))
    return 1;  // A type-cast of one result.

  if (Operator->isSubClassOf("ComplexPattern"))
    return 1;

  Operator->dump();
  PrintFatalError("Unhandled node in GetNumNodeResults");
}

void TreePatternNode::print(raw_ostream &OS) const {
  if (isLeaf())
    OS << *getLeafValue();
  else
    OS << '(' << getOperator()->getName();

  for (unsigned i = 0, e = Types.size(); i != e; ++i)
    OS << ':' << getExtType(i).getName();

  if (!isLeaf()) {
    if (getNumChildren() != 0) {
      OS << " ";
      getChild(0)->print(OS);
      for (unsigned i = 1, e = getNumChildren(); i != e; ++i) {
        OS << ", ";
        getChild(i)->print(OS);
      }
    }
    OS << ")";
  }

  for (const TreePredicateFn &Pred : PredicateFns)
    OS << "<<P:" << Pred.getFnName() << ">>";
  if (TransformFn)
    OS << "<<X:" << TransformFn->getName() << ">>";
  if (!getName().empty())
    OS << ":$" << getName();

}
void TreePatternNode::dump() const {
  print(errs());
}

/// isIsomorphicTo - Return true if this node is recursively
/// isomorphic to the specified node.  For this comparison, the node's
/// entire state is considered. The assigned name is ignored, since
/// nodes with differing names are considered isomorphic. However, if
/// the assigned name is present in the dependent variable set, then
/// the assigned name is considered significant and the node is
/// isomorphic if the names match.
bool TreePatternNode::isIsomorphicTo(const TreePatternNode *N,
                                     const MultipleUseVarSet &DepVars) const {
  if (N == this) return true;
  if (N->isLeaf() != isLeaf() || getExtTypes() != N->getExtTypes() ||
      getPredicateFns() != N->getPredicateFns() ||
      getTransformFn() != N->getTransformFn())
    return false;

  if (isLeaf()) {
    if (DefInit *DI = dyn_cast<DefInit>(getLeafValue())) {
      if (DefInit *NDI = dyn_cast<DefInit>(N->getLeafValue())) {
        return ((DI->getDef() == NDI->getDef())
                && (DepVars.find(getName()) == DepVars.end()
                    || getName() == N->getName()));
      }
    }
    return getLeafValue() == N->getLeafValue();
  }

  if (N->getOperator() != getOperator() ||
      N->getNumChildren() != getNumChildren()) return false;
  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    if (!getChild(i)->isIsomorphicTo(N->getChild(i), DepVars))
      return false;
  return true;
}

/// clone - Make a copy of this tree and all of its children.
///
TreePatternNode *TreePatternNode::clone() const {
  TreePatternNode *New;
  if (isLeaf()) {
    New = new TreePatternNode(getLeafValue(), getNumTypes());
  } else {
    std::vector<TreePatternNode*> CChildren;
    CChildren.reserve(Children.size());
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
      CChildren.push_back(getChild(i)->clone());
    New = new TreePatternNode(getOperator(), CChildren, getNumTypes());
  }
  New->setName(getName());
  New->Types = Types;
  New->setPredicateFns(getPredicateFns());
  New->setTransformFn(getTransformFn());
  return New;
}

/// RemoveAllTypes - Recursively strip all the types of this tree.
void TreePatternNode::RemoveAllTypes() {
  // Reset to unknown type.
  std::fill(Types.begin(), Types.end(), EEVT::TypeSet());
  if (isLeaf()) return;
  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    getChild(i)->RemoveAllTypes();
}


/// SubstituteFormalArguments - Replace the formal arguments in this tree
/// with actual values specified by ArgMap.
void TreePatternNode::
SubstituteFormalArguments(std::map<std::string, TreePatternNode*> &ArgMap) {
  if (isLeaf()) return;

  for (unsigned i = 0, e = getNumChildren(); i != e; ++i) {
    TreePatternNode *Child = getChild(i);
    if (Child->isLeaf()) {
      Init *Val = Child->getLeafValue();
      // Note that, when substituting into an output pattern, Val might be an
      // UnsetInit.
      if (isa<UnsetInit>(Val) || (isa<DefInit>(Val) &&
          cast<DefInit>(Val)->getDef()->getName() == "node")) {
        // We found a use of a formal argument, replace it with its value.
        TreePatternNode *NewChild = ArgMap[Child->getName()];
        assert(NewChild && "Couldn't find formal argument!");
        assert((Child->getPredicateFns().empty() ||
                NewChild->getPredicateFns() == Child->getPredicateFns()) &&
               "Non-empty child predicate clobbered!");
        setChild(i, NewChild);
      }
    } else {
      getChild(i)->SubstituteFormalArguments(ArgMap);
    }
  }
}


/// InlinePatternFragments - If this pattern refers to any pattern
/// fragments, inline them into place, giving us a pattern without any
/// PatFrag references.
TreePatternNode *TreePatternNode::InlinePatternFragments(TreePattern &TP) {
  if (TP.hasError())
    return nullptr;

  if (isLeaf())
     return this;  // nothing to do.
  Record *Op = getOperator();

  if (!Op->isSubClassOf("PatFrag")) {
    // Just recursively inline children nodes.
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i) {
      TreePatternNode *Child = getChild(i);
      TreePatternNode *NewChild = Child->InlinePatternFragments(TP);

      assert((Child->getPredicateFns().empty() ||
              NewChild->getPredicateFns() == Child->getPredicateFns()) &&
             "Non-empty child predicate clobbered!");

      setChild(i, NewChild);
    }
    return this;
  }

  // Otherwise, we found a reference to a fragment.  First, look up its
  // TreePattern record.
  TreePattern *Frag = TP.getDAGPatterns().getPatternFragment(Op);

  // Verify that we are passing the right number of operands.
  if (Frag->getNumArgs() != Children.size()) {
    TP.error("'" + Op->getName() + "' fragment requires " +
             utostr(Frag->getNumArgs()) + " operands!");
    return nullptr;
  }

  TreePatternNode *FragTree = Frag->getOnlyTree()->clone();

  TreePredicateFn PredFn(Frag);
  if (!PredFn.isAlwaysTrue())
    FragTree->addPredicateFn(PredFn);

  // Resolve formal arguments to their actual value.
  if (Frag->getNumArgs()) {
    // Compute the map of formal to actual arguments.
    std::map<std::string, TreePatternNode*> ArgMap;
    for (unsigned i = 0, e = Frag->getNumArgs(); i != e; ++i)
      ArgMap[Frag->getArgName(i)] = getChild(i)->InlinePatternFragments(TP);

    FragTree->SubstituteFormalArguments(ArgMap);
  }

  FragTree->setName(getName());
  for (unsigned i = 0, e = Types.size(); i != e; ++i)
    FragTree->UpdateNodeType(i, getExtType(i), TP);

  // Transfer in the old predicates.
  for (const TreePredicateFn &Pred : getPredicateFns())
    FragTree->addPredicateFn(Pred);

  // Get a new copy of this fragment to stitch into here.
  //delete this;    // FIXME: implement refcounting!

  // The fragment we inlined could have recursive inlining that is needed.  See
  // if there are any pattern fragments in it and inline them as needed.
  return FragTree->InlinePatternFragments(TP);
}

/// getImplicitType - Check to see if the specified record has an implicit
/// type which should be applied to it.  This will infer the type of register
/// references from the register file information, for example.
///
/// When Unnamed is set, return the type of a DAG operand with no name, such as
/// the F8RC register class argument in:
///
///   (COPY_TO_REGCLASS GPR:$src, F8RC)
///
/// When Unnamed is false, return the type of a named DAG operand such as the
/// GPR:$src operand above.
///
static EEVT::TypeSet getImplicitType(Record *R, unsigned ResNo,
                                     bool NotRegisters,
                                     bool Unnamed,
                                     TreePattern &TP) {
  // Check to see if this is a register operand.
  if (R->isSubClassOf("RegisterOperand")) {
    assert(ResNo == 0 && "Regoperand ref only has one result!");
    if (NotRegisters)
      return EEVT::TypeSet(); // Unknown.
    Record *RegClass = R->getValueAsDef("RegClass");
    const CodeGenTarget &T = TP.getDAGPatterns().getTargetInfo();
    return EEVT::TypeSet(T.getRegisterClass(RegClass).getValueTypes());
  }

  // Check to see if this is a register or a register class.
  if (R->isSubClassOf("RegisterClass")) {
    assert(ResNo == 0 && "Regclass ref only has one result!");
    // An unnamed register class represents itself as an i32 immediate, for
    // example on a COPY_TO_REGCLASS instruction.
    if (Unnamed)
      return EEVT::TypeSet(MVT::i32, TP);

    // In a named operand, the register class provides the possible set of
    // types.
    if (NotRegisters)
      return EEVT::TypeSet(); // Unknown.
    const CodeGenTarget &T = TP.getDAGPatterns().getTargetInfo();
    return EEVT::TypeSet(T.getRegisterClass(R).getValueTypes());
  }

  if (R->isSubClassOf("PatFrag")) {
    assert(ResNo == 0 && "FIXME: PatFrag with multiple results?");
    // Pattern fragment types will be resolved when they are inlined.
    return EEVT::TypeSet(); // Unknown.
  }

  if (R->isSubClassOf("Register")) {
    assert(ResNo == 0 && "Registers only produce one result!");
    if (NotRegisters)
      return EEVT::TypeSet(); // Unknown.
    const CodeGenTarget &T = TP.getDAGPatterns().getTargetInfo();
    return EEVT::TypeSet(T.getRegisterVTs(R));
  }

  if (R->isSubClassOf("SubRegIndex")) {
    assert(ResNo == 0 && "SubRegisterIndices only produce one result!");
    return EEVT::TypeSet(MVT::i32, TP);
  }

  if (R->isSubClassOf("ValueType")) {
    assert(ResNo == 0 && "This node only has one result!");
    // An unnamed VTSDNode represents itself as an MVT::Other immediate.
    //
    //   (sext_inreg GPR:$src, i16)
    //                         ~~~
    if (Unnamed)
      return EEVT::TypeSet(MVT::Other, TP);
    // With a name, the ValueType simply provides the type of the named
    // variable.
    //
    //   (sext_inreg i32:$src, i16)
    //               ~~~~~~~~
    if (NotRegisters)
      return EEVT::TypeSet(); // Unknown.
    return EEVT::TypeSet(getValueType(R), TP);
  }

  if (R->isSubClassOf("CondCode")) {
    assert(ResNo == 0 && "This node only has one result!");
    // Using a CondCodeSDNode.
    return EEVT::TypeSet(MVT::Other, TP);
  }

  if (R->isSubClassOf("ComplexPattern")) {
    assert(ResNo == 0 && "FIXME: ComplexPattern with multiple results?");
    if (NotRegisters)
      return EEVT::TypeSet(); // Unknown.
   return EEVT::TypeSet(TP.getDAGPatterns().getComplexPattern(R).getValueType(),
                         TP);
  }
  if (R->isSubClassOf("PointerLikeRegClass")) {
    assert(ResNo == 0 && "Regclass can only have one result!");
    return EEVT::TypeSet(MVT::iPTR, TP);
  }

  if (R->getName() == "node" || R->getName() == "srcvalue" ||
      R->getName() == "zero_reg") {
    // Placeholder.
    return EEVT::TypeSet(); // Unknown.
  }

  if (R->isSubClassOf("Operand"))
    return EEVT::TypeSet(getValueType(R->getValueAsDef("Type")));

  TP.error("Unknown node flavor used in pattern: " + R->getName());
  return EEVT::TypeSet(MVT::Other, TP);
}


/// getIntrinsicInfo - If this node corresponds to an intrinsic, return the
/// CodeGenIntrinsic information for it, otherwise return a null pointer.
const CodeGenIntrinsic *TreePatternNode::
getIntrinsicInfo(const CodeGenDAGPatterns &CDP) const {
  if (getOperator() != CDP.get_intrinsic_void_sdnode() &&
      getOperator() != CDP.get_intrinsic_w_chain_sdnode() &&
      getOperator() != CDP.get_intrinsic_wo_chain_sdnode())
    return nullptr;

  unsigned IID = cast<IntInit>(getChild(0)->getLeafValue())->getValue();
  return &CDP.getIntrinsicInfo(IID);
}

/// getComplexPatternInfo - If this node corresponds to a ComplexPattern,
/// return the ComplexPattern information, otherwise return null.
const ComplexPattern *
TreePatternNode::getComplexPatternInfo(const CodeGenDAGPatterns &CGP) const {
  Record *Rec;
  if (isLeaf()) {
    DefInit *DI = dyn_cast<DefInit>(getLeafValue());
    if (!DI)
      return nullptr;
    Rec = DI->getDef();
  } else
    Rec = getOperator();

  if (!Rec->isSubClassOf("ComplexPattern"))
    return nullptr;
  return &CGP.getComplexPattern(Rec);
}

unsigned TreePatternNode::getNumMIResults(const CodeGenDAGPatterns &CGP) const {
  // A ComplexPattern specifically declares how many results it fills in.
  if (const ComplexPattern *CP = getComplexPatternInfo(CGP))
    return CP->getNumOperands();

  // If MIOperandInfo is specified, that gives the count.
  if (isLeaf()) {
    DefInit *DI = dyn_cast<DefInit>(getLeafValue());
    if (DI && DI->getDef()->isSubClassOf("Operand")) {
      DagInit *MIOps = DI->getDef()->getValueAsDag("MIOperandInfo");
      if (MIOps->getNumArgs())
        return MIOps->getNumArgs();
    }
  }

  // Otherwise there is just one result.
  return 1;
}

/// NodeHasProperty - Return true if this node has the specified property.
bool TreePatternNode::NodeHasProperty(SDNP Property,
                                      const CodeGenDAGPatterns &CGP) const {
  if (isLeaf()) {
    if (const ComplexPattern *CP = getComplexPatternInfo(CGP))
      return CP->hasProperty(Property);
    return false;
  }

  Record *Operator = getOperator();
  if (!Operator->isSubClassOf("SDNode")) return false;

  return CGP.getSDNodeInfo(Operator).hasProperty(Property);
}




/// TreeHasProperty - Return true if any node in this tree has the specified
/// property.
bool TreePatternNode::TreeHasProperty(SDNP Property,
                                      const CodeGenDAGPatterns &CGP) const {
  if (NodeHasProperty(Property, CGP))
    return true;
  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    if (getChild(i)->TreeHasProperty(Property, CGP))
      return true;
  return false;
}

/// isCommutativeIntrinsic - Return true if the node corresponds to a
/// commutative intrinsic.
bool
TreePatternNode::isCommutativeIntrinsic(const CodeGenDAGPatterns &CDP) const {
  if (const CodeGenIntrinsic *Int = getIntrinsicInfo(CDP))
    return Int->isCommutative;
  return false;
}

static bool isOperandClass(const TreePatternNode *N, StringRef Class) {
  if (!N->isLeaf())
    return N->getOperator()->isSubClassOf(Class);

  DefInit *DI = dyn_cast<DefInit>(N->getLeafValue());
  if (DI && DI->getDef()->isSubClassOf(Class))
    return true;

  return false;
}

static void emitTooManyOperandsError(TreePattern &TP,
                                     StringRef InstName,
                                     unsigned Expected,
                                     unsigned Actual) {
  TP.error("Instruction '" + InstName + "' was provided " + Twine(Actual) +
           " operands but expected only " + Twine(Expected) + "!");
}

static void emitTooFewOperandsError(TreePattern &TP,
                                    StringRef InstName,
                                    unsigned Actual) {
  TP.error("Instruction '" + InstName +
           "' expects more than the provided " + Twine(Actual) + " operands!");
}

/// ApplyTypeConstraints - Apply all of the type constraints relevant to
/// this node and its children in the tree.  This returns true if it makes a
/// change, false otherwise.  If a type contradiction is found, flag an error.
bool TreePatternNode::ApplyTypeConstraints(TreePattern &TP, bool NotRegisters) {
  if (TP.hasError())
    return false;

  CodeGenDAGPatterns &CDP = TP.getDAGPatterns();
  if (isLeaf()) {
    if (DefInit *DI = dyn_cast<DefInit>(getLeafValue())) {
      // If it's a regclass or something else known, include the type.
      bool MadeChange = false;
      for (unsigned i = 0, e = Types.size(); i != e; ++i)
        MadeChange |= UpdateNodeType(i, getImplicitType(DI->getDef(), i,
                                                        NotRegisters,
                                                        !hasName(), TP), TP);
      return MadeChange;
    }

    if (IntInit *II = dyn_cast<IntInit>(getLeafValue())) {
      assert(Types.size() == 1 && "Invalid IntInit");

      // Int inits are always integers. :)
      bool MadeChange = Types[0].EnforceInteger(TP);

      if (!Types[0].isConcrete())
        return MadeChange;

      MVT::SimpleValueType VT = getType(0);
      if (VT == MVT::iPTR || VT == MVT::iPTRAny)
        return MadeChange;

      unsigned Size = MVT(VT).getSizeInBits();
      // Make sure that the value is representable for this type.
      if (Size >= 32) return MadeChange;

      // Check that the value doesn't use more bits than we have. It must either
      // be a sign- or zero-extended equivalent of the original.
      int64_t SignBitAndAbove = II->getValue() >> (Size - 1);
      if (SignBitAndAbove == -1 || SignBitAndAbove == 0 || SignBitAndAbove == 1)
        return MadeChange;

      TP.error("Integer value '" + itostr(II->getValue()) +
               "' is out of range for type '" + getEnumName(getType(0)) + "'!");
      return false;
    }
    return false;
  }

  // special handling for set, which isn't really an SDNode.
  if (getOperator()->getName() == "set") {
    assert(getNumTypes() == 0 && "Set doesn't produce a value");
    assert(getNumChildren() >= 2 && "Missing RHS of a set?");
    unsigned NC = getNumChildren();

    TreePatternNode *SetVal = getChild(NC-1);
    bool MadeChange = SetVal->ApplyTypeConstraints(TP, NotRegisters);

    for (unsigned i = 0; i < NC-1; ++i) {
      TreePatternNode *Child = getChild(i);
      MadeChange |= Child->ApplyTypeConstraints(TP, NotRegisters);

      // Types of operands must match.
      MadeChange |= Child->UpdateNodeType(0, SetVal->getExtType(i), TP);
      MadeChange |= SetVal->UpdateNodeType(i, Child->getExtType(0), TP);
    }
    return MadeChange;
  }

  if (getOperator()->getName() == "implicit") {
    assert(getNumTypes() == 0 && "Node doesn't produce a value");

    bool MadeChange = false;
    for (unsigned i = 0; i < getNumChildren(); ++i)
      MadeChange = getChild(i)->ApplyTypeConstraints(TP, NotRegisters);
    return MadeChange;
  }

  if (const CodeGenIntrinsic *Int = getIntrinsicInfo(CDP)) {
    bool MadeChange = false;

    // Apply the result type to the node.
    unsigned NumRetVTs = Int->IS.RetVTs.size();
    unsigned NumParamVTs = Int->IS.ParamVTs.size();

    for (unsigned i = 0, e = NumRetVTs; i != e; ++i)
      MadeChange |= UpdateNodeType(i, Int->IS.RetVTs[i], TP);

    if (getNumChildren() != NumParamVTs + 1) {
      TP.error("Intrinsic '" + Int->Name + "' expects " +
               utostr(NumParamVTs) + " operands, not " +
               utostr(getNumChildren() - 1) + " operands!");
      return false;
    }

    // Apply type info to the intrinsic ID.
    MadeChange |= getChild(0)->UpdateNodeType(0, MVT::iPTR, TP);

    for (unsigned i = 0, e = getNumChildren()-1; i != e; ++i) {
      MadeChange |= getChild(i+1)->ApplyTypeConstraints(TP, NotRegisters);

      MVT::SimpleValueType OpVT = Int->IS.ParamVTs[i];
      assert(getChild(i+1)->getNumTypes() == 1 && "Unhandled case");
      MadeChange |= getChild(i+1)->UpdateNodeType(0, OpVT, TP);
    }
    return MadeChange;
  }

  if (getOperator()->isSubClassOf("SDNode")) {
    const SDNodeInfo &NI = CDP.getSDNodeInfo(getOperator());

    // Check that the number of operands is sane.  Negative operands -> varargs.
    if (NI.getNumOperands() >= 0 &&
        getNumChildren() != (unsigned)NI.getNumOperands()) {
      TP.error(getOperator()->getName() + " node requires exactly " +
               itostr(NI.getNumOperands()) + " operands!");
      return false;
    }

    bool MadeChange = NI.ApplyTypeConstraints(this, TP);
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
      MadeChange |= getChild(i)->ApplyTypeConstraints(TP, NotRegisters);
    return MadeChange;
  }

  if (getOperator()->isSubClassOf("Instruction")) {
    const DAGInstruction &Inst = CDP.getInstruction(getOperator());
    CodeGenInstruction &InstInfo =
      CDP.getTargetInfo().getInstruction(getOperator());

    bool MadeChange = false;

    // Apply the result types to the node, these come from the things in the
    // (outs) list of the instruction.
    unsigned NumResultsToAdd = std::min(InstInfo.Operands.NumDefs,
                                        Inst.getNumResults());
    for (unsigned ResNo = 0; ResNo != NumResultsToAdd; ++ResNo)
      MadeChange |= UpdateNodeTypeFromInst(ResNo, Inst.getResult(ResNo), TP);

    // If the instruction has implicit defs, we apply the first one as a result.
    // FIXME: This sucks, it should apply all implicit defs.
    if (!InstInfo.ImplicitDefs.empty()) {
      unsigned ResNo = NumResultsToAdd;

      // FIXME: Generalize to multiple possible types and multiple possible
      // ImplicitDefs.
      MVT::SimpleValueType VT =
        InstInfo.HasOneImplicitDefWithKnownVT(CDP.getTargetInfo());

      if (VT != MVT::Other)
        MadeChange |= UpdateNodeType(ResNo, VT, TP);
    }

    // If this is an INSERT_SUBREG, constrain the source and destination VTs to
    // be the same.
    if (getOperator()->getName() == "INSERT_SUBREG") {
      assert(getChild(0)->getNumTypes() == 1 && "FIXME: Unhandled");
      MadeChange |= UpdateNodeType(0, getChild(0)->getExtType(0), TP);
      MadeChange |= getChild(0)->UpdateNodeType(0, getExtType(0), TP);
    } else if (getOperator()->getName() == "REG_SEQUENCE") {
      // We need to do extra, custom typechecking for REG_SEQUENCE since it is
      // variadic.

      unsigned NChild = getNumChildren();
      if (NChild < 3) {
        TP.error("REG_SEQUENCE requires at least 3 operands!");
        return false;
      }

      if (NChild % 2 == 0) {
        TP.error("REG_SEQUENCE requires an odd number of operands!");
        return false;
      }

      if (!isOperandClass(getChild(0), "RegisterClass")) {
        TP.error("REG_SEQUENCE requires a RegisterClass for first operand!");
        return false;
      }

      for (unsigned I = 1; I < NChild; I += 2) {
        TreePatternNode *SubIdxChild = getChild(I + 1);
        if (!isOperandClass(SubIdxChild, "SubRegIndex")) {
          TP.error("REG_SEQUENCE requires a SubRegIndex for operand " +
                   itostr(I + 1) + "!");
          return false;
        }
      }
    }

    unsigned ChildNo = 0;
    for (unsigned i = 0, e = Inst.getNumOperands(); i != e; ++i) {
      Record *OperandNode = Inst.getOperand(i);

      // If the instruction expects a predicate or optional def operand, we
      // codegen this by setting the operand to it's default value if it has a
      // non-empty DefaultOps field.
      if (OperandNode->isSubClassOf("OperandWithDefaultOps") &&
          !CDP.getDefaultOperand(OperandNode).DefaultOps.empty())
        continue;

      // Verify that we didn't run out of provided operands.
      if (ChildNo >= getNumChildren()) {
        emitTooFewOperandsError(TP, getOperator()->getName(), getNumChildren());
        return false;
      }

      TreePatternNode *Child = getChild(ChildNo++);
      unsigned ChildResNo = 0;  // Instructions always use res #0 of their op.

      // If the operand has sub-operands, they may be provided by distinct
      // child patterns, so attempt to match each sub-operand separately.
      if (OperandNode->isSubClassOf("Operand")) {
        DagInit *MIOpInfo = OperandNode->getValueAsDag("MIOperandInfo");
        if (unsigned NumArgs = MIOpInfo->getNumArgs()) {
          // But don't do that if the whole operand is being provided by
          // a single ComplexPattern-related Operand.

          if (Child->getNumMIResults(CDP) < NumArgs) {
            // Match first sub-operand against the child we already have.
            Record *SubRec = cast<DefInit>(MIOpInfo->getArg(0))->getDef();
            MadeChange |=
              Child->UpdateNodeTypeFromInst(ChildResNo, SubRec, TP);

            // And the remaining sub-operands against subsequent children.
            for (unsigned Arg = 1; Arg < NumArgs; ++Arg) {
              if (ChildNo >= getNumChildren()) {
                emitTooFewOperandsError(TP, getOperator()->getName(),
                                        getNumChildren());
                return false;
              }
              Child = getChild(ChildNo++);

              SubRec = cast<DefInit>(MIOpInfo->getArg(Arg))->getDef();
              MadeChange |=
                Child->UpdateNodeTypeFromInst(ChildResNo, SubRec, TP);
            }
            continue;
          }
        }
      }

      // If we didn't match by pieces above, attempt to match the whole
      // operand now.
      MadeChange |= Child->UpdateNodeTypeFromInst(ChildResNo, OperandNode, TP);
    }

    if (!InstInfo.Operands.isVariadic && ChildNo != getNumChildren()) {
      emitTooManyOperandsError(TP, getOperator()->getName(),
                               ChildNo, getNumChildren());
      return false;
    }

    for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
      MadeChange |= getChild(i)->ApplyTypeConstraints(TP, NotRegisters);
    return MadeChange;
  }

  if (getOperator()->isSubClassOf("ComplexPattern")) {
    bool MadeChange = false;

    for (unsigned i = 0; i < getNumChildren(); ++i)
      MadeChange |= getChild(i)->ApplyTypeConstraints(TP, NotRegisters);

    return MadeChange;
  }

  assert(getOperator()->isSubClassOf("SDNodeXForm") && "Unknown node type!");

  // Node transforms always take one operand.
  if (getNumChildren() != 1) {
    TP.error("Node transform '" + getOperator()->getName() +
             "' requires one operand!");
    return false;
  }

  bool MadeChange = getChild(0)->ApplyTypeConstraints(TP, NotRegisters);


  // If either the output or input of the xform does not have exact
  // type info. We assume they must be the same. Otherwise, it is perfectly
  // legal to transform from one type to a completely different type.
#if 0
  if (!hasTypeSet() || !getChild(0)->hasTypeSet()) {
    bool MadeChange = UpdateNodeType(getChild(0)->getExtType(), TP);
    MadeChange |= getChild(0)->UpdateNodeType(getExtType(), TP);
    return MadeChange;
  }
#endif
  return MadeChange;
}

/// OnlyOnRHSOfCommutative - Return true if this value is only allowed on the
/// RHS of a commutative operation, not the on LHS.
static bool OnlyOnRHSOfCommutative(TreePatternNode *N) {
  if (!N->isLeaf() && N->getOperator()->getName() == "imm")
    return true;
  if (N->isLeaf() && isa<IntInit>(N->getLeafValue()))
    return true;
  return false;
}


/// canPatternMatch - If it is impossible for this pattern to match on this
/// target, fill in Reason and return false.  Otherwise, return true.  This is
/// used as a sanity check for .td files (to prevent people from writing stuff
/// that can never possibly work), and to prevent the pattern permuter from
/// generating stuff that is useless.
bool TreePatternNode::canPatternMatch(std::string &Reason,
                                      const CodeGenDAGPatterns &CDP) {
  if (isLeaf()) return true;

  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    if (!getChild(i)->canPatternMatch(Reason, CDP))
      return false;

  // If this is an intrinsic, handle cases that would make it not match.  For
  // example, if an operand is required to be an immediate.
  if (getOperator()->isSubClassOf("Intrinsic")) {
    // TODO:
    return true;
  }

  if (getOperator()->isSubClassOf("ComplexPattern"))
    return true;

  // If this node is a commutative operator, check that the LHS isn't an
  // immediate.
  const SDNodeInfo &NodeInfo = CDP.getSDNodeInfo(getOperator());
  bool isCommIntrinsic = isCommutativeIntrinsic(CDP);
  if (NodeInfo.hasProperty(SDNPCommutative) || isCommIntrinsic) {
    // Scan all of the operands of the node and make sure that only the last one
    // is a constant node, unless the RHS also is.
    if (!OnlyOnRHSOfCommutative(getChild(getNumChildren()-1))) {
      bool Skip = isCommIntrinsic ? 1 : 0; // First operand is intrinsic id.
      for (unsigned i = Skip, e = getNumChildren()-1; i != e; ++i)
        if (OnlyOnRHSOfCommutative(getChild(i))) {
          Reason="Immediate value must be on the RHS of commutative operators!";
          return false;
        }
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// TreePattern implementation
//

TreePattern::TreePattern(Record *TheRec, ListInit *RawPat, bool isInput,
                         CodeGenDAGPatterns &cdp) : TheRecord(TheRec), CDP(cdp),
                         isInputPattern(isInput), HasError(false) {
  for (Init *I : RawPat->getValues())
    Trees.push_back(ParseTreePattern(I, ""));
}

TreePattern::TreePattern(Record *TheRec, DagInit *Pat, bool isInput,
                         CodeGenDAGPatterns &cdp) : TheRecord(TheRec), CDP(cdp),
                         isInputPattern(isInput), HasError(false) {
  Trees.push_back(ParseTreePattern(Pat, ""));
}

TreePattern::TreePattern(Record *TheRec, TreePatternNode *Pat, bool isInput,
                         CodeGenDAGPatterns &cdp) : TheRecord(TheRec), CDP(cdp),
                         isInputPattern(isInput), HasError(false) {
  Trees.push_back(Pat);
}

void TreePattern::error(const Twine &Msg) {
  if (HasError)
    return;
  dump();
  PrintError(TheRecord->getLoc(), "In " + TheRecord->getName() + ": " + Msg);
  HasError = true;
}

void TreePattern::ComputeNamedNodes() {
  for (TreePatternNode *Tree : Trees)
    ComputeNamedNodes(Tree);
}

void TreePattern::ComputeNamedNodes(TreePatternNode *N) {
  if (!N->getName().empty())
    NamedNodes[N->getName()].push_back(N);

  for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i)
    ComputeNamedNodes(N->getChild(i));
}


TreePatternNode *TreePattern::ParseTreePattern(Init *TheInit, StringRef OpName){
  if (DefInit *DI = dyn_cast<DefInit>(TheInit)) {
    Record *R = DI->getDef();

    // Direct reference to a leaf DagNode or PatFrag?  Turn it into a
    // TreePatternNode of its own.  For example:
    ///   (foo GPR, imm) -> (foo GPR, (imm))
    if (R->isSubClassOf("SDNode") || R->isSubClassOf("PatFrag"))
      return ParseTreePattern(
        DagInit::get(DI, "",
                     std::vector<std::pair<Init*, std::string> >()),
        OpName);

    // Input argument?
    TreePatternNode *Res = new TreePatternNode(DI, 1);
    if (R->getName() == "node" && !OpName.empty()) {
      if (OpName.empty())
        error("'node' argument requires a name to match with operand list");
      Args.push_back(OpName);
    }

    Res->setName(OpName);
    return Res;
  }

  // ?:$name or just $name.
  if (isa<UnsetInit>(TheInit)) {
    if (OpName.empty())
      error("'?' argument requires a name to match with operand list");
    TreePatternNode *Res = new TreePatternNode(TheInit, 1);
    Args.push_back(OpName);
    Res->setName(OpName);
    return Res;
  }

  if (IntInit *II = dyn_cast<IntInit>(TheInit)) {
    if (!OpName.empty())
      error("Constant int argument should not have a name!");
    return new TreePatternNode(II, 1);
  }

  if (BitsInit *BI = dyn_cast<BitsInit>(TheInit)) {
    // Turn this into an IntInit.
    Init *II = BI->convertInitializerTo(IntRecTy::get());
    if (!II || !isa<IntInit>(II))
      error("Bits value must be constants!");
    return ParseTreePattern(II, OpName);
  }

  DagInit *Dag = dyn_cast<DagInit>(TheInit);
  if (!Dag) {
    TheInit->dump();
    error("Pattern has unexpected init kind!");
  }
  DefInit *OpDef = dyn_cast<DefInit>(Dag->getOperator());
  if (!OpDef) error("Pattern has unexpected operator type!");
  Record *Operator = OpDef->getDef();

  if (Operator->isSubClassOf("ValueType")) {
    // If the operator is a ValueType, then this must be "type cast" of a leaf
    // node.
    if (Dag->getNumArgs() != 1)
      error("Type cast only takes one operand!");

    TreePatternNode *New = ParseTreePattern(Dag->getArg(0), Dag->getArgName(0));

    // Apply the type cast.
    assert(New->getNumTypes() == 1 && "FIXME: Unhandled");
    New->UpdateNodeType(0, getValueType(Operator), *this);

    if (!OpName.empty())
      error("ValueType cast should not have a name!");
    return New;
  }

  // Verify that this is something that makes sense for an operator.
  if (!Operator->isSubClassOf("PatFrag") &&
      !Operator->isSubClassOf("SDNode") &&
      !Operator->isSubClassOf("Instruction") &&
      !Operator->isSubClassOf("SDNodeXForm") &&
      !Operator->isSubClassOf("Intrinsic") &&
      !Operator->isSubClassOf("ComplexPattern") &&
      Operator->getName() != "set" &&
      Operator->getName() != "implicit")
    error("Unrecognized node '" + Operator->getName() + "'!");

  //  Check to see if this is something that is illegal in an input pattern.
  if (isInputPattern) {
    if (Operator->isSubClassOf("Instruction") ||
        Operator->isSubClassOf("SDNodeXForm"))
      error("Cannot use '" + Operator->getName() + "' in an input pattern!");
  } else {
    if (Operator->isSubClassOf("Intrinsic"))
      error("Cannot use '" + Operator->getName() + "' in an output pattern!");

    if (Operator->isSubClassOf("SDNode") &&
        Operator->getName() != "imm" &&
        Operator->getName() != "fpimm" &&
        Operator->getName() != "tglobaltlsaddr" &&
        Operator->getName() != "tconstpool" &&
        Operator->getName() != "tjumptable" &&
        Operator->getName() != "tframeindex" &&
        Operator->getName() != "texternalsym" &&
        Operator->getName() != "tblockaddress" &&
        Operator->getName() != "tglobaladdr" &&
        Operator->getName() != "bb" &&
        Operator->getName() != "vt" &&
        Operator->getName() != "mcsym")
      error("Cannot use '" + Operator->getName() + "' in an output pattern!");
  }

  std::vector<TreePatternNode*> Children;

  // Parse all the operands.
  for (unsigned i = 0, e = Dag->getNumArgs(); i != e; ++i)
    Children.push_back(ParseTreePattern(Dag->getArg(i), Dag->getArgName(i)));

  // If the operator is an intrinsic, then this is just syntactic sugar for for
  // (intrinsic_* <number>, ..children..).  Pick the right intrinsic node, and
  // convert the intrinsic name to a number.
  if (Operator->isSubClassOf("Intrinsic")) {
    const CodeGenIntrinsic &Int = getDAGPatterns().getIntrinsic(Operator);
    unsigned IID = getDAGPatterns().getIntrinsicID(Operator)+1;

    // If this intrinsic returns void, it must have side-effects and thus a
    // chain.
    if (Int.IS.RetVTs.empty())
      Operator = getDAGPatterns().get_intrinsic_void_sdnode();
    else if (Int.ModRef != CodeGenIntrinsic::NoMem)
      // Has side-effects, requires chain.
      Operator = getDAGPatterns().get_intrinsic_w_chain_sdnode();
    else // Otherwise, no chain.
      Operator = getDAGPatterns().get_intrinsic_wo_chain_sdnode();

    TreePatternNode *IIDNode = new TreePatternNode(IntInit::get(IID), 1);
    Children.insert(Children.begin(), IIDNode);
  }

  if (Operator->isSubClassOf("ComplexPattern")) {
    for (unsigned i = 0; i < Children.size(); ++i) {
      TreePatternNode *Child = Children[i];

      if (Child->getName().empty())
        error("All arguments to a ComplexPattern must be named");

      // Check that the ComplexPattern uses are consistent: "(MY_PAT $a, $b)"
      // and "(MY_PAT $b, $a)" should not be allowed in the same pattern;
      // neither should "(MY_PAT_1 $a, $b)" and "(MY_PAT_2 $a, $b)".
      auto OperandId = std::make_pair(Operator, i);
      auto PrevOp = ComplexPatternOperands.find(Child->getName());
      if (PrevOp != ComplexPatternOperands.end()) {
        if (PrevOp->getValue() != OperandId)
          error("All ComplexPattern operands must appear consistently: "
                "in the same order in just one ComplexPattern instance.");
      } else
        ComplexPatternOperands[Child->getName()] = OperandId;
    }
  }

  unsigned NumResults = GetNumNodeResults(Operator, CDP);
  TreePatternNode *Result = new TreePatternNode(Operator, Children, NumResults);
  Result->setName(OpName);

  if (!Dag->getName().empty()) {
    assert(Result->getName().empty());
    Result->setName(Dag->getName());
  }
  return Result;
}

/// SimplifyTree - See if we can simplify this tree to eliminate something that
/// will never match in favor of something obvious that will.  This is here
/// strictly as a convenience to target authors because it allows them to write
/// more type generic things and have useless type casts fold away.
///
/// This returns true if any change is made.
static bool SimplifyTree(TreePatternNode *&N) {
  if (N->isLeaf())
    return false;

  // If we have a bitconvert with a resolved type and if the source and
  // destination types are the same, then the bitconvert is useless, remove it.
  if (N->getOperator()->getName() == "bitconvert" &&
      N->getExtType(0).isConcrete() &&
      N->getExtType(0) == N->getChild(0)->getExtType(0) &&
      N->getName().empty()) {
    N = N->getChild(0);
    SimplifyTree(N);
    return true;
  }

  // Walk all children.
  bool MadeChange = false;
  for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i) {
    TreePatternNode *Child = N->getChild(i);
    MadeChange |= SimplifyTree(Child);
    N->setChild(i, Child);
  }
  return MadeChange;
}



/// InferAllTypes - Infer/propagate as many types throughout the expression
/// patterns as possible.  Return true if all types are inferred, false
/// otherwise.  Flags an error if a type contradiction is found.
bool TreePattern::
InferAllTypes(const StringMap<SmallVector<TreePatternNode*,1> > *InNamedTypes) {
  if (NamedNodes.empty())
    ComputeNamedNodes();

  bool MadeChange = true;
  while (MadeChange) {
    MadeChange = false;
    for (TreePatternNode *Tree : Trees) {
      MadeChange |= Tree->ApplyTypeConstraints(*this, false);
      MadeChange |= SimplifyTree(Tree);
    }

    // If there are constraints on our named nodes, apply them.
    for (auto &Entry : NamedNodes) {
      SmallVectorImpl<TreePatternNode*> &Nodes = Entry.second;

      // If we have input named node types, propagate their types to the named
      // values here.
      if (InNamedTypes) {
        if (!InNamedTypes->count(Entry.getKey())) {
          error("Node '" + std::string(Entry.getKey()) +
                "' in output pattern but not input pattern");
          return true;
        }

        const SmallVectorImpl<TreePatternNode*> &InNodes =
          InNamedTypes->find(Entry.getKey())->second;

        // The input types should be fully resolved by now.
        for (TreePatternNode *Node : Nodes) {
          // If this node is a register class, and it is the root of the pattern
          // then we're mapping something onto an input register.  We allow
          // changing the type of the input register in this case.  This allows
          // us to match things like:
          //  def : Pat<(v1i64 (bitconvert(v2i32 DPR:$src))), (v1i64 DPR:$src)>;
          if (Node == Trees[0] && Node->isLeaf()) {
            DefInit *DI = dyn_cast<DefInit>(Node->getLeafValue());
            if (DI && (DI->getDef()->isSubClassOf("RegisterClass") ||
                       DI->getDef()->isSubClassOf("RegisterOperand")))
              continue;
          }

          assert(Node->getNumTypes() == 1 &&
                 InNodes[0]->getNumTypes() == 1 &&
                 "FIXME: cannot name multiple result nodes yet");
          MadeChange |= Node->UpdateNodeType(0, InNodes[0]->getExtType(0),
                                             *this);
        }
      }

      // If there are multiple nodes with the same name, they must all have the
      // same type.
      if (Entry.second.size() > 1) {
        for (unsigned i = 0, e = Nodes.size()-1; i != e; ++i) {
          TreePatternNode *N1 = Nodes[i], *N2 = Nodes[i+1];
          assert(N1->getNumTypes() == 1 && N2->getNumTypes() == 1 &&
                 "FIXME: cannot name multiple result nodes yet");

          MadeChange |= N1->UpdateNodeType(0, N2->getExtType(0), *this);
          MadeChange |= N2->UpdateNodeType(0, N1->getExtType(0), *this);
        }
      }
    }
  }

  bool HasUnresolvedTypes = false;
  for (const TreePatternNode *Tree : Trees)
    HasUnresolvedTypes |= Tree->ContainsUnresolvedType();
  return !HasUnresolvedTypes;
}

void TreePattern::print(raw_ostream &OS) const {
  OS << getRecord()->getName();
  if (!Args.empty()) {
    OS << "(" << Args[0];
    for (unsigned i = 1, e = Args.size(); i != e; ++i)
      OS << ", " << Args[i];
    OS << ")";
  }
  OS << ": ";

  if (Trees.size() > 1)
    OS << "[\n";
  for (const TreePatternNode *Tree : Trees) {
    OS << "\t";
    Tree->print(OS);
    OS << "\n";
  }

  if (Trees.size() > 1)
    OS << "]\n";
}

void TreePattern::dump() const { print(errs()); }

//===----------------------------------------------------------------------===//
// CodeGenDAGPatterns implementation
//

CodeGenDAGPatterns::CodeGenDAGPatterns(RecordKeeper &R) :
  Records(R), Target(R) {

  Intrinsics = LoadIntrinsics(Records, false);
  TgtIntrinsics = LoadIntrinsics(Records, true);
  ParseNodeInfo();
  ParseNodeTransforms();
  ParseComplexPatterns();
  ParsePatternFragments();
  ParseDefaultOperands();
  ParseInstructions();
  ParsePatternFragments(/*OutFrags*/true);
  ParsePatterns();

  // Generate variants.  For example, commutative patterns can match
  // multiple ways.  Add them to PatternsToMatch as well.
  GenerateVariants();

  // Infer instruction flags.  For example, we can detect loads,
  // stores, and side effects in many cases by examining an
  // instruction's pattern.
  InferInstructionFlags();

  // Verify that instruction flags match the patterns.
  VerifyInstructionFlags();
}

Record *CodeGenDAGPatterns::getSDNodeNamed(const std::string &Name) const {
  Record *N = Records.getDef(Name);
  if (!N || !N->isSubClassOf("SDNode"))
    PrintFatalError("Error getting SDNode '" + Name + "'!");

  return N;
}

// Parse all of the SDNode definitions for the target, populating SDNodes.
void CodeGenDAGPatterns::ParseNodeInfo() {
  std::vector<Record*> Nodes = Records.getAllDerivedDefinitions("SDNode");
  while (!Nodes.empty()) {
    SDNodes.insert(std::make_pair(Nodes.back(), Nodes.back()));
    Nodes.pop_back();
  }

  // Get the builtin intrinsic nodes.
  intrinsic_void_sdnode     = getSDNodeNamed("intrinsic_void");
  intrinsic_w_chain_sdnode  = getSDNodeNamed("intrinsic_w_chain");
  intrinsic_wo_chain_sdnode = getSDNodeNamed("intrinsic_wo_chain");
}

/// ParseNodeTransforms - Parse all SDNodeXForm instances into the SDNodeXForms
/// map, and emit them to the file as functions.
void CodeGenDAGPatterns::ParseNodeTransforms() {
  std::vector<Record*> Xforms = Records.getAllDerivedDefinitions("SDNodeXForm");
  while (!Xforms.empty()) {
    Record *XFormNode = Xforms.back();
    Record *SDNode = XFormNode->getValueAsDef("Opcode");
    std::string Code = XFormNode->getValueAsString("XFormFunction");
    SDNodeXForms.insert(std::make_pair(XFormNode, NodeXForm(SDNode, Code)));

    Xforms.pop_back();
  }
}

void CodeGenDAGPatterns::ParseComplexPatterns() {
  std::vector<Record*> AMs = Records.getAllDerivedDefinitions("ComplexPattern");
  while (!AMs.empty()) {
    ComplexPatterns.insert(std::make_pair(AMs.back(), AMs.back()));
    AMs.pop_back();
  }
}


/// ParsePatternFragments - Parse all of the PatFrag definitions in the .td
/// file, building up the PatternFragments map.  After we've collected them all,
/// inline fragments together as necessary, so that there are no references left
/// inside a pattern fragment to a pattern fragment.
///
void CodeGenDAGPatterns::ParsePatternFragments(bool OutFrags) {
  std::vector<Record*> Fragments = Records.getAllDerivedDefinitions("PatFrag");

  // First step, parse all of the fragments.
  for (Record *Frag : Fragments) {
    if (OutFrags != Frag->isSubClassOf("OutPatFrag"))
      continue;

    DagInit *Tree = Frag->getValueAsDag("Fragment");
    TreePattern *P =
        (PatternFragments[Frag] = llvm::make_unique<TreePattern>(
             Frag, Tree, !Frag->isSubClassOf("OutPatFrag"),
             *this)).get();

    // Validate the argument list, converting it to set, to discard duplicates.
    std::vector<std::string> &Args = P->getArgList();
    std::set<std::string> OperandsSet(Args.begin(), Args.end());

    if (OperandsSet.count(""))
      P->error("Cannot have unnamed 'node' values in pattern fragment!");

    // Parse the operands list.
    DagInit *OpsList = Frag->getValueAsDag("Operands");
    DefInit *OpsOp = dyn_cast<DefInit>(OpsList->getOperator());
    // Special cases: ops == outs == ins. Different names are used to
    // improve readability.
    if (!OpsOp ||
        (OpsOp->getDef()->getName() != "ops" &&
         OpsOp->getDef()->getName() != "outs" &&
         OpsOp->getDef()->getName() != "ins"))
      P->error("Operands list should start with '(ops ... '!");

    // Copy over the arguments.
    Args.clear();
    for (unsigned j = 0, e = OpsList->getNumArgs(); j != e; ++j) {
      if (!isa<DefInit>(OpsList->getArg(j)) ||
          cast<DefInit>(OpsList->getArg(j))->getDef()->getName() != "node")
        P->error("Operands list should all be 'node' values.");
      if (OpsList->getArgName(j).empty())
        P->error("Operands list should have names for each operand!");
      if (!OperandsSet.count(OpsList->getArgName(j)))
        P->error("'" + OpsList->getArgName(j) +
                 "' does not occur in pattern or was multiply specified!");
      OperandsSet.erase(OpsList->getArgName(j));
      Args.push_back(OpsList->getArgName(j));
    }

    if (!OperandsSet.empty())
      P->error("Operands list does not contain an entry for operand '" +
               *OperandsSet.begin() + "'!");

    // If there is a code init for this fragment, keep track of the fact that
    // this fragment uses it.
    TreePredicateFn PredFn(P);
    if (!PredFn.isAlwaysTrue())
      P->getOnlyTree()->addPredicateFn(PredFn);

    // If there is a node transformation corresponding to this, keep track of
    // it.
    Record *Transform = Frag->getValueAsDef("OperandTransform");
    if (!getSDNodeTransform(Transform).second.empty())    // not noop xform?
      P->getOnlyTree()->setTransformFn(Transform);
  }

  // Now that we've parsed all of the tree fragments, do a closure on them so
  // that there are not references to PatFrags left inside of them.
  for (Record *Frag : Fragments) {
    if (OutFrags != Frag->isSubClassOf("OutPatFrag"))
      continue;

    TreePattern &ThePat = *PatternFragments[Frag];
    ThePat.InlinePatternFragments();

    // Infer as many types as possible.  Don't worry about it if we don't infer
    // all of them, some may depend on the inputs of the pattern.
    ThePat.InferAllTypes();
    ThePat.resetError();

    // If debugging, print out the pattern fragment result.
    DEBUG(ThePat.dump());
  }
}

void CodeGenDAGPatterns::ParseDefaultOperands() {
  std::vector<Record*> DefaultOps;
  DefaultOps = Records.getAllDerivedDefinitions("OperandWithDefaultOps");

  // Find some SDNode.
  assert(!SDNodes.empty() && "No SDNodes parsed?");
  Init *SomeSDNode = DefInit::get(SDNodes.begin()->first);

  for (unsigned i = 0, e = DefaultOps.size(); i != e; ++i) {
    DagInit *DefaultInfo = DefaultOps[i]->getValueAsDag("DefaultOps");

    // Clone the DefaultInfo dag node, changing the operator from 'ops' to
    // SomeSDnode so that we can parse this.
    std::vector<std::pair<Init*, std::string> > Ops;
    for (unsigned op = 0, e = DefaultInfo->getNumArgs(); op != e; ++op)
      Ops.push_back(std::make_pair(DefaultInfo->getArg(op),
                                   DefaultInfo->getArgName(op)));
    DagInit *DI = DagInit::get(SomeSDNode, "", Ops);

    // Create a TreePattern to parse this.
    TreePattern P(DefaultOps[i], DI, false, *this);
    assert(P.getNumTrees() == 1 && "This ctor can only produce one tree!");

    // Copy the operands over into a DAGDefaultOperand.
    DAGDefaultOperand DefaultOpInfo;

    TreePatternNode *T = P.getTree(0);
    for (unsigned op = 0, e = T->getNumChildren(); op != e; ++op) {
      TreePatternNode *TPN = T->getChild(op);
      while (TPN->ApplyTypeConstraints(P, false))
        /* Resolve all types */;

      if (TPN->ContainsUnresolvedType()) {
        PrintFatalError("Value #" + Twine(i) + " of OperandWithDefaultOps '" +
                        DefaultOps[i]->getName() +
                        "' doesn't have a concrete type!");
      }
      DefaultOpInfo.DefaultOps.push_back(TPN);
    }

    // Insert it into the DefaultOperands map so we can find it later.
    DefaultOperands[DefaultOps[i]] = DefaultOpInfo;
  }
}

/// HandleUse - Given "Pat" a leaf in the pattern, check to see if it is an
/// instruction input.  Return true if this is a real use.
static bool HandleUse(TreePattern *I, TreePatternNode *Pat,
                      std::map<std::string, TreePatternNode*> &InstInputs) {
  // No name -> not interesting.
  if (Pat->getName().empty()) {
    if (Pat->isLeaf()) {
      DefInit *DI = dyn_cast<DefInit>(Pat->getLeafValue());
      if (DI && (DI->getDef()->isSubClassOf("RegisterClass") ||
                 DI->getDef()->isSubClassOf("RegisterOperand")))
        I->error("Input " + DI->getDef()->getName() + " must be named!");
    }
    return false;
  }

  Record *Rec;
  if (Pat->isLeaf()) {
    DefInit *DI = dyn_cast<DefInit>(Pat->getLeafValue());
    if (!DI) I->error("Input $" + Pat->getName() + " must be an identifier!");
    Rec = DI->getDef();
  } else {
    Rec = Pat->getOperator();
  }

  // SRCVALUE nodes are ignored.
  if (Rec->getName() == "srcvalue")
    return false;

  TreePatternNode *&Slot = InstInputs[Pat->getName()];
  if (!Slot) {
    Slot = Pat;
    return true;
  }
  Record *SlotRec;
  if (Slot->isLeaf()) {
    SlotRec = cast<DefInit>(Slot->getLeafValue())->getDef();
  } else {
    assert(Slot->getNumChildren() == 0 && "can't be a use with children!");
    SlotRec = Slot->getOperator();
  }

  // Ensure that the inputs agree if we've already seen this input.
  if (Rec != SlotRec)
    I->error("All $" + Pat->getName() + " inputs must agree with each other");
  if (Slot->getExtTypes() != Pat->getExtTypes())
    I->error("All $" + Pat->getName() + " inputs must agree with each other");
  return true;
}

/// FindPatternInputsAndOutputs - Scan the specified TreePatternNode (which is
/// part of "I", the instruction), computing the set of inputs and outputs of
/// the pattern.  Report errors if we see anything naughty.
void CodeGenDAGPatterns::
FindPatternInputsAndOutputs(TreePattern *I, TreePatternNode *Pat,
                            std::map<std::string, TreePatternNode*> &InstInputs,
                            std::map<std::string, TreePatternNode*>&InstResults,
                            std::vector<Record*> &InstImpResults) {
  if (Pat->isLeaf()) {
    bool isUse = HandleUse(I, Pat, InstInputs);
    if (!isUse && Pat->getTransformFn())
      I->error("Cannot specify a transform function for a non-input value!");
    return;
  }

  if (Pat->getOperator()->getName() == "implicit") {
    for (unsigned i = 0, e = Pat->getNumChildren(); i != e; ++i) {
      TreePatternNode *Dest = Pat->getChild(i);
      if (!Dest->isLeaf())
        I->error("implicitly defined value should be a register!");

      DefInit *Val = dyn_cast<DefInit>(Dest->getLeafValue());
      if (!Val || !Val->getDef()->isSubClassOf("Register"))
        I->error("implicitly defined value should be a register!");
      InstImpResults.push_back(Val->getDef());
    }
    return;
  }

  if (Pat->getOperator()->getName() != "set") {
    // If this is not a set, verify that the children nodes are not void typed,
    // and recurse.
    for (unsigned i = 0, e = Pat->getNumChildren(); i != e; ++i) {
      if (Pat->getChild(i)->getNumTypes() == 0)
        I->error("Cannot have void nodes inside of patterns!");
      FindPatternInputsAndOutputs(I, Pat->getChild(i), InstInputs, InstResults,
                                  InstImpResults);
    }

    // If this is a non-leaf node with no children, treat it basically as if
    // it were a leaf.  This handles nodes like (imm).
    bool isUse = HandleUse(I, Pat, InstInputs);

    if (!isUse && Pat->getTransformFn())
      I->error("Cannot specify a transform function for a non-input value!");
    return;
  }

  // Otherwise, this is a set, validate and collect instruction results.
  if (Pat->getNumChildren() == 0)
    I->error("set requires operands!");

  if (Pat->getTransformFn())
    I->error("Cannot specify a transform function on a set node!");

  // Check the set destinations.
  unsigned NumDests = Pat->getNumChildren()-1;
  for (unsigned i = 0; i != NumDests; ++i) {
    TreePatternNode *Dest = Pat->getChild(i);
    if (!Dest->isLeaf())
      I->error("set destination should be a register!");

    DefInit *Val = dyn_cast<DefInit>(Dest->getLeafValue());
    if (!Val) {
      I->error("set destination should be a register!");
      continue;
    }

    if (Val->getDef()->isSubClassOf("RegisterClass") ||
        Val->getDef()->isSubClassOf("ValueType") ||
        Val->getDef()->isSubClassOf("RegisterOperand") ||
        Val->getDef()->isSubClassOf("PointerLikeRegClass")) {
      if (Dest->getName().empty())
        I->error("set destination must have a name!");
      if (InstResults.count(Dest->getName()))
        I->error("cannot set '" + Dest->getName() +"' multiple times");
      InstResults[Dest->getName()] = Dest;
    } else if (Val->getDef()->isSubClassOf("Register")) {
      InstImpResults.push_back(Val->getDef());
    } else {
      I->error("set destination should be a register!");
    }
  }

  // Verify and collect info from the computation.
  FindPatternInputsAndOutputs(I, Pat->getChild(NumDests),
                              InstInputs, InstResults, InstImpResults);
}

//===----------------------------------------------------------------------===//
// Instruction Analysis
//===----------------------------------------------------------------------===//

class InstAnalyzer {
  const CodeGenDAGPatterns &CDP;
public:
  bool hasSideEffects;
  bool mayStore;
  bool mayLoad;
  bool isBitcast;
  bool isVariadic;

  InstAnalyzer(const CodeGenDAGPatterns &cdp)
    : CDP(cdp), hasSideEffects(false), mayStore(false), mayLoad(false),
      isBitcast(false), isVariadic(false) {}

  void Analyze(const TreePattern *Pat) {
    // Assume only the first tree is the pattern. The others are clobber nodes.
    AnalyzeNode(Pat->getTree(0));
  }

  void Analyze(const PatternToMatch *Pat) {
    AnalyzeNode(Pat->getSrcPattern());
  }

private:
  bool IsNodeBitcast(const TreePatternNode *N) const {
    if (hasSideEffects || mayLoad || mayStore || isVariadic)
      return false;

    if (N->getNumChildren() != 2)
      return false;

    const TreePatternNode *N0 = N->getChild(0);
    if (!N0->isLeaf() || !isa<DefInit>(N0->getLeafValue()))
      return false;

    const TreePatternNode *N1 = N->getChild(1);
    if (N1->isLeaf())
      return false;
    if (N1->getNumChildren() != 1 || !N1->getChild(0)->isLeaf())
      return false;

    const SDNodeInfo &OpInfo = CDP.getSDNodeInfo(N1->getOperator());
    if (OpInfo.getNumResults() != 1 || OpInfo.getNumOperands() != 1)
      return false;
    return OpInfo.getEnumName() == "ISD::BITCAST";
  }

public:
  void AnalyzeNode(const TreePatternNode *N) {
    if (N->isLeaf()) {
      if (DefInit *DI = dyn_cast<DefInit>(N->getLeafValue())) {
        Record *LeafRec = DI->getDef();
        // Handle ComplexPattern leaves.
        if (LeafRec->isSubClassOf("ComplexPattern")) {
          const ComplexPattern &CP = CDP.getComplexPattern(LeafRec);
          if (CP.hasProperty(SDNPMayStore)) mayStore = true;
          if (CP.hasProperty(SDNPMayLoad)) mayLoad = true;
          if (CP.hasProperty(SDNPSideEffect)) hasSideEffects = true;
        }
      }
      return;
    }

    // Analyze children.
    for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i)
      AnalyzeNode(N->getChild(i));

    // Ignore set nodes, which are not SDNodes.
    if (N->getOperator()->getName() == "set") {
      isBitcast = IsNodeBitcast(N);
      return;
    }

    // Notice properties of the node.
    if (N->NodeHasProperty(SDNPMayStore, CDP)) mayStore = true;
    if (N->NodeHasProperty(SDNPMayLoad, CDP)) mayLoad = true;
    if (N->NodeHasProperty(SDNPSideEffect, CDP)) hasSideEffects = true;
    if (N->NodeHasProperty(SDNPVariadic, CDP)) isVariadic = true;

    if (const CodeGenIntrinsic *IntInfo = N->getIntrinsicInfo(CDP)) {
      // If this is an intrinsic, analyze it.
      if (IntInfo->ModRef >= CodeGenIntrinsic::ReadArgMem)
        mayLoad = true;// These may load memory.

      if (IntInfo->ModRef >= CodeGenIntrinsic::ReadWriteArgMem)
        mayStore = true;// Intrinsics that can write to memory are 'mayStore'.

      if (IntInfo->ModRef >= CodeGenIntrinsic::ReadWriteMem)
        // WriteMem intrinsics can have other strange effects.
        hasSideEffects = true;
    }
  }

};

static bool InferFromPattern(CodeGenInstruction &InstInfo,
                             const InstAnalyzer &PatInfo,
                             Record *PatDef) {
  bool Error = false;

  // Remember where InstInfo got its flags.
  if (InstInfo.hasUndefFlags())
      InstInfo.InferredFrom = PatDef;

  // Check explicitly set flags for consistency.
  if (InstInfo.hasSideEffects != PatInfo.hasSideEffects &&
      !InstInfo.hasSideEffects_Unset) {
    // Allow explicitly setting hasSideEffects = 1 on instructions, even when
    // the pattern has no side effects. That could be useful for div/rem
    // instructions that may trap.
    if (!InstInfo.hasSideEffects) {
      Error = true;
      PrintError(PatDef->getLoc(), "Pattern doesn't match hasSideEffects = " +
                 Twine(InstInfo.hasSideEffects));
    }
  }

  if (InstInfo.mayStore != PatInfo.mayStore && !InstInfo.mayStore_Unset) {
    Error = true;
    PrintError(PatDef->getLoc(), "Pattern doesn't match mayStore = " +
               Twine(InstInfo.mayStore));
  }

  if (InstInfo.mayLoad != PatInfo.mayLoad && !InstInfo.mayLoad_Unset) {
    // Allow explicitly setting mayLoad = 1, even when the pattern has no loads.
    // Some targets translate immediates to loads.
    if (!InstInfo.mayLoad) {
      Error = true;
      PrintError(PatDef->getLoc(), "Pattern doesn't match mayLoad = " +
                 Twine(InstInfo.mayLoad));
    }
  }

  // Transfer inferred flags.
  InstInfo.hasSideEffects |= PatInfo.hasSideEffects;
  InstInfo.mayStore |= PatInfo.mayStore;
  InstInfo.mayLoad |= PatInfo.mayLoad;

  // These flags are silently added without any verification.
  InstInfo.isBitcast |= PatInfo.isBitcast;

  // Don't infer isVariadic. This flag means something different on SDNodes and
  // instructions. For example, a CALL SDNode is variadic because it has the
  // call arguments as operands, but a CALL instruction is not variadic - it
  // has argument registers as implicit, not explicit uses.

  return Error;
}

/// hasNullFragReference - Return true if the DAG has any reference to the
/// null_frag operator.
static bool hasNullFragReference(DagInit *DI) {
  DefInit *OpDef = dyn_cast<DefInit>(DI->getOperator());
  if (!OpDef) return false;
  Record *Operator = OpDef->getDef();

  // If this is the null fragment, return true.
  if (Operator->getName() == "null_frag") return true;
  // If any of the arguments reference the null fragment, return true.
  for (unsigned i = 0, e = DI->getNumArgs(); i != e; ++i) {
    DagInit *Arg = dyn_cast<DagInit>(DI->getArg(i));
    if (Arg && hasNullFragReference(Arg))
      return true;
  }

  return false;
}

/// hasNullFragReference - Return true if any DAG in the list references
/// the null_frag operator.
static bool hasNullFragReference(ListInit *LI) {
  for (Init *I : LI->getValues()) {
    DagInit *DI = dyn_cast<DagInit>(I);
    assert(DI && "non-dag in an instruction Pattern list?!");
    if (hasNullFragReference(DI))
      return true;
  }
  return false;
}

/// Get all the instructions in a tree.
static void
getInstructionsInTree(TreePatternNode *Tree, SmallVectorImpl<Record*> &Instrs) {
  if (Tree->isLeaf())
    return;
  if (Tree->getOperator()->isSubClassOf("Instruction"))
    Instrs.push_back(Tree->getOperator());
  for (unsigned i = 0, e = Tree->getNumChildren(); i != e; ++i)
    getInstructionsInTree(Tree->getChild(i), Instrs);
}

/// Check the class of a pattern leaf node against the instruction operand it
/// represents.
static bool checkOperandClass(CGIOperandList::OperandInfo &OI,
                              Record *Leaf) {
  if (OI.Rec == Leaf)
    return true;

  // Allow direct value types to be used in instruction set patterns.
  // The type will be checked later.
  if (Leaf->isSubClassOf("ValueType"))
    return true;

  // Patterns can also be ComplexPattern instances.
  if (Leaf->isSubClassOf("ComplexPattern"))
    return true;

  return false;
}

const DAGInstruction &CodeGenDAGPatterns::parseInstructionPattern(
    CodeGenInstruction &CGI, ListInit *Pat, DAGInstMap &DAGInsts) {

  assert(!DAGInsts.count(CGI.TheDef) && "Instruction already parsed!");

  // Parse the instruction.
  TreePattern *I = new TreePattern(CGI.TheDef, Pat, true, *this);
  // Inline pattern fragments into it.
  I->InlinePatternFragments();

  // Infer as many types as possible.  If we cannot infer all of them, we can
  // never do anything with this instruction pattern: report it to the user.
  if (!I->InferAllTypes())
    I->error("Could not infer all types in pattern!");

  // InstInputs - Keep track of all of the inputs of the instruction, along
  // with the record they are declared as.
  std::map<std::string, TreePatternNode*> InstInputs;

  // InstResults - Keep track of all the virtual registers that are 'set'
  // in the instruction, including what reg class they are.
  std::map<std::string, TreePatternNode*> InstResults;

  std::vector<Record*> InstImpResults;

  // Verify that the top-level forms in the instruction are of void type, and
  // fill in the InstResults map.
  for (unsigned j = 0, e = I->getNumTrees(); j != e; ++j) {
    TreePatternNode *Pat = I->getTree(j);
    if (Pat->getNumTypes() != 0)
      I->error("Top-level forms in instruction pattern should have"
               " void types");

    // Find inputs and outputs, and verify the structure of the uses/defs.
    FindPatternInputsAndOutputs(I, Pat, InstInputs, InstResults,
                                InstImpResults);
  }

  // Now that we have inputs and outputs of the pattern, inspect the operands
  // list for the instruction.  This determines the order that operands are
  // added to the machine instruction the node corresponds to.
  unsigned NumResults = InstResults.size();

  // Parse the operands list from the (ops) list, validating it.
  assert(I->getArgList().empty() && "Args list should still be empty here!");

  // Check that all of the results occur first in the list.
  std::vector<Record*> Results;
  SmallVector<TreePatternNode *, 2> ResNodes;
  for (unsigned i = 0; i != NumResults; ++i) {
    if (i == CGI.Operands.size())
      I->error("'" + InstResults.begin()->first +
               "' set but does not appear in operand list!");
    const std::string &OpName = CGI.Operands[i].Name;

    // Check that it exists in InstResults.
    TreePatternNode *RNode = InstResults[OpName];
    if (!RNode)
      I->error("Operand $" + OpName + " does not exist in operand list!");

    ResNodes.push_back(RNode);

    Record *R = cast<DefInit>(RNode->getLeafValue())->getDef();
    if (!R)
      I->error("Operand $" + OpName + " should be a set destination: all "
               "outputs must occur before inputs in operand list!");

    if (!checkOperandClass(CGI.Operands[i], R))
      I->error("Operand $" + OpName + " class mismatch!");

    // Remember the return type.
    Results.push_back(CGI.Operands[i].Rec);

    // Okay, this one checks out.
    InstResults.erase(OpName);
  }

  // Loop over the inputs next.  Make a copy of InstInputs so we can destroy
  // the copy while we're checking the inputs.
  std::map<std::string, TreePatternNode*> InstInputsCheck(InstInputs);

  std::vector<TreePatternNode*> ResultNodeOperands;
  std::vector<Record*> Operands;
  for (unsigned i = NumResults, e = CGI.Operands.size(); i != e; ++i) {
    CGIOperandList::OperandInfo &Op = CGI.Operands[i];
    const std::string &OpName = Op.Name;
    if (OpName.empty())
      I->error("Operand #" + utostr(i) + " in operands list has no name!");

    if (!InstInputsCheck.count(OpName)) {
      // If this is an operand with a DefaultOps set filled in, we can ignore
      // this.  When we codegen it, we will do so as always executed.
      if (Op.Rec->isSubClassOf("OperandWithDefaultOps")) {
        // Does it have a non-empty DefaultOps field?  If so, ignore this
        // operand.
        if (!getDefaultOperand(Op.Rec).DefaultOps.empty())
          continue;
      }
      I->error("Operand $" + OpName +
               " does not appear in the instruction pattern");
    }
    TreePatternNode *InVal = InstInputsCheck[OpName];
    InstInputsCheck.erase(OpName);   // It occurred, remove from map.

    if (InVal->isLeaf() && isa<DefInit>(InVal->getLeafValue())) {
      Record *InRec = static_cast<DefInit*>(InVal->getLeafValue())->getDef();
      if (!checkOperandClass(Op, InRec))
        I->error("Operand $" + OpName + "'s register class disagrees"
                 " between the operand and pattern");
    }
    Operands.push_back(Op.Rec);

    // Construct the result for the dest-pattern operand list.
    TreePatternNode *OpNode = InVal->clone();

    // No predicate is useful on the result.
    OpNode->clearPredicateFns();

    // Promote the xform function to be an explicit node if set.
    if (Record *Xform = OpNode->getTransformFn()) {
      OpNode->setTransformFn(nullptr);
      std::vector<TreePatternNode*> Children;
      Children.push_back(OpNode);
      OpNode = new TreePatternNode(Xform, Children, OpNode->getNumTypes());
    }

    ResultNodeOperands.push_back(OpNode);
  }

  if (!InstInputsCheck.empty())
    I->error("Input operand $" + InstInputsCheck.begin()->first +
             " occurs in pattern but not in operands list!");

  TreePatternNode *ResultPattern =
    new TreePatternNode(I->getRecord(), ResultNodeOperands,
                        GetNumNodeResults(I->getRecord(), *this));
  // Copy fully inferred output node types to instruction result pattern.
  for (unsigned i = 0; i != NumResults; ++i) {
    assert(ResNodes[i]->getNumTypes() == 1 && "FIXME: Unhandled");
    ResultPattern->setType(i, ResNodes[i]->getExtType(0));
  }

  // Create and insert the instruction.
  // FIXME: InstImpResults should not be part of DAGInstruction.
  DAGInstruction TheInst(I, Results, Operands, InstImpResults);
  DAGInsts.insert(std::make_pair(I->getRecord(), TheInst));

  // Use a temporary tree pattern to infer all types and make sure that the
  // constructed result is correct.  This depends on the instruction already
  // being inserted into the DAGInsts map.
  TreePattern Temp(I->getRecord(), ResultPattern, false, *this);
  Temp.InferAllTypes(&I->getNamedNodesMap());

  DAGInstruction &TheInsertedInst = DAGInsts.find(I->getRecord())->second;
  TheInsertedInst.setResultPattern(Temp.getOnlyTree());

  return TheInsertedInst;
}

/// ParseInstructions - Parse all of the instructions, inlining and resolving
/// any fragments involved.  This populates the Instructions list with fully
/// resolved instructions.
void CodeGenDAGPatterns::ParseInstructions() {
  std::vector<Record*> Instrs = Records.getAllDerivedDefinitions("Instruction");

  for (Record *Instr : Instrs) {
    ListInit *LI = nullptr;

    if (isa<ListInit>(Instr->getValueInit("Pattern")))
      LI = Instr->getValueAsListInit("Pattern");

    // If there is no pattern, only collect minimal information about the
    // instruction for its operand list.  We have to assume that there is one
    // result, as we have no detailed info. A pattern which references the
    // null_frag operator is as-if no pattern were specified. Normally this
    // is from a multiclass expansion w/ a SDPatternOperator passed in as
    // null_frag.
    if (!LI || LI->empty() || hasNullFragReference(LI)) {
      std::vector<Record*> Results;
      std::vector<Record*> Operands;

      CodeGenInstruction &InstInfo = Target.getInstruction(Instr);

      if (InstInfo.Operands.size() != 0) {
        for (unsigned j = 0, e = InstInfo.Operands.NumDefs; j < e; ++j)
          Results.push_back(InstInfo.Operands[j].Rec);

        // The rest are inputs.
        for (unsigned j = InstInfo.Operands.NumDefs,
               e = InstInfo.Operands.size(); j < e; ++j)
          Operands.push_back(InstInfo.Operands[j].Rec);
      }

      // Create and insert the instruction.
      std::vector<Record*> ImpResults;
      Instructions.insert(std::make_pair(Instr,
                          DAGInstruction(nullptr, Results, Operands, ImpResults)));
      continue;  // no pattern.
    }

    CodeGenInstruction &CGI = Target.getInstruction(Instr);
    const DAGInstruction &DI = parseInstructionPattern(CGI, LI, Instructions);

    (void)DI;
    DEBUG(DI.getPattern()->dump());
  }

  // If we can, convert the instructions to be patterns that are matched!
  for (auto &Entry : Instructions) {
    DAGInstruction &TheInst = Entry.second;
    TreePattern *I = TheInst.getPattern();
    if (!I) continue;  // No pattern.

    // FIXME: Assume only the first tree is the pattern. The others are clobber
    // nodes.
    TreePatternNode *Pattern = I->getTree(0);
    TreePatternNode *SrcPattern;
    if (Pattern->getOperator()->getName() == "set") {
      SrcPattern = Pattern->getChild(Pattern->getNumChildren()-1)->clone();
    } else{
      // Not a set (store or something?)
      SrcPattern = Pattern;
    }

    Record *Instr = Entry.first;
    AddPatternToMatch(I,
                      PatternToMatch(Instr,
                                     Instr->getValueAsListInit("Predicates"),
                                     SrcPattern,
                                     TheInst.getResultPattern(),
                                     TheInst.getImpResults(),
                                     Instr->getValueAsInt("AddedComplexity"),
                                     Instr->getID()));
  }
}


typedef std::pair<const TreePatternNode*, unsigned> NameRecord;

static void FindNames(const TreePatternNode *P,
                      std::map<std::string, NameRecord> &Names,
                      TreePattern *PatternTop) {
  if (!P->getName().empty()) {
    NameRecord &Rec = Names[P->getName()];
    // If this is the first instance of the name, remember the node.
    if (Rec.second++ == 0)
      Rec.first = P;
    else if (Rec.first->getExtTypes() != P->getExtTypes())
      PatternTop->error("repetition of value: $" + P->getName() +
                        " where different uses have different types!");
  }

  if (!P->isLeaf()) {
    for (unsigned i = 0, e = P->getNumChildren(); i != e; ++i)
      FindNames(P->getChild(i), Names, PatternTop);
  }
}

void CodeGenDAGPatterns::AddPatternToMatch(TreePattern *Pattern,
                                           const PatternToMatch &PTM) {
  // Do some sanity checking on the pattern we're about to match.
  std::string Reason;
  if (!PTM.getSrcPattern()->canPatternMatch(Reason, *this)) {
    PrintWarning(Pattern->getRecord()->getLoc(),
      Twine("Pattern can never match: ") + Reason);
    return;
  }

  // If the source pattern's root is a complex pattern, that complex pattern
  // must specify the nodes it can potentially match.
  if (const ComplexPattern *CP =
        PTM.getSrcPattern()->getComplexPatternInfo(*this))
    if (CP->getRootNodes().empty())
      Pattern->error("ComplexPattern at root must specify list of opcodes it"
                     " could match");


  // Find all of the named values in the input and output, ensure they have the
  // same type.
  std::map<std::string, NameRecord> SrcNames, DstNames;
  FindNames(PTM.getSrcPattern(), SrcNames, Pattern);
  FindNames(PTM.getDstPattern(), DstNames, Pattern);

  // Scan all of the named values in the destination pattern, rejecting them if
  // they don't exist in the input pattern.
  for (const auto &Entry : DstNames) {
    if (SrcNames[Entry.first].first == nullptr)
      Pattern->error("Pattern has input without matching name in output: $" +
                     Entry.first);
  }

  // Scan all of the named values in the source pattern, rejecting them if the
  // name isn't used in the dest, and isn't used to tie two values together.
  for (const auto &Entry : SrcNames)
    if (DstNames[Entry.first].first == nullptr &&
        SrcNames[Entry.first].second == 1)
      Pattern->error("Pattern has dead named input: $" + Entry.first);

  PatternsToMatch.push_back(PTM);
}



void CodeGenDAGPatterns::InferInstructionFlags() {
  const std::vector<const CodeGenInstruction*> &Instructions =
    Target.getInstructionsByEnumValue();

  // First try to infer flags from the primary instruction pattern, if any.
  SmallVector<CodeGenInstruction*, 8> Revisit;
  unsigned Errors = 0;
  for (unsigned i = 0, e = Instructions.size(); i != e; ++i) {
    CodeGenInstruction &InstInfo =
      const_cast<CodeGenInstruction &>(*Instructions[i]);

    // Get the primary instruction pattern.
    const TreePattern *Pattern = getInstruction(InstInfo.TheDef).getPattern();
    if (!Pattern) {
      if (InstInfo.hasUndefFlags())
        Revisit.push_back(&InstInfo);
      continue;
    }
    InstAnalyzer PatInfo(*this);
    PatInfo.Analyze(Pattern);
    Errors += InferFromPattern(InstInfo, PatInfo, InstInfo.TheDef);
  }

  // Second, look for single-instruction patterns defined outside the
  // instruction.
  for (ptm_iterator I = ptm_begin(), E = ptm_end(); I != E; ++I) {
    const PatternToMatch &PTM = *I;

    // We can only infer from single-instruction patterns, otherwise we won't
    // know which instruction should get the flags.
    SmallVector<Record*, 8> PatInstrs;
    getInstructionsInTree(PTM.getDstPattern(), PatInstrs);
    if (PatInstrs.size() != 1)
      continue;

    // Get the single instruction.
    CodeGenInstruction &InstInfo = Target.getInstruction(PatInstrs.front());

    // Only infer properties from the first pattern. We'll verify the others.
    if (InstInfo.InferredFrom)
      continue;

    InstAnalyzer PatInfo(*this);
    PatInfo.Analyze(&PTM);
    Errors += InferFromPattern(InstInfo, PatInfo, PTM.getSrcRecord());
  }

  if (Errors)
    PrintFatalError("pattern conflicts");

  // Revisit instructions with undefined flags and no pattern.
  if (Target.guessInstructionProperties()) {
    for (CodeGenInstruction *InstInfo : Revisit) {
      if (InstInfo->InferredFrom)
        continue;
      // The mayLoad and mayStore flags default to false.
      // Conservatively assume hasSideEffects if it wasn't explicit.
      if (InstInfo->hasSideEffects_Unset)
        InstInfo->hasSideEffects = true;
    }
    return;
  }

  // Complain about any flags that are still undefined.
  for (CodeGenInstruction *InstInfo : Revisit) {
    if (InstInfo->InferredFrom)
      continue;
    if (InstInfo->hasSideEffects_Unset)
      PrintError(InstInfo->TheDef->getLoc(),
                 "Can't infer hasSideEffects from patterns");
    if (InstInfo->mayStore_Unset)
      PrintError(InstInfo->TheDef->getLoc(),
                 "Can't infer mayStore from patterns");
    if (InstInfo->mayLoad_Unset)
      PrintError(InstInfo->TheDef->getLoc(),
                 "Can't infer mayLoad from patterns");
  }
}


/// Verify instruction flags against pattern node properties.
void CodeGenDAGPatterns::VerifyInstructionFlags() {
  unsigned Errors = 0;
  for (ptm_iterator I = ptm_begin(), E = ptm_end(); I != E; ++I) {
    const PatternToMatch &PTM = *I;
    SmallVector<Record*, 8> Instrs;
    getInstructionsInTree(PTM.getDstPattern(), Instrs);
    if (Instrs.empty())
      continue;

    // Count the number of instructions with each flag set.
    unsigned NumSideEffects = 0;
    unsigned NumStores = 0;
    unsigned NumLoads = 0;
    for (const Record *Instr : Instrs) {
      const CodeGenInstruction &InstInfo = Target.getInstruction(Instr);
      NumSideEffects += InstInfo.hasSideEffects;
      NumStores += InstInfo.mayStore;
      NumLoads += InstInfo.mayLoad;
    }

    // Analyze the source pattern.
    InstAnalyzer PatInfo(*this);
    PatInfo.Analyze(&PTM);

    // Collect error messages.
    SmallVector<std::string, 4> Msgs;

    // Check for missing flags in the output.
    // Permit extra flags for now at least.
    if (PatInfo.hasSideEffects && !NumSideEffects)
      Msgs.push_back("pattern has side effects, but hasSideEffects isn't set");

    // Don't verify store flags on instructions with side effects. At least for
    // intrinsics, side effects implies mayStore.
    if (!PatInfo.hasSideEffects && PatInfo.mayStore && !NumStores)
      Msgs.push_back("pattern may store, but mayStore isn't set");

    // Similarly, mayStore implies mayLoad on intrinsics.
    if (!PatInfo.mayStore && PatInfo.mayLoad && !NumLoads)
      Msgs.push_back("pattern may load, but mayLoad isn't set");

    // Print error messages.
    if (Msgs.empty())
      continue;
    ++Errors;

    for (const std::string &Msg : Msgs)
      PrintError(PTM.getSrcRecord()->getLoc(), Twine(Msg) + " on the " +
                 (Instrs.size() == 1 ?
                  "instruction" : "output instructions"));
    // Provide the location of the relevant instruction definitions.
    for (const Record *Instr : Instrs) {
      if (Instr != PTM.getSrcRecord())
        PrintError(Instr->getLoc(), "defined here");
      const CodeGenInstruction &InstInfo = Target.getInstruction(Instr);
      if (InstInfo.InferredFrom &&
          InstInfo.InferredFrom != InstInfo.TheDef &&
          InstInfo.InferredFrom != PTM.getSrcRecord())
        PrintError(InstInfo.InferredFrom->getLoc(), "inferred from pattern");
    }
  }
  if (Errors)
    PrintFatalError("Errors in DAG patterns");
}

/// Given a pattern result with an unresolved type, see if we can find one
/// instruction with an unresolved result type.  Force this result type to an
/// arbitrary element if it's possible types to converge results.
static bool ForceArbitraryInstResultType(TreePatternNode *N, TreePattern &TP) {
  if (N->isLeaf())
    return false;

  // Analyze children.
  for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i)
    if (ForceArbitraryInstResultType(N->getChild(i), TP))
      return true;

  if (!N->getOperator()->isSubClassOf("Instruction"))
    return false;

  // If this type is already concrete or completely unknown we can't do
  // anything.
  for (unsigned i = 0, e = N->getNumTypes(); i != e; ++i) {
    if (N->getExtType(i).isCompletelyUnknown() || N->getExtType(i).isConcrete())
      continue;

    // Otherwise, force its type to the first possibility (an arbitrary choice).
    if (N->getExtType(i).MergeInTypeInfo(N->getExtType(i).getTypeList()[0], TP))
      return true;
  }

  return false;
}

void CodeGenDAGPatterns::ParsePatterns() {
  std::vector<Record*> Patterns = Records.getAllDerivedDefinitions("Pattern");

  for (Record *CurPattern : Patterns) {
    DagInit *Tree = CurPattern->getValueAsDag("PatternToMatch");

    // If the pattern references the null_frag, there's nothing to do.
    if (hasNullFragReference(Tree))
      continue;

    TreePattern *Pattern = new TreePattern(CurPattern, Tree, true, *this);

    // Inline pattern fragments into it.
    Pattern->InlinePatternFragments();

    ListInit *LI = CurPattern->getValueAsListInit("ResultInstrs");
    if (LI->empty()) continue;  // no pattern.

    // Parse the instruction.
    TreePattern Result(CurPattern, LI, false, *this);

    // Inline pattern fragments into it.
    Result.InlinePatternFragments();

    if (Result.getNumTrees() != 1)
      Result.error("Cannot handle instructions producing instructions "
                   "with temporaries yet!");

    bool IterateInference;
    bool InferredAllPatternTypes, InferredAllResultTypes;
    do {
      // Infer as many types as possible.  If we cannot infer all of them, we
      // can never do anything with this pattern: report it to the user.
      InferredAllPatternTypes =
        Pattern->InferAllTypes(&Pattern->getNamedNodesMap());

      // Infer as many types as possible.  If we cannot infer all of them, we
      // can never do anything with this pattern: report it to the user.
      InferredAllResultTypes =
          Result.InferAllTypes(&Pattern->getNamedNodesMap());

      IterateInference = false;

      // Apply the type of the result to the source pattern.  This helps us
      // resolve cases where the input type is known to be a pointer type (which
      // is considered resolved), but the result knows it needs to be 32- or
      // 64-bits.  Infer the other way for good measure.
      for (unsigned i = 0, e = std::min(Result.getTree(0)->getNumTypes(),
                                        Pattern->getTree(0)->getNumTypes());
           i != e; ++i) {
        IterateInference = Pattern->getTree(0)->UpdateNodeType(
            i, Result.getTree(0)->getExtType(i), Result);
        IterateInference |= Result.getTree(0)->UpdateNodeType(
            i, Pattern->getTree(0)->getExtType(i), Result);
      }

      // If our iteration has converged and the input pattern's types are fully
      // resolved but the result pattern is not fully resolved, we may have a
      // situation where we have two instructions in the result pattern and
      // the instructions require a common register class, but don't care about
      // what actual MVT is used.  This is actually a bug in our modelling:
      // output patterns should have register classes, not MVTs.
      //
      // In any case, to handle this, we just go through and disambiguate some
      // arbitrary types to the result pattern's nodes.
      if (!IterateInference && InferredAllPatternTypes &&
          !InferredAllResultTypes)
        IterateInference =
            ForceArbitraryInstResultType(Result.getTree(0), Result);
    } while (IterateInference);

    // Verify that we inferred enough types that we can do something with the
    // pattern and result.  If these fire the user has to add type casts.
    if (!InferredAllPatternTypes)
      Pattern->error("Could not infer all types in pattern!");
    if (!InferredAllResultTypes) {
      Pattern->dump();
      Result.error("Could not infer all types in pattern result!");
    }

    // Validate that the input pattern is correct.
    std::map<std::string, TreePatternNode*> InstInputs;
    std::map<std::string, TreePatternNode*> InstResults;
    std::vector<Record*> InstImpResults;
    for (unsigned j = 0, ee = Pattern->getNumTrees(); j != ee; ++j)
      FindPatternInputsAndOutputs(Pattern, Pattern->getTree(j),
                                  InstInputs, InstResults,
                                  InstImpResults);

    // Promote the xform function to be an explicit node if set.
    TreePatternNode *DstPattern = Result.getOnlyTree();
    std::vector<TreePatternNode*> ResultNodeOperands;
    for (unsigned ii = 0, ee = DstPattern->getNumChildren(); ii != ee; ++ii) {
      TreePatternNode *OpNode = DstPattern->getChild(ii);
      if (Record *Xform = OpNode->getTransformFn()) {
        OpNode->setTransformFn(nullptr);
        std::vector<TreePatternNode*> Children;
        Children.push_back(OpNode);
        OpNode = new TreePatternNode(Xform, Children, OpNode->getNumTypes());
      }
      ResultNodeOperands.push_back(OpNode);
    }
    DstPattern = Result.getOnlyTree();
    if (!DstPattern->isLeaf())
      DstPattern = new TreePatternNode(DstPattern->getOperator(),
                                       ResultNodeOperands,
                                       DstPattern->getNumTypes());

    for (unsigned i = 0, e = Result.getOnlyTree()->getNumTypes(); i != e; ++i)
      DstPattern->setType(i, Result.getOnlyTree()->getExtType(i));

    TreePattern Temp(Result.getRecord(), DstPattern, false, *this);
    Temp.InferAllTypes();


    AddPatternToMatch(Pattern,
                    PatternToMatch(CurPattern,
                                   CurPattern->getValueAsListInit("Predicates"),
                                   Pattern->getTree(0),
                                   Temp.getOnlyTree(), InstImpResults,
                                   CurPattern->getValueAsInt("AddedComplexity"),
                                   CurPattern->getID()));
  }
}

/// CombineChildVariants - Given a bunch of permutations of each child of the
/// 'operator' node, put them together in all possible ways.
static void CombineChildVariants(TreePatternNode *Orig,
               const std::vector<std::vector<TreePatternNode*> > &ChildVariants,
                                 std::vector<TreePatternNode*> &OutVariants,
                                 CodeGenDAGPatterns &CDP,
                                 const MultipleUseVarSet &DepVars) {
  // Make sure that each operand has at least one variant to choose from.
  for (const auto &Variants : ChildVariants)
    if (Variants.empty())
      return;

  // The end result is an all-pairs construction of the resultant pattern.
  std::vector<unsigned> Idxs;
  Idxs.resize(ChildVariants.size());
  bool NotDone;
  do {
#ifndef NDEBUG
    DEBUG(if (!Idxs.empty()) {
            errs() << Orig->getOperator()->getName() << ": Idxs = [ ";
              for (unsigned Idx : Idxs) {
                errs() << Idx << " ";
            }
            errs() << "]\n";
          });
#endif
    // Create the variant and add it to the output list.
    std::vector<TreePatternNode*> NewChildren;
    for (unsigned i = 0, e = ChildVariants.size(); i != e; ++i)
      NewChildren.push_back(ChildVariants[i][Idxs[i]]);
    auto R = llvm::make_unique<TreePatternNode>(
        Orig->getOperator(), NewChildren, Orig->getNumTypes());

    // Copy over properties.
    R->setName(Orig->getName());
    R->setPredicateFns(Orig->getPredicateFns());
    R->setTransformFn(Orig->getTransformFn());
    for (unsigned i = 0, e = Orig->getNumTypes(); i != e; ++i)
      R->setType(i, Orig->getExtType(i));

    // If this pattern cannot match, do not include it as a variant.
    std::string ErrString;
    // Scan to see if this pattern has already been emitted.  We can get
    // duplication due to things like commuting:
    //   (and GPRC:$a, GPRC:$b) -> (and GPRC:$b, GPRC:$a)
    // which are the same pattern.  Ignore the dups.
    if (R->canPatternMatch(ErrString, CDP) &&
        std::none_of(OutVariants.begin(), OutVariants.end(),
                     [&](TreePatternNode *Variant) {
                       return R->isIsomorphicTo(Variant, DepVars);
                     }))
      OutVariants.push_back(R.release());

    // Increment indices to the next permutation by incrementing the
    // indices from last index backward, e.g., generate the sequence
    // [0, 0], [0, 1], [1, 0], [1, 1].
    int IdxsIdx;
    for (IdxsIdx = Idxs.size() - 1; IdxsIdx >= 0; --IdxsIdx) {
      if (++Idxs[IdxsIdx] == ChildVariants[IdxsIdx].size())
        Idxs[IdxsIdx] = 0;
      else
        break;
    }
    NotDone = (IdxsIdx >= 0);
  } while (NotDone);
}

/// CombineChildVariants - A helper function for binary operators.
///
static void CombineChildVariants(TreePatternNode *Orig,
                                 const std::vector<TreePatternNode*> &LHS,
                                 const std::vector<TreePatternNode*> &RHS,
                                 std::vector<TreePatternNode*> &OutVariants,
                                 CodeGenDAGPatterns &CDP,
                                 const MultipleUseVarSet &DepVars) {
  std::vector<std::vector<TreePatternNode*> > ChildVariants;
  ChildVariants.push_back(LHS);
  ChildVariants.push_back(RHS);
  CombineChildVariants(Orig, ChildVariants, OutVariants, CDP, DepVars);
}


static void GatherChildrenOfAssociativeOpcode(TreePatternNode *N,
                                     std::vector<TreePatternNode *> &Children) {
  assert(N->getNumChildren()==2 &&"Associative but doesn't have 2 children!");
  Record *Operator = N->getOperator();

  // Only permit raw nodes.
  if (!N->getName().empty() || !N->getPredicateFns().empty() ||
      N->getTransformFn()) {
    Children.push_back(N);
    return;
  }

  if (N->getChild(0)->isLeaf() || N->getChild(0)->getOperator() != Operator)
    Children.push_back(N->getChild(0));
  else
    GatherChildrenOfAssociativeOpcode(N->getChild(0), Children);

  if (N->getChild(1)->isLeaf() || N->getChild(1)->getOperator() != Operator)
    Children.push_back(N->getChild(1));
  else
    GatherChildrenOfAssociativeOpcode(N->getChild(1), Children);
}

/// GenerateVariantsOf - Given a pattern N, generate all permutations we can of
/// the (potentially recursive) pattern by using algebraic laws.
///
static void GenerateVariantsOf(TreePatternNode *N,
                               std::vector<TreePatternNode*> &OutVariants,
                               CodeGenDAGPatterns &CDP,
                               const MultipleUseVarSet &DepVars) {
  // We cannot permute leaves or ComplexPattern uses.
  if (N->isLeaf() || N->getOperator()->isSubClassOf("ComplexPattern")) {
    OutVariants.push_back(N);
    return;
  }

  // Look up interesting info about the node.
  const SDNodeInfo &NodeInfo = CDP.getSDNodeInfo(N->getOperator());

  // If this node is associative, re-associate.
  if (NodeInfo.hasProperty(SDNPAssociative)) {
    // Re-associate by pulling together all of the linked operators
    std::vector<TreePatternNode*> MaximalChildren;
    GatherChildrenOfAssociativeOpcode(N, MaximalChildren);

    // Only handle child sizes of 3.  Otherwise we'll end up trying too many
    // permutations.
    if (MaximalChildren.size() == 3) {
      // Find the variants of all of our maximal children.
      std::vector<TreePatternNode*> AVariants, BVariants, CVariants;
      GenerateVariantsOf(MaximalChildren[0], AVariants, CDP, DepVars);
      GenerateVariantsOf(MaximalChildren[1], BVariants, CDP, DepVars);
      GenerateVariantsOf(MaximalChildren[2], CVariants, CDP, DepVars);

      // There are only two ways we can permute the tree:
      //   (A op B) op C    and    A op (B op C)
      // Within these forms, we can also permute A/B/C.

      // Generate legal pair permutations of A/B/C.
      std::vector<TreePatternNode*> ABVariants;
      std::vector<TreePatternNode*> BAVariants;
      std::vector<TreePatternNode*> ACVariants;
      std::vector<TreePatternNode*> CAVariants;
      std::vector<TreePatternNode*> BCVariants;
      std::vector<TreePatternNode*> CBVariants;
      CombineChildVariants(N, AVariants, BVariants, ABVariants, CDP, DepVars);
      CombineChildVariants(N, BVariants, AVariants, BAVariants, CDP, DepVars);
      CombineChildVariants(N, AVariants, CVariants, ACVariants, CDP, DepVars);
      CombineChildVariants(N, CVariants, AVariants, CAVariants, CDP, DepVars);
      CombineChildVariants(N, BVariants, CVariants, BCVariants, CDP, DepVars);
      CombineChildVariants(N, CVariants, BVariants, CBVariants, CDP, DepVars);

      // Combine those into the result: (x op x) op x
      CombineChildVariants(N, ABVariants, CVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, BAVariants, CVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, ACVariants, BVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, CAVariants, BVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, BCVariants, AVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, CBVariants, AVariants, OutVariants, CDP, DepVars);

      // Combine those into the result: x op (x op x)
      CombineChildVariants(N, CVariants, ABVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, CVariants, BAVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, BVariants, ACVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, BVariants, CAVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, AVariants, BCVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, AVariants, CBVariants, OutVariants, CDP, DepVars);
      return;
    }
  }

  // Compute permutations of all children.
  std::vector<std::vector<TreePatternNode*> > ChildVariants;
  ChildVariants.resize(N->getNumChildren());
  for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i)
    GenerateVariantsOf(N->getChild(i), ChildVariants[i], CDP, DepVars);

  // Build all permutations based on how the children were formed.
  CombineChildVariants(N, ChildVariants, OutVariants, CDP, DepVars);

  // If this node is commutative, consider the commuted order.
  bool isCommIntrinsic = N->isCommutativeIntrinsic(CDP);
  if (NodeInfo.hasProperty(SDNPCommutative) || isCommIntrinsic) {
    assert((N->getNumChildren()==2 || isCommIntrinsic) &&
           "Commutative but doesn't have 2 children!");
    // Don't count children which are actually register references.
    unsigned NC = 0;
    for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i) {
      TreePatternNode *Child = N->getChild(i);
      if (Child->isLeaf())
        if (DefInit *DI = dyn_cast<DefInit>(Child->getLeafValue())) {
          Record *RR = DI->getDef();
          if (RR->isSubClassOf("Register"))
            continue;
        }
      NC++;
    }
    // Consider the commuted order.
    if (isCommIntrinsic) {
      // Commutative intrinsic. First operand is the intrinsic id, 2nd and 3rd
      // operands are the commutative operands, and there might be more operands
      // after those.
      assert(NC >= 3 &&
             "Commutative intrinsic should have at least 3 children!");
      std::vector<std::vector<TreePatternNode*> > Variants;
      Variants.push_back(ChildVariants[0]); // Intrinsic id.
      Variants.push_back(ChildVariants[2]);
      Variants.push_back(ChildVariants[1]);
      for (unsigned i = 3; i != NC; ++i)
        Variants.push_back(ChildVariants[i]);
      CombineChildVariants(N, Variants, OutVariants, CDP, DepVars);
    } else if (NC == 2)
      CombineChildVariants(N, ChildVariants[1], ChildVariants[0],
                           OutVariants, CDP, DepVars);
  }
}


// GenerateVariants - Generate variants.  For example, commutative patterns can
// match multiple ways.  Add them to PatternsToMatch as well.
void CodeGenDAGPatterns::GenerateVariants() {
  DEBUG(errs() << "Generating instruction variants.\n");

  // Loop over all of the patterns we've collected, checking to see if we can
  // generate variants of the instruction, through the exploitation of
  // identities.  This permits the target to provide aggressive matching without
  // the .td file having to contain tons of variants of instructions.
  //
  // Note that this loop adds new patterns to the PatternsToMatch list, but we
  // intentionally do not reconsider these.  Any variants of added patterns have
  // already been added.
  //
  for (unsigned i = 0, e = PatternsToMatch.size(); i != e; ++i) {
    MultipleUseVarSet             DepVars;
    std::vector<TreePatternNode*> Variants;
    FindDepVars(PatternsToMatch[i].getSrcPattern(), DepVars);
    DEBUG(errs() << "Dependent/multiply used variables: ");
    DEBUG(DumpDepVars(DepVars));
    DEBUG(errs() << "\n");
    GenerateVariantsOf(PatternsToMatch[i].getSrcPattern(), Variants, *this,
                       DepVars);

    assert(!Variants.empty() && "Must create at least original variant!");
    Variants.erase(Variants.begin());  // Remove the original pattern.

    if (Variants.empty())  // No variants for this pattern.
      continue;

    DEBUG(errs() << "FOUND VARIANTS OF: ";
          PatternsToMatch[i].getSrcPattern()->dump();
          errs() << "\n");

    for (unsigned v = 0, e = Variants.size(); v != e; ++v) {
      TreePatternNode *Variant = Variants[v];

      DEBUG(errs() << "  VAR#" << v <<  ": ";
            Variant->dump();
            errs() << "\n");

      // Scan to see if an instruction or explicit pattern already matches this.
      bool AlreadyExists = false;
      for (unsigned p = 0, e = PatternsToMatch.size(); p != e; ++p) {
        // Skip if the top level predicates do not match.
        if (PatternsToMatch[i].getPredicates() !=
            PatternsToMatch[p].getPredicates())
          continue;
        // Check to see if this variant already exists.
        if (Variant->isIsomorphicTo(PatternsToMatch[p].getSrcPattern(),
                                    DepVars)) {
          DEBUG(errs() << "  *** ALREADY EXISTS, ignoring variant.\n");
          AlreadyExists = true;
          break;
        }
      }
      // If we already have it, ignore the variant.
      if (AlreadyExists) continue;

      // Otherwise, add it to the list of patterns we have.
      PatternsToMatch.emplace_back(
          PatternsToMatch[i].getSrcRecord(), PatternsToMatch[i].getPredicates(),
          Variant, PatternsToMatch[i].getDstPattern(),
          PatternsToMatch[i].getDstRegs(),
          PatternsToMatch[i].getAddedComplexity(), Record::getNewUID());
    }

    DEBUG(errs() << "\n");
  }
}
