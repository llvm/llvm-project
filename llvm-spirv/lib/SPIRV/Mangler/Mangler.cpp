//===--------------------------- Mangler.cpp -----------------------------===//
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
#include "ManglingUtils.h"
#include "NameMangleAPI.h"
#include "ParameterType.h"
#include "SPIRVInternal.h"
#include <algorithm>
#include <map>
#include <sstream>
#include <string>

// According to IA64 name mangling spec,
// builtin vector types should not be substituted
// This is a workaround till this gets fixed in CLang
#define ENABLE_MANGLER_VECTOR_SUBSTITUTION 1

namespace SPIR {

class MangleVisitor : public TypeVisitor {
public:
  MangleVisitor(SPIRversion Ver, std::stringstream &S)
      : TypeVisitor(Ver), Stream(S), SeqId(0) {}

  //
  // mangle substitution methods
  //
  void mangleSequenceID(unsigned SeqID) {
    if (SeqID == 1)
      Stream << '0';
    else if (SeqID > 1) {
      std::string Bstr;
      std::string Charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
      SeqID--;
      Bstr.reserve(7);
      for (; SeqID != 0; SeqID /= 36)
        Bstr += Charset.substr(SeqID % 36, 1);
      std::reverse(Bstr.begin(), Bstr.end());
      Stream << Bstr;
    }
    Stream << '_';
  }

  bool mangleSubstitution(const ParamType *Type, std::string TypeStr) {
    size_t Fpos;
    std::stringstream ThistypeStr;
    ThistypeStr << TypeStr;
    if ((Fpos = Stream.str().find(TypeStr)) != std::string::npos) {
      const char *NType;
      if (const PointerType *P = SPIR::dynCast<PointerType>(Type)) {
        if ((NType =
                 mangledPrimitiveStringfromName(P->getPointee()->toString())))
          ThistypeStr << NType;
      }
#if defined(ENABLE_MANGLER_VECTOR_SUBSTITUTION)
      else if (const VectorType *PVec = SPIR::dynCast<VectorType>(Type)) {
        if ((NType = mangledPrimitiveStringfromName(
                 PVec->getScalarType()->toString())))
          ThistypeStr << NType;
      }
#endif
      std::map<std::string, unsigned>::iterator I =
          Substitutions.find(ThistypeStr.str());
      if (I == Substitutions.end())
        return false;

      unsigned SeqID = I->second;
      Stream << 'S';
      mangleSequenceID(SeqID);
      return true;
    }
    return false;
  }

  //
  // Visit methods
  //
  MangleError visit(const PrimitiveType *T) override {
    MangleError Me = MANGLE_SUCCESS;
#if defined(SPIRV_SPIR20_MANGLING_REQUIREMENTS)
    Stream << mangledPrimitiveString(t->getPrimitive());
#else
    std::string MangledPrimitive =
        std::string(mangledPrimitiveString(T->getPrimitive()));
    // out of all enums it makes sense to substitute only
    // memory_scope/memory_order since only they appear several times in the
    // builtin declaration.
    if (MangledPrimitive == "12memory_scope" ||
        MangledPrimitive == "12memory_order") {
      if (!mangleSubstitution(T, mangledPrimitiveString(T->getPrimitive()))) {
        size_t Index = Stream.str().size();
        Stream << mangledPrimitiveString(T->getPrimitive());
        Substitutions[Stream.str().substr(Index)] = SeqId++;
      }
    } else {
      Stream << MangledPrimitive;
    }
#endif
    return Me;
  }

  MangleError visit(const PointerType *P) override {
    size_t Fpos = Stream.str().size();
    std::string QualStr;
    MangleError Me = MANGLE_SUCCESS;
    QualStr += getMangledAttribute((P->getAddressSpace()));
    for (unsigned int I = ATTR_QUALIFIER_FIRST; I <= ATTR_QUALIFIER_LAST; I++) {
      TypeAttributeEnum Qualifier = (TypeAttributeEnum)I;
      if (P->hasQualifier(Qualifier)) {
        QualStr += getMangledAttribute(Qualifier);
      }
    }
    if (!mangleSubstitution(P, "P" + QualStr)) {
      // A pointee type is substituted when it is a user type, a vector type
      // (but see a comment in the beginning of this file), a pointer type,
      // or a primitive type with qualifiers (addr. space and/or CV qualifiers).
      // So, stream "P", type qualifiers
      Stream << "P" << QualStr;
      // and the pointee type itself.
      Me = P->getPointee()->accept(this);
      // The type qualifiers plus a pointee type is a substitutable entity
      Substitutions[Stream.str().substr(Fpos + 1)] = SeqId++;
      // The complete pointer type is substitutable as well
      Substitutions[Stream.str().substr(Fpos)] = SeqId++;
    }
    return Me;
  }

  MangleError visit(const VectorType *V) override {
    size_t Index = Stream.str().size();
    std::stringstream TypeStr;
    TypeStr << "Dv" << V->getLength() << "_";
    MangleError Me = MANGLE_SUCCESS;
#if defined(ENABLE_MANGLER_VECTOR_SUBSTITUTION)
    if (!mangleSubstitution(V, TypeStr.str()))
#endif
    {
      Stream << TypeStr.str();
      Me = V->getScalarType()->accept(this);
      Substitutions[Stream.str().substr(Index)] = SeqId++;
    }
    return Me;
  }

  MangleError visit(const AtomicType *P) override {
    MangleError Me = MANGLE_SUCCESS;
    size_t Index = Stream.str().size();
    const char *TypeStr = "U7_Atomic";
    if (!mangleSubstitution(P, TypeStr)) {
      Stream << TypeStr;
      Me = P->getBaseType()->accept(this);
      Substitutions[Stream.str().substr(Index)] = SeqId++;
    }
    return Me;
  }

  MangleError visit(const BlockType *P) override {
    Stream << "U"
             << "13block_pointerFv";
    if (P->getNumOfParams() == 0)
      Stream << "v";
    else
      for (unsigned int I = 0; I < P->getNumOfParams(); ++I) {
        MangleError Err = P->getParam(I)->accept(this);
        if (Err != MANGLE_SUCCESS) {
          return Err;
        }
      }
    Stream << "E";
    return MANGLE_SUCCESS;
  }

  MangleError visit(const UserDefinedType *PTy) override {
    std::string Name = PTy->toString();
    Stream << Name.size() << Name;
    return MANGLE_SUCCESS;
  }

private:
  // Holds the mangled string representing the prototype of the function.
  std::stringstream &Stream;
  unsigned SeqId;
  std::map<std::string, unsigned> Substitutions;
};

//
// NameMangler
//
NameMangler::NameMangler(SPIRversion Version) : SpirVersion(Version) {}

MangleError NameMangler::mangle(const FunctionDescriptor &Fd,
                                std::string &MangledName) {
  if (Fd.isNull()) {
    MangledName.assign(FunctionDescriptor::nullString());
    return MANGLE_NULL_FUNC_DESCRIPTOR;
  }
  std::stringstream Ret;
  Ret << "_Z" << Fd.Name.length() << Fd.Name;
  MangleVisitor Visitor(SpirVersion, Ret);
  for (unsigned int I = 0; I < Fd.Parameters.size(); ++I) {
    MangleError Err = Fd.Parameters[I]->accept(&Visitor);
    if (Err == MANGLE_TYPE_NOT_SUPPORTED) {
      MangledName.assign("Type ");
      MangledName.append(Fd.Parameters[I]->toString());
      MangledName.append(" is not supported in ");
      std::string Ver = getSPIRVersionAsString(SpirVersion);
      MangledName.append(Ver);
      return Err;
    }
  }
  MangledName.assign(Ret.str());
  return MANGLE_SUCCESS;
}

} // namespace SPIR
