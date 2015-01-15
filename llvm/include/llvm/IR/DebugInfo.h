//===- DebugInfo.h - Debug Information Helpers ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a bunch of datatypes that are useful for creating and
// walking debug info in LLVM IR form. They essentially provide wrappers around
// the information in the global variables that's needed when constructing the
// DWARF information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DEBUGINFO_H
#define LLVM_IR_DEBUGINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include <iterator>

namespace llvm {
class BasicBlock;
class Constant;
class Function;
class GlobalVariable;
class Module;
class Type;
class Value;
class DbgDeclareInst;
class DbgValueInst;
class Instruction;
class Metadata;
class MDNode;
class MDString;
class NamedMDNode;
class LLVMContext;
class raw_ostream;

class DIFile;
class DISubprogram;
class DILexicalBlock;
class DILexicalBlockFile;
class DIVariable;
class DIType;
class DIScope;
class DIObjCProperty;

/// \brief Maps from type identifier to the actual MDNode.
typedef DenseMap<const MDString *, MDNode *> DITypeIdentifierMap;

class DIHeaderFieldIterator
    : public std::iterator<std::input_iterator_tag, StringRef, std::ptrdiff_t,
                           const StringRef *, StringRef> {
  StringRef Header;
  StringRef Current;

public:
  DIHeaderFieldIterator() {}
  DIHeaderFieldIterator(StringRef Header)
      : Header(Header), Current(Header.slice(0, Header.find('\0'))) {}
  StringRef operator*() const { return Current; }
  const StringRef * operator->() const { return &Current; }
  DIHeaderFieldIterator &operator++() {
    increment();
    return *this;
  }
  DIHeaderFieldIterator operator++(int) {
    DIHeaderFieldIterator X(*this);
    increment();
    return X;
  }
  bool operator==(const DIHeaderFieldIterator &X) const {
    return Current.data() == X.Current.data();
  }
  bool operator!=(const DIHeaderFieldIterator &X) const {
    return !(*this == X);
  }

  StringRef getHeader() const { return Header; }
  StringRef getCurrent() const { return Current; }
  StringRef getPrefix() const {
    if (Current.begin() == Header.begin())
      return StringRef();
    return Header.slice(0, Current.begin() - Header.begin() - 1);
  }
  StringRef getSuffix() const {
    if (Current.end() == Header.end())
      return StringRef();
    return Header.slice(Current.end() - Header.begin() + 1, StringRef::npos);
  }

private:
  void increment() {
    assert(Current.data() != nullptr && "Cannot increment past the end");
    StringRef Suffix = getSuffix();
    Current = Suffix.slice(0, Suffix.find('\0'));
  }
};

/// \brief A thin wraper around MDNode to access encoded debug info.
///
/// This should not be stored in a container, because the underlying MDNode may
/// change in certain situations.
class DIDescriptor {
  // Befriends DIRef so DIRef can befriend the protected member
  // function: getFieldAs<DIRef>.
  template <typename T> friend class DIRef;

public:
  /// \brief Accessibility flags.
  ///
  /// The three accessibility flags are mutually exclusive and rolled together
  /// in the first two bits.
  enum {
    FlagAccessibility     = 1 << 0 | 1 << 1,
    FlagPrivate           = 1,
    FlagProtected         = 2,
    FlagPublic            = 3,

    FlagFwdDecl           = 1 << 2,
    FlagAppleBlock        = 1 << 3,
    FlagBlockByrefStruct  = 1 << 4,
    FlagVirtual           = 1 << 5,
    FlagArtificial        = 1 << 6,
    FlagExplicit          = 1 << 7,
    FlagPrototyped        = 1 << 8,
    FlagObjcClassComplete = 1 << 9,
    FlagObjectPointer     = 1 << 10,
    FlagVector            = 1 << 11,
    FlagStaticMember      = 1 << 12,
    FlagIndirectVariable  = 1 << 13,
    FlagLValueReference   = 1 << 14,
    FlagRValueReference   = 1 << 15
  };

protected:
  const MDNode *DbgNode;

  StringRef getStringField(unsigned Elt) const;
  unsigned getUnsignedField(unsigned Elt) const {
    return (unsigned)getUInt64Field(Elt);
  }
  uint64_t getUInt64Field(unsigned Elt) const;
  int64_t getInt64Field(unsigned Elt) const;
  DIDescriptor getDescriptorField(unsigned Elt) const;

  template <typename DescTy> DescTy getFieldAs(unsigned Elt) const {
    return DescTy(getDescriptorField(Elt));
  }

  GlobalVariable *getGlobalVariableField(unsigned Elt) const;
  Constant *getConstantField(unsigned Elt) const;
  Function *getFunctionField(unsigned Elt) const;
  void replaceFunctionField(unsigned Elt, Function *F);

public:
  explicit DIDescriptor(const MDNode *N = nullptr) : DbgNode(N) {}

  bool Verify() const;

  MDNode *get() const { return const_cast<MDNode *>(DbgNode); }
  operator MDNode *() const { return get(); }
  MDNode *operator->() const { return get(); }

  // An explicit operator bool so that we can do testing of DI values
  // easily.
  // FIXME: This operator bool isn't actually protecting anything at the
  // moment due to the conversion operator above making DIDescriptor nodes
  // implicitly convertable to bool.
  LLVM_EXPLICIT operator bool() const { return DbgNode != nullptr; }

  bool operator==(DIDescriptor Other) const { return DbgNode == Other.DbgNode; }
  bool operator!=(DIDescriptor Other) const { return !operator==(Other); }

  StringRef getHeader() const {
    return getStringField(0);
  }

  size_t getNumHeaderFields() const {
    return std::distance(DIHeaderFieldIterator(getHeader()),
                         DIHeaderFieldIterator());
  }

  StringRef getHeaderField(unsigned Index) const {
    // Since callers expect an empty string for out-of-range accesses, we can't
    // use std::advance() here.
    for (DIHeaderFieldIterator I(getHeader()), E; I != E; ++I, --Index)
      if (!Index)
        return *I;
    return StringRef();
  }

  template <class T> T getHeaderFieldAs(unsigned Index) const {
    T Int;
    if (getHeaderField(Index).getAsInteger(0, Int))
      return 0;
    return Int;
  }

  uint16_t getTag() const { return getHeaderFieldAs<uint16_t>(0); }

  bool isDerivedType() const;
  bool isCompositeType() const;
  bool isSubroutineType() const;
  bool isBasicType() const;
  bool isVariable() const;
  bool isSubprogram() const;
  bool isGlobalVariable() const;
  bool isScope() const;
  bool isFile() const;
  bool isCompileUnit() const;
  bool isNameSpace() const;
  bool isLexicalBlockFile() const;
  bool isLexicalBlock() const;
  bool isSubrange() const;
  bool isEnumerator() const;
  bool isType() const;
  bool isTemplateTypeParameter() const;
  bool isTemplateValueParameter() const;
  bool isObjCProperty() const;
  bool isImportedEntity() const;
  bool isExpression() const;

  void print(raw_ostream &OS) const;
  void dump() const;

  /// \brief Replace all uses of debug info referenced by this descriptor.
  void replaceAllUsesWith(LLVMContext &VMContext, DIDescriptor D);
  void replaceAllUsesWith(MDNode *D);
};

/// \brief This is used to represent ranges, for array bounds.
class DISubrange : public DIDescriptor {
  friend class DIDescriptor;
  void printInternal(raw_ostream &OS) const;

public:
  explicit DISubrange(const MDNode *N = nullptr) : DIDescriptor(N) {}

  int64_t getLo() const { return getHeaderFieldAs<int64_t>(1); }
  int64_t getCount() const { return getHeaderFieldAs<int64_t>(2); }
  bool Verify() const;
};

/// \brief This descriptor holds an array of nodes with type T.
template <typename T> class DITypedArray : public DIDescriptor {
public:
  explicit DITypedArray(const MDNode *N = nullptr) : DIDescriptor(N) {}
  unsigned getNumElements() const {
    return DbgNode ? DbgNode->getNumOperands() : 0;
  }
  T getElement(unsigned Idx) const {
    return getFieldAs<T>(Idx);
  }
};

typedef DITypedArray<DIDescriptor> DIArray;

/// \brief A wrapper for an enumerator (e.g. X and Y in 'enum {X,Y}').
///
/// FIXME: it seems strange that this doesn't have either a reference to the
/// type/precision or a file/line pair for location info.
class DIEnumerator : public DIDescriptor {
  friend class DIDescriptor;
  void printInternal(raw_ostream &OS) const;

public:
  explicit DIEnumerator(const MDNode *N = nullptr) : DIDescriptor(N) {}

  StringRef getName() const { return getHeaderField(1); }
  int64_t getEnumValue() const { return getHeaderFieldAs<int64_t>(2); }
  bool Verify() const;
};

template <typename T> class DIRef;
typedef DIRef<DIScope> DIScopeRef;
typedef DIRef<DIType> DITypeRef;
typedef DITypedArray<DITypeRef> DITypeArray;

/// \brief A base class for various scopes.
///
/// Although, implementation-wise, DIScope is the parent class of most
/// other DIxxx classes, including DIType and its descendants, most of
/// DIScope's descendants are not a substitutable subtype of
/// DIScope. The DIDescriptor::isScope() method only is true for
/// DIScopes that are scopes in the strict lexical scope sense
/// (DICompileUnit, DISubprogram, etc.), but not for, e.g., a DIType.
class DIScope : public DIDescriptor {
protected:
  friend class DIDescriptor;
  void printInternal(raw_ostream &OS) const;

public:
  explicit DIScope(const MDNode *N = nullptr) : DIDescriptor(N) {}

  /// \brief Get the parent scope.
  ///
  /// Gets the parent scope for this scope node or returns a default
  /// constructed scope.
  DIScopeRef getContext() const;
  /// \brief Get the scope name.
  ///
  /// If the scope node has a name, return that, else return an empty string.
  StringRef getName() const;
  StringRef getFilename() const;
  StringRef getDirectory() const;

  /// \brief Generate a reference to this DIScope.
  ///
  /// Uses the type identifier instead of the actual MDNode if possible, to
  /// help type uniquing.
  DIScopeRef getRef() const;
};

/// \brief Represents reference to a DIDescriptor.
///
/// Abstracts over direct and identifier-based metadata references.
template <typename T> class DIRef {
  template <typename DescTy>
  friend DescTy DIDescriptor::getFieldAs(unsigned Elt) const;
  friend DIScopeRef DIScope::getContext() const;
  friend DIScopeRef DIScope::getRef() const;
  friend class DIType;

  /// \brief Val can be either a MDNode or a MDString.
  ///
  /// In the latter, MDString specifies the type identifier.
  const Metadata *Val;
  explicit DIRef(const Metadata *V);

public:
  T resolve(const DITypeIdentifierMap &Map) const;
  StringRef getName() const;
  operator Metadata *() const { return const_cast<Metadata *>(Val); }
};

template <typename T>
T DIRef<T>::resolve(const DITypeIdentifierMap &Map) const {
  if (!Val)
    return T();

  if (const MDNode *MD = dyn_cast<MDNode>(Val))
    return T(MD);

  const MDString *MS = cast<MDString>(Val);
  // Find the corresponding MDNode.
  DITypeIdentifierMap::const_iterator Iter = Map.find(MS);
  assert(Iter != Map.end() && "Identifier not in the type map?");
  assert(DIDescriptor(Iter->second).isType() &&
         "MDNode in DITypeIdentifierMap should be a DIType.");
  return T(Iter->second);
}

template <typename T> StringRef DIRef<T>::getName() const {
  if (!Val)
    return StringRef();

  if (const MDNode *MD = dyn_cast<MDNode>(Val))
    return T(MD).getName();

  const MDString *MS = cast<MDString>(Val);
  return MS->getString();
}

/// \brief Handle fields that are references to DIScopes.
template <> DIScopeRef DIDescriptor::getFieldAs<DIScopeRef>(unsigned Elt) const;
/// \brief Specialize DIRef constructor for DIScopeRef.
template <> DIRef<DIScope>::DIRef(const Metadata *V);

/// \brief Handle fields that are references to DITypes.
template <> DITypeRef DIDescriptor::getFieldAs<DITypeRef>(unsigned Elt) const;
/// \brief Specialize DIRef constructor for DITypeRef.
template <> DIRef<DIType>::DIRef(const Metadata *V);

/// \briefThis is a wrapper for a type.
///
/// FIXME: Types should be factored much better so that CV qualifiers and
/// others do not require a huge and empty descriptor full of zeros.
class DIType : public DIScope {
protected:
  friend class DIDescriptor;
  void printInternal(raw_ostream &OS) const;

public:
  explicit DIType(const MDNode *N = nullptr) : DIScope(N) {}
  operator DITypeRef () const {
    assert(isType() &&
           "constructing DITypeRef from an MDNode that is not a type");
    return DITypeRef(&*getRef());
  }

  bool Verify() const;

  DIScopeRef getContext() const { return getFieldAs<DIScopeRef>(2); }
  StringRef getName() const { return getHeaderField(1); }
  unsigned getLineNumber() const {
    return getHeaderFieldAs<unsigned>(2);
  }
  uint64_t getSizeInBits() const {
    return getHeaderFieldAs<unsigned>(3);
  }
  uint64_t getAlignInBits() const {
    return getHeaderFieldAs<unsigned>(4);
  }
  // FIXME: Offset is only used for DW_TAG_member nodes.  Making every type
  // carry this is just plain insane.
  uint64_t getOffsetInBits() const {
    return getHeaderFieldAs<unsigned>(5);
  }
  unsigned getFlags() const { return getHeaderFieldAs<unsigned>(6); }
  bool isPrivate() const {
    return (getFlags() & FlagAccessibility) == FlagPrivate;
  }
  bool isProtected() const {
    return (getFlags() & FlagAccessibility) == FlagProtected;
  }
  bool isPublic() const {
    return (getFlags() & FlagAccessibility) == FlagPublic;
  }
  bool isForwardDecl() const { return (getFlags() & FlagFwdDecl) != 0; }
  bool isAppleBlockExtension() const {
    return (getFlags() & FlagAppleBlock) != 0;
  }
  bool isBlockByrefStruct() const {
    return (getFlags() & FlagBlockByrefStruct) != 0;
  }
  bool isVirtual() const { return (getFlags() & FlagVirtual) != 0; }
  bool isArtificial() const { return (getFlags() & FlagArtificial) != 0; }
  bool isObjectPointer() const { return (getFlags() & FlagObjectPointer) != 0; }
  bool isObjcClassComplete() const {
    return (getFlags() & FlagObjcClassComplete) != 0;
  }
  bool isVector() const { return (getFlags() & FlagVector) != 0; }
  bool isStaticMember() const { return (getFlags() & FlagStaticMember) != 0; }
  bool isLValueReference() const {
    return (getFlags() & FlagLValueReference) != 0;
  }
  bool isRValueReference() const {
    return (getFlags() & FlagRValueReference) != 0;
  }
  bool isValid() const { return DbgNode && isType(); }
};

/// \brief A basic type, like 'int' or 'float'.
class DIBasicType : public DIType {
public:
  explicit DIBasicType(const MDNode *N = nullptr) : DIType(N) {}

  unsigned getEncoding() const { return getHeaderFieldAs<unsigned>(7); }

  bool Verify() const;
};

/// \brief A simple derived type
///
/// Like a const qualified type, a typedef, a pointer or reference, et cetera.
/// Or, a data member of a class/struct/union.
class DIDerivedType : public DIType {
  friend class DIDescriptor;
  void printInternal(raw_ostream &OS) const;

public:
  explicit DIDerivedType(const MDNode *N = nullptr) : DIType(N) {}

  DITypeRef getTypeDerivedFrom() const { return getFieldAs<DITypeRef>(3); }

  /// \brief Return property node, if this ivar is associated with one.
  MDNode *getObjCProperty() const;

  DITypeRef getClassType() const {
    assert(getTag() == dwarf::DW_TAG_ptr_to_member_type);
    return getFieldAs<DITypeRef>(4);
  }

  Constant *getConstant() const {
    assert((getTag() == dwarf::DW_TAG_member) && isStaticMember());
    return getConstantField(4);
  }

  bool Verify() const;
};

/// \brief Types that refer to multiple other types.
///
/// This descriptor holds a type that can refer to multiple other types, like a
/// function or struct.
///
/// DICompositeType is derived from DIDerivedType because some
/// composite types (such as enums) can be derived from basic types
// FIXME: Make this derive from DIType directly & just store the
// base type in a single DIType field.
class DICompositeType : public DIDerivedType {
  friend class DIBuilder;
  friend class DIDescriptor;
  void printInternal(raw_ostream &OS) const;

  /// \brief Set the array of member DITypes.
  void setArraysHelper(MDNode *Elements, MDNode *TParams);

public:
  explicit DICompositeType(const MDNode *N = nullptr) : DIDerivedType(N) {}

  DIArray getElements() const {
    assert(!isSubroutineType() && "no elements for DISubroutineType");
    return getFieldAs<DIArray>(4);
  }

private:
  template <typename T>
  void setArrays(DITypedArray<T> Elements, DIArray TParams = DIArray()) {
    assert((!TParams || DbgNode->getNumOperands() == 8) &&
           "If you're setting the template parameters this should include a slot "
           "for that!");
    setArraysHelper(Elements, TParams);
  }

public:
  unsigned getRunTimeLang() const {
    return getHeaderFieldAs<unsigned>(7);
  }
  DITypeRef getContainingType() const { return getFieldAs<DITypeRef>(5); }

private:
  /// \brief Set the containing type.
  void setContainingType(DICompositeType ContainingType);

public:
  DIArray getTemplateParams() const { return getFieldAs<DIArray>(6); }
  MDString *getIdentifier() const;

  bool Verify() const;
};

class DISubroutineType : public DICompositeType {
public:
  explicit DISubroutineType(const MDNode *N = nullptr) : DICompositeType(N) {}
  DITypedArray<DITypeRef> getTypeArray() const {
    return getFieldAs<DITypedArray<DITypeRef>>(4);
  }
};

/// \brief This is a wrapper for a file.
class DIFile : public DIScope {
  friend class DIDescriptor;

public:
  explicit DIFile(const MDNode *N = nullptr) : DIScope(N) {}

  /// \brief Retrieve the MDNode for the directory/file pair.
  MDNode *getFileNode() const;
  bool Verify() const;
};

/// \brief A wrapper for a compile unit.
class DICompileUnit : public DIScope {
  friend class DIDescriptor;
  void printInternal(raw_ostream &OS) const;

public:
  explicit DICompileUnit(const MDNode *N = nullptr) : DIScope(N) {}

  dwarf::SourceLanguage getLanguage() const {
    return static_cast<dwarf::SourceLanguage>(getHeaderFieldAs<unsigned>(1));
  }
  StringRef getProducer() const { return getHeaderField(2); }

  bool isOptimized() const { return getHeaderFieldAs<bool>(3) != 0; }
  StringRef getFlags() const { return getHeaderField(4); }
  unsigned getRunTimeVersion() const { return getHeaderFieldAs<unsigned>(5); }

  DIArray getEnumTypes() const;
  DIArray getRetainedTypes() const;
  DIArray getSubprograms() const;
  DIArray getGlobalVariables() const;
  DIArray getImportedEntities() const;

  void replaceSubprograms(DIArray Subprograms);
  void replaceGlobalVariables(DIArray GlobalVariables);

  StringRef getSplitDebugFilename() const { return getHeaderField(6); }
  unsigned getEmissionKind() const { return getHeaderFieldAs<unsigned>(7); }

  bool Verify() const;
};

/// \brief This is a wrapper for a subprogram (e.g. a function).
class DISubprogram : public DIScope {
  friend class DIDescriptor;
  void printInternal(raw_ostream &OS) const;

public:
  explicit DISubprogram(const MDNode *N = nullptr) : DIScope(N) {}

  StringRef getName() const { return getHeaderField(1); }
  StringRef getDisplayName() const { return getHeaderField(2); }
  StringRef getLinkageName() const { return getHeaderField(3); }
  unsigned getLineNumber() const { return getHeaderFieldAs<unsigned>(4); }

  /// \brief Check if this is local (like 'static' in C).
  unsigned isLocalToUnit() const { return getHeaderFieldAs<unsigned>(5); }
  unsigned isDefinition() const { return getHeaderFieldAs<unsigned>(6); }

  unsigned getVirtuality() const { return getHeaderFieldAs<unsigned>(7); }
  unsigned getVirtualIndex() const { return getHeaderFieldAs<unsigned>(8); }

  unsigned getFlags() const { return getHeaderFieldAs<unsigned>(9); }

  unsigned isOptimized() const { return getHeaderFieldAs<bool>(10); }

  /// \brief Get the beginning of the scope of the function (not the name).
  unsigned getScopeLineNumber() const { return getHeaderFieldAs<unsigned>(11); }

  DIScopeRef getContext() const { return getFieldAs<DIScopeRef>(2); }
  DISubroutineType getType() const { return getFieldAs<DISubroutineType>(3); }

  DITypeRef getContainingType() const { return getFieldAs<DITypeRef>(4); }

  bool Verify() const;

  /// \brief Check if this provides debugging information for the function F.
  bool describes(const Function *F);

  Function *getFunction() const { return getFunctionField(5); }
  void replaceFunction(Function *F) { replaceFunctionField(5, F); }
  DIArray getTemplateParams() const { return getFieldAs<DIArray>(6); }
  DISubprogram getFunctionDeclaration() const {
    return getFieldAs<DISubprogram>(7);
  }
  MDNode *getVariablesNodes() const;
  DIArray getVariables() const;

  unsigned isArtificial() const { return (getFlags() & FlagArtificial) != 0; }
  /// \brief Check for the "private" access specifier.
  bool isPrivate() const {
    return (getFlags() & FlagAccessibility) == FlagPrivate;
  }
  /// \brief Check for the "protected" access specifier.
  bool isProtected() const {
    return (getFlags() & FlagAccessibility) == FlagProtected;
  }
  /// \brief Check for the "public" access specifier.
  bool isPublic() const {
    return (getFlags() & FlagAccessibility) == FlagPublic;
  }
  /// \brief Check for "explicit".
  bool isExplicit() const { return (getFlags() & FlagExplicit) != 0; }
  /// \brief Check if this is prototyped.
  bool isPrototyped() const { return (getFlags() & FlagPrototyped) != 0; }

  /// \brief Check if this is reference-qualified.
  ///
  /// Return true if this subprogram is a C++11 reference-qualified non-static
  /// member function (void foo() &).
  unsigned isLValueReference() const {
    return (getFlags() & FlagLValueReference) != 0;
  }

  /// \brief Check if this is rvalue-reference-qualified.
  ///
  /// Return true if this subprogram is a C++11 rvalue-reference-qualified
  /// non-static member function (void foo() &&).
  unsigned isRValueReference() const {
    return (getFlags() & FlagRValueReference) != 0;
  }

};

/// \brief This is a wrapper for a lexical block.
class DILexicalBlock : public DIScope {
public:
  explicit DILexicalBlock(const MDNode *N = nullptr) : DIScope(N) {}
  DIScope getContext() const { return getFieldAs<DIScope>(2); }
  unsigned getLineNumber() const {
    return getHeaderFieldAs<unsigned>(1);
  }
  unsigned getColumnNumber() const {
    return getHeaderFieldAs<unsigned>(2);
  }
  bool Verify() const;
};

/// \brief This is a wrapper for a lexical block with a filename change.
class DILexicalBlockFile : public DIScope {
public:
  explicit DILexicalBlockFile(const MDNode *N = nullptr) : DIScope(N) {}
  DIScope getContext() const {
    if (getScope().isSubprogram())
      return getScope();
    return getScope().getContext();
  }
  unsigned getLineNumber() const { return getScope().getLineNumber(); }
  unsigned getColumnNumber() const { return getScope().getColumnNumber(); }
  DILexicalBlock getScope() const { return getFieldAs<DILexicalBlock>(2); }
  unsigned getDiscriminator() const { return getHeaderFieldAs<unsigned>(1); }
  bool Verify() const;
};

/// \brief A wrapper for a C++ style name space.
class DINameSpace : public DIScope {
  friend class DIDescriptor;
  void printInternal(raw_ostream &OS) const;

public:
  explicit DINameSpace(const MDNode *N = nullptr) : DIScope(N) {}
  StringRef getName() const { return getHeaderField(1); }
  unsigned getLineNumber() const { return getHeaderFieldAs<unsigned>(2); }
  DIScope getContext() const { return getFieldAs<DIScope>(2); }
  bool Verify() const;
};

/// \brief This is a wrapper for template type parameter.
class DITemplateTypeParameter : public DIDescriptor {
public:
  explicit DITemplateTypeParameter(const MDNode *N = nullptr)
    : DIDescriptor(N) {}

  StringRef getName() const { return getHeaderField(1); }
  unsigned getLineNumber() const { return getHeaderFieldAs<unsigned>(2); }
  unsigned getColumnNumber() const { return getHeaderFieldAs<unsigned>(3); }

  DIScopeRef getContext() const { return getFieldAs<DIScopeRef>(1); }
  DITypeRef getType() const { return getFieldAs<DITypeRef>(2); }
  StringRef getFilename() const { return getFieldAs<DIFile>(3).getFilename(); }
  StringRef getDirectory() const {
    return getFieldAs<DIFile>(3).getDirectory();
  }
  bool Verify() const;
};

/// \brief This is a wrapper for template value parameter.
class DITemplateValueParameter : public DIDescriptor {
public:
  explicit DITemplateValueParameter(const MDNode *N = nullptr)
    : DIDescriptor(N) {}

  StringRef getName() const { return getHeaderField(1); }
  unsigned getLineNumber() const { return getHeaderFieldAs<unsigned>(2); }
  unsigned getColumnNumber() const { return getHeaderFieldAs<unsigned>(3); }

  DIScopeRef getContext() const { return getFieldAs<DIScopeRef>(1); }
  DITypeRef getType() const { return getFieldAs<DITypeRef>(2); }
  Metadata *getValue() const;
  StringRef getFilename() const { return getFieldAs<DIFile>(4).getFilename(); }
  StringRef getDirectory() const {
    return getFieldAs<DIFile>(4).getDirectory();
  }
  bool Verify() const;
};

/// \brief This is a wrapper for a global variable.
class DIGlobalVariable : public DIDescriptor {
  friend class DIDescriptor;
  void printInternal(raw_ostream &OS) const;

public:
  explicit DIGlobalVariable(const MDNode *N = nullptr) : DIDescriptor(N) {}

  StringRef getName() const { return getHeaderField(1); }
  StringRef getDisplayName() const { return getHeaderField(2); }
  StringRef getLinkageName() const { return getHeaderField(3); }
  unsigned getLineNumber() const { return getHeaderFieldAs<unsigned>(4); }
  unsigned isLocalToUnit() const { return getHeaderFieldAs<bool>(5); }
  unsigned isDefinition() const { return getHeaderFieldAs<bool>(6); }

  DIScope getContext() const { return getFieldAs<DIScope>(1); }
  StringRef getFilename() const { return getFieldAs<DIFile>(2).getFilename(); }
  StringRef getDirectory() const {
    return getFieldAs<DIFile>(2).getDirectory();
  }
  DITypeRef getType() const { return getFieldAs<DITypeRef>(3); }

  GlobalVariable *getGlobal() const { return getGlobalVariableField(4); }
  Constant *getConstant() const { return getConstantField(4); }
  DIDerivedType getStaticDataMemberDeclaration() const {
    return getFieldAs<DIDerivedType>(5);
  }

  bool Verify() const;
};

/// \brief This is a wrapper for a variable (e.g. parameter, local, global etc).
class DIVariable : public DIDescriptor {
  friend class DIDescriptor;
  void printInternal(raw_ostream &OS) const;

public:
  explicit DIVariable(const MDNode *N = nullptr) : DIDescriptor(N) {}

  StringRef getName() const { return getHeaderField(1); }
  unsigned getLineNumber() const {
    // FIXME: Line number and arg number shouldn't be merged together like this.
    return (getHeaderFieldAs<unsigned>(2) << 8) >> 8;
  }
  unsigned getArgNumber() const { return getHeaderFieldAs<unsigned>(2) >> 24; }

  DIScope getContext() const { return getFieldAs<DIScope>(1); }
  DIFile getFile() const { return getFieldAs<DIFile>(2); }
  DITypeRef getType() const { return getFieldAs<DITypeRef>(3); }

  /// \brief Return true if this variable is marked as "artificial".
  bool isArtificial() const {
    return (getHeaderFieldAs<unsigned>(3) & FlagArtificial) != 0;
  }

  bool isObjectPointer() const {
    return (getHeaderFieldAs<unsigned>(3) & FlagObjectPointer) != 0;
  }

  /// \brief Return true if this variable is represented as a pointer.
  bool isIndirect() const {
    return (getHeaderFieldAs<unsigned>(3) & FlagIndirectVariable) != 0;
  }

  /// \brief If this variable is inlined then return inline location.
  MDNode *getInlinedAt() const;

  bool Verify() const;

  /// \brief Check if this is a "__block" variable (Apple Blocks).
  bool isBlockByrefVariable(const DITypeIdentifierMap &Map) const {
    return (getType().resolve(Map)).isBlockByrefStruct();
  }

  /// \brief Check if this is an inlined function argument.
  bool isInlinedFnArgument(const Function *CurFn);

  /// \brief Return the size reported by the variable's type.
  unsigned getSizeInBits(const DITypeIdentifierMap &Map);

  void printExtendedName(raw_ostream &OS) const;
};

/// \brief A complex location expression.
class DIExpression : public DIDescriptor {
  friend class DIDescriptor;
  void printInternal(raw_ostream &OS) const;

public:
  explicit DIExpression(const MDNode *N = nullptr) : DIDescriptor(N) {}

  bool Verify() const;

  /// \brief Return the number of elements in the complex expression.
  unsigned getNumElements() const {
    if (!DbgNode)
      return 0;
    unsigned N = getNumHeaderFields();
    assert(N > 0 && "missing tag");
    return N - 1;
  }

  /// \brief return the Idx'th complex address element.
  uint64_t getElement(unsigned Idx) const;

  /// \brief Return whether this is a piece of an aggregate variable.
  bool isVariablePiece() const;
  /// \brief Return the offset of this piece in bytes.
  uint64_t getPieceOffset() const;
  /// \brief Return the size of this piece in bytes.
  uint64_t getPieceSize() const;
};

/// \brief This object holds location information.
///
/// This object is not associated with any DWARF tag.
class DILocation : public DIDescriptor {
public:
  explicit DILocation(const MDNode *N) : DIDescriptor(N) {}

  unsigned getLineNumber() const {
    if (auto *L = dyn_cast_or_null<MDLocation>(DbgNode))
      return L->getLine();
    return 0;
  }
  unsigned getColumnNumber() const {
    if (auto *L = dyn_cast_or_null<MDLocation>(DbgNode))
      return L->getColumn();
    return 0;
  }
  DIScope getScope() const {
    if (auto *L = dyn_cast_or_null<MDLocation>(DbgNode))
      return DIScope(dyn_cast_or_null<MDNode>(L->getScope()));
    return DIScope(nullptr);
  }
  DILocation getOrigLocation() const {
    if (auto *L = dyn_cast_or_null<MDLocation>(DbgNode))
      return DILocation(dyn_cast_or_null<MDNode>(L->getInlinedAt()));
    return DILocation(nullptr);
  }
  StringRef getFilename() const { return getScope().getFilename(); }
  StringRef getDirectory() const { return getScope().getDirectory(); }
  bool Verify() const;
  bool atSameLineAs(const DILocation &Other) const {
    return (getLineNumber() == Other.getLineNumber() &&
            getFilename() == Other.getFilename());
  }
  /// \brief Get the DWAF discriminator.
  ///
  /// DWARF discriminators are used to distinguish identical file locations for
  /// instructions that are on different basic blocks. If two instructions are
  /// inside the same lexical block and are in different basic blocks, we
  /// create a new lexical block with identical location as the original but
  /// with a different discriminator value
  /// (lib/Transforms/Util/AddDiscriminators.cpp for details).
  unsigned getDiscriminator() const {
    // Since discriminators are associated with lexical blocks, make
    // sure this location is a lexical block before retrieving its
    // value.
    return getScope().isLexicalBlockFile()
               ? DILexicalBlockFile(
                     cast<MDNode>(cast<MDLocation>(DbgNode)->getScope()))
                     .getDiscriminator()
               : 0;
  }

  /// \brief Generate a new discriminator value for this location.
  unsigned computeNewDiscriminator(LLVMContext &Ctx);

  /// \brief Return a copy of this location with a different scope.
  DILocation copyWithNewScope(LLVMContext &Ctx, DILexicalBlockFile NewScope);
};

class DIObjCProperty : public DIDescriptor {
  friend class DIDescriptor;
  void printInternal(raw_ostream &OS) const;

public:
  explicit DIObjCProperty(const MDNode *N) : DIDescriptor(N) {}

  StringRef getObjCPropertyName() const { return getHeaderField(1); }
  DIFile getFile() const { return getFieldAs<DIFile>(1); }
  unsigned getLineNumber() const { return getHeaderFieldAs<unsigned>(2); }

  StringRef getObjCPropertyGetterName() const { return getHeaderField(3); }
  StringRef getObjCPropertySetterName() const { return getHeaderField(4); }
  unsigned getAttributes() const { return getHeaderFieldAs<unsigned>(5); }
  bool isReadOnlyObjCProperty() const {
    return (getAttributes() & dwarf::DW_APPLE_PROPERTY_readonly) != 0;
  }
  bool isReadWriteObjCProperty() const {
    return (getAttributes() & dwarf::DW_APPLE_PROPERTY_readwrite) != 0;
  }
  bool isAssignObjCProperty() const {
    return (getAttributes() & dwarf::DW_APPLE_PROPERTY_assign) != 0;
  }
  bool isRetainObjCProperty() const {
    return (getAttributes() & dwarf::DW_APPLE_PROPERTY_retain) != 0;
  }
  bool isCopyObjCProperty() const {
    return (getAttributes() & dwarf::DW_APPLE_PROPERTY_copy) != 0;
  }
  bool isNonAtomicObjCProperty() const {
    return (getAttributes() & dwarf::DW_APPLE_PROPERTY_nonatomic) != 0;
  }

  /// \brief Get the type.
  ///
  /// \note Objective-C doesn't have an ODR, so there is no benefit in storing
  /// the type as a DITypeRef here.
  DIType getType() const { return getFieldAs<DIType>(2); }

  bool Verify() const;
};

/// \brief An imported module (C++ using directive or similar).
class DIImportedEntity : public DIDescriptor {
  friend class DIDescriptor;
  void printInternal(raw_ostream &OS) const;

public:
  explicit DIImportedEntity(const MDNode *N) : DIDescriptor(N) {}
  DIScope getContext() const { return getFieldAs<DIScope>(1); }
  DIScopeRef getEntity() const { return getFieldAs<DIScopeRef>(2); }
  unsigned getLineNumber() const { return getHeaderFieldAs<unsigned>(1); }
  StringRef getName() const { return getHeaderField(2); }
  bool Verify() const;
};

/// \brief Find subprogram that is enclosing this scope.
DISubprogram getDISubprogram(const MDNode *Scope);

/// \brief Find debug info for a given function.
/// \returns a valid DISubprogram, if found. Otherwise, it returns an empty
/// DISubprogram.
DISubprogram getDISubprogram(const Function *F);

/// \brief Find underlying composite type.
DICompositeType getDICompositeType(DIType T);

/// \brief Create a new inlined variable based on current variable.
///
/// @param DV            Current Variable.
/// @param InlinedScope  Location at current variable is inlined.
DIVariable createInlinedVariable(MDNode *DV, MDNode *InlinedScope,
                                 LLVMContext &VMContext);

/// \brief Remove inlined scope from the variable.
DIVariable cleanseInlinedVariable(MDNode *DV, LLVMContext &VMContext);

/// \brief Generate map by visiting all retained types.
DITypeIdentifierMap generateDITypeIdentifierMap(const NamedMDNode *CU_Nodes);

/// \brief Strip debug info in the module if it exists.
///
/// To do this, we remove all calls to the debugger intrinsics and any named
/// metadata for debugging. We also remove debug locations for instructions.
/// Return true if module is modified.
bool StripDebugInfo(Module &M);

/// \brief Return Debug Info Metadata Version by checking module flags.
unsigned getDebugMetadataVersionFromModule(const Module &M);

/// \brief Utility to find all debug info in a module.
///
/// DebugInfoFinder tries to list all debug info MDNodes used in a module. To
/// list debug info MDNodes used by an instruction, DebugInfoFinder uses
/// processDeclare, processValue and processLocation to handle DbgDeclareInst,
/// DbgValueInst and DbgLoc attached to instructions. processModule will go
/// through all DICompileUnits in llvm.dbg.cu and list debug info MDNodes
/// used by the CUs.
class DebugInfoFinder {
public:
  DebugInfoFinder() : TypeMapInitialized(false) {}

  /// \brief Process entire module and collect debug info anchors.
  void processModule(const Module &M);

  /// \brief Process DbgDeclareInst.
  void processDeclare(const Module &M, const DbgDeclareInst *DDI);
  /// \brief Process DbgValueInst.
  void processValue(const Module &M, const DbgValueInst *DVI);
  /// \brief Process DILocation.
  void processLocation(const Module &M, DILocation Loc);

  /// \brief Clear all lists.
  void reset();

private:
  void InitializeTypeMap(const Module &M);

  void processType(DIType DT);
  void processSubprogram(DISubprogram SP);
  void processScope(DIScope Scope);
  bool addCompileUnit(DICompileUnit CU);
  bool addGlobalVariable(DIGlobalVariable DIG);
  bool addSubprogram(DISubprogram SP);
  bool addType(DIType DT);
  bool addScope(DIScope Scope);

public:
  typedef SmallVectorImpl<DICompileUnit>::const_iterator compile_unit_iterator;
  typedef SmallVectorImpl<DISubprogram>::const_iterator subprogram_iterator;
  typedef SmallVectorImpl<DIGlobalVariable>::const_iterator global_variable_iterator;
  typedef SmallVectorImpl<DIType>::const_iterator type_iterator;
  typedef SmallVectorImpl<DIScope>::const_iterator scope_iterator;

  iterator_range<compile_unit_iterator> compile_units() const {
    return iterator_range<compile_unit_iterator>(CUs.begin(), CUs.end());
  }

  iterator_range<subprogram_iterator> subprograms() const {
    return iterator_range<subprogram_iterator>(SPs.begin(), SPs.end());
  }

  iterator_range<global_variable_iterator> global_variables() const {
    return iterator_range<global_variable_iterator>(GVs.begin(), GVs.end());
  }

  iterator_range<type_iterator> types() const {
    return iterator_range<type_iterator>(TYs.begin(), TYs.end());
  }

  iterator_range<scope_iterator> scopes() const {
    return iterator_range<scope_iterator>(Scopes.begin(), Scopes.end());
  }

  unsigned compile_unit_count() const { return CUs.size(); }
  unsigned global_variable_count() const { return GVs.size(); }
  unsigned subprogram_count() const { return SPs.size(); }
  unsigned type_count() const { return TYs.size(); }
  unsigned scope_count() const { return Scopes.size(); }

private:
  SmallVector<DICompileUnit, 8> CUs;
  SmallVector<DISubprogram, 8> SPs;
  SmallVector<DIGlobalVariable, 8> GVs;
  SmallVector<DIType, 8> TYs;
  SmallVector<DIScope, 8> Scopes;
  SmallPtrSet<MDNode *, 64> NodesSeen;
  DITypeIdentifierMap TypeIdentifierMap;

  /// \brief Specify if TypeIdentifierMap is initialized.
  bool TypeMapInitialized;
};

DenseMap<const Function *, DISubprogram> makeSubprogramMap(const Module &M);

} // end namespace llvm

#endif
