//===- llvm/IR/DebugInfoMetadata.h - Debug info metadata --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Declarations for metadata specific to debug info.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DEBUGINFOMETADATA_H
#define LLVM_IR_DEBUGINFOMETADATA_H

#include "llvm/IR/Metadata.h"
#include "llvm/Support/Dwarf.h"

// Helper macros for defining get() overrides.
#define DEFINE_MDNODE_GET_UNPACK_IMPL(...) __VA_ARGS__
#define DEFINE_MDNODE_GET_UNPACK(ARGS) DEFINE_MDNODE_GET_UNPACK_IMPL ARGS
#define DEFINE_MDNODE_GET_DISTINCT_TEMPORARY(CLASS, FORMAL, ARGS)              \
  static CLASS *getDistinct(LLVMContext &Context,                              \
                            DEFINE_MDNODE_GET_UNPACK(FORMAL)) {                \
    return getImpl(Context, DEFINE_MDNODE_GET_UNPACK(ARGS), Distinct);         \
  }                                                                            \
  static Temp##CLASS getTemporary(LLVMContext &Context,                        \
                                  DEFINE_MDNODE_GET_UNPACK(FORMAL)) {          \
    return Temp##CLASS(                                                        \
        getImpl(Context, DEFINE_MDNODE_GET_UNPACK(ARGS), Temporary));          \
  }
#define DEFINE_MDNODE_GET(CLASS, FORMAL, ARGS)                                 \
  static CLASS *get(LLVMContext &Context, DEFINE_MDNODE_GET_UNPACK(FORMAL)) {  \
    return getImpl(Context, DEFINE_MDNODE_GET_UNPACK(ARGS), Uniqued);          \
  }                                                                            \
  static CLASS *getIfExists(LLVMContext &Context,                              \
                            DEFINE_MDNODE_GET_UNPACK(FORMAL)) {                \
    return getImpl(Context, DEFINE_MDNODE_GET_UNPACK(ARGS), Uniqued,           \
                   /* ShouldCreate */ false);                                  \
  }                                                                            \
  DEFINE_MDNODE_GET_DISTINCT_TEMPORARY(CLASS, FORMAL, ARGS)

namespace llvm {

template <typename T> class Optional;

/// Holds a subclass of DINode.
///
/// FIXME: This class doesn't currently make much sense.  Previously it was a
/// union beteen MDString (for ODR-uniqued types) and things like DIType.  To
/// support CodeView work, it wasn't deleted outright when MDString-based type
/// references were deleted; we'll soon need a similar concept for CodeView
/// DITypeIndex.
template <class T> class TypedDINodeRef {
  const Metadata *MD = nullptr;

public:
  TypedDINodeRef() = default;
  TypedDINodeRef(std::nullptr_t) {}
  TypedDINodeRef(const T *MD) : MD(MD) {}

  explicit TypedDINodeRef(const Metadata *MD) : MD(MD) {
    assert((!MD || isa<T>(MD)) && "Expected valid type ref");
  }

  template <class U>
  TypedDINodeRef(
      const TypedDINodeRef<U> &X,
      typename std::enable_if<std::is_convertible<U *, T *>::value>::type * =
          nullptr)
      : MD(X) {}

  operator Metadata *() const { return const_cast<Metadata *>(MD); }

  T *resolve() const { return const_cast<T *>(cast_or_null<T>(MD)); }

  bool operator==(const TypedDINodeRef<T> &X) const { return MD == X.MD; }
  bool operator!=(const TypedDINodeRef<T> &X) const { return MD != X.MD; }
};

typedef TypedDINodeRef<DINode> DINodeRef;
typedef TypedDINodeRef<DIScope> DIScopeRef;
typedef TypedDINodeRef<DIType> DITypeRef;

class DITypeRefArray {
  const MDTuple *N = nullptr;

public:
  DITypeRefArray() = default;
  DITypeRefArray(const MDTuple *N) : N(N) {}

  explicit operator bool() const { return get(); }
  explicit operator MDTuple *() const { return get(); }

  MDTuple *get() const { return const_cast<MDTuple *>(N); }
  MDTuple *operator->() const { return get(); }
  MDTuple &operator*() const { return *get(); }

  // FIXME: Fix callers and remove condition on N.
  unsigned size() const { return N ? N->getNumOperands() : 0u; }
  DITypeRef operator[](unsigned I) const { return DITypeRef(N->getOperand(I)); }

  class iterator : std::iterator<std::input_iterator_tag, DITypeRef,
                                 std::ptrdiff_t, void, DITypeRef> {
    MDNode::op_iterator I = nullptr;

  public:
    iterator() = default;
    explicit iterator(MDNode::op_iterator I) : I(I) {}
    DITypeRef operator*() const { return DITypeRef(*I); }
    iterator &operator++() {
      ++I;
      return *this;
    }
    iterator operator++(int) {
      iterator Temp(*this);
      ++I;
      return Temp;
    }
    bool operator==(const iterator &X) const { return I == X.I; }
    bool operator!=(const iterator &X) const { return I != X.I; }
  };

  // FIXME: Fix callers and remove condition on N.
  iterator begin() const { return N ? iterator(N->op_begin()) : iterator(); }
  iterator end() const { return N ? iterator(N->op_end()) : iterator(); }
};

/// \brief Tagged DWARF-like metadata node.
///
/// A metadata node with a DWARF tag (i.e., a constant named \c DW_TAG_*,
/// defined in llvm/Support/Dwarf.h).  Called \a DINode because it's
/// potentially used for non-DWARF output.
class DINode : public MDNode {
  friend class LLVMContextImpl;
  friend class MDNode;

protected:
  DINode(LLVMContext &C, unsigned ID, StorageType Storage, unsigned Tag,
         ArrayRef<Metadata *> Ops1, ArrayRef<Metadata *> Ops2 = None)
      : MDNode(C, ID, Storage, Ops1, Ops2) {
    assert(Tag < 1u << 16);
    SubclassData16 = Tag;
  }
  ~DINode() = default;

  template <class Ty> Ty *getOperandAs(unsigned I) const {
    return cast_or_null<Ty>(getOperand(I));
  }

  StringRef getStringOperand(unsigned I) const {
    if (auto *S = getOperandAs<MDString>(I))
      return S->getString();
    return StringRef();
  }

  static MDString *getCanonicalMDString(LLVMContext &Context, StringRef S) {
    if (S.empty())
      return nullptr;
    return MDString::get(Context, S);
  }

  /// Allow subclasses to mutate the tag.
  void setTag(unsigned Tag) { SubclassData16 = Tag; }

public:
  unsigned getTag() const { return SubclassData16; }

  /// \brief Debug info flags.
  ///
  /// The three accessibility flags are mutually exclusive and rolled together
  /// in the first two bits.
  enum DIFlags {
#define HANDLE_DI_FLAG(ID, NAME) Flag##NAME = ID,
#include "llvm/IR/DebugInfoFlags.def"
    FlagAccessibility = FlagPrivate | FlagProtected | FlagPublic,
    FlagPtrToMemberRep = FlagSingleInheritance | FlagMultipleInheritance |
                         FlagVirtualInheritance,
  };

  static unsigned getFlag(StringRef Flag);
  static const char *getFlagString(unsigned Flag);

  /// \brief Split up a flags bitfield.
  ///
  /// Split \c Flags into \c SplitFlags, a vector of its components.  Returns
  /// any remaining (unrecognized) bits.
  static unsigned splitFlags(unsigned Flags,
                             SmallVectorImpl<unsigned> &SplitFlags);

  static bool classof(const Metadata *MD) {
    switch (MD->getMetadataID()) {
    default:
      return false;
    case GenericDINodeKind:
    case DISubrangeKind:
    case DIEnumeratorKind:
    case DIBasicTypeKind:
    case DIDerivedTypeKind:
    case DICompositeTypeKind:
    case DISubroutineTypeKind:
    case DIFileKind:
    case DICompileUnitKind:
    case DISubprogramKind:
    case DILexicalBlockKind:
    case DILexicalBlockFileKind:
    case DINamespaceKind:
    case DITemplateTypeParameterKind:
    case DITemplateValueParameterKind:
    case DIGlobalVariableKind:
    case DILocalVariableKind:
    case DIObjCPropertyKind:
    case DIImportedEntityKind:
    case DIModuleKind:
      return true;
    }
  }
};

template <class T> struct simplify_type<const TypedDINodeRef<T>> {
  typedef Metadata *SimpleType;
  static SimpleType getSimplifiedValue(const TypedDINodeRef<T> &MD) {
    return MD;
  }
};

template <class T>
struct simplify_type<TypedDINodeRef<T>>
    : simplify_type<const TypedDINodeRef<T>> {};

/// \brief Generic tagged DWARF-like metadata node.
///
/// An un-specialized DWARF-like metadata node.  The first operand is a
/// (possibly empty) null-separated \a MDString header that contains arbitrary
/// fields.  The remaining operands are \a dwarf_operands(), and are pointers
/// to other metadata.
class GenericDINode : public DINode {
  friend class LLVMContextImpl;
  friend class MDNode;

  GenericDINode(LLVMContext &C, StorageType Storage, unsigned Hash,
                unsigned Tag, ArrayRef<Metadata *> Ops1,
                ArrayRef<Metadata *> Ops2)
      : DINode(C, GenericDINodeKind, Storage, Tag, Ops1, Ops2) {
    setHash(Hash);
  }
  ~GenericDINode() { dropAllReferences(); }

  void setHash(unsigned Hash) { SubclassData32 = Hash; }
  void recalculateHash();

  static GenericDINode *getImpl(LLVMContext &Context, unsigned Tag,
                                StringRef Header, ArrayRef<Metadata *> DwarfOps,
                                StorageType Storage, bool ShouldCreate = true) {
    return getImpl(Context, Tag, getCanonicalMDString(Context, Header),
                   DwarfOps, Storage, ShouldCreate);
  }

  static GenericDINode *getImpl(LLVMContext &Context, unsigned Tag,
                                MDString *Header, ArrayRef<Metadata *> DwarfOps,
                                StorageType Storage, bool ShouldCreate = true);

  TempGenericDINode cloneImpl() const {
    return getTemporary(
        getContext(), getTag(), getHeader(),
        SmallVector<Metadata *, 4>(dwarf_op_begin(), dwarf_op_end()));
  }

public:
  unsigned getHash() const { return SubclassData32; }

  DEFINE_MDNODE_GET(GenericDINode, (unsigned Tag, StringRef Header,
                                    ArrayRef<Metadata *> DwarfOps),
                    (Tag, Header, DwarfOps))
  DEFINE_MDNODE_GET(GenericDINode, (unsigned Tag, MDString *Header,
                                    ArrayRef<Metadata *> DwarfOps),
                    (Tag, Header, DwarfOps))

  /// \brief Return a (temporary) clone of this.
  TempGenericDINode clone() const { return cloneImpl(); }

  unsigned getTag() const { return SubclassData16; }
  StringRef getHeader() const { return getStringOperand(0); }
  MDString *getRawHeader() const { return getOperandAs<MDString>(0); }

  op_iterator dwarf_op_begin() const { return op_begin() + 1; }
  op_iterator dwarf_op_end() const { return op_end(); }
  op_range dwarf_operands() const {
    return op_range(dwarf_op_begin(), dwarf_op_end());
  }

  unsigned getNumDwarfOperands() const { return getNumOperands() - 1; }
  const MDOperand &getDwarfOperand(unsigned I) const {
    return getOperand(I + 1);
  }
  void replaceDwarfOperandWith(unsigned I, Metadata *New) {
    replaceOperandWith(I + 1, New);
  }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == GenericDINodeKind;
  }
};

/// \brief Array subrange.
///
/// TODO: Merge into node for DW_TAG_array_type, which should have a custom
/// type.
class DISubrange : public DINode {
  friend class LLVMContextImpl;
  friend class MDNode;

  int64_t Count;
  int64_t LowerBound;

  DISubrange(LLVMContext &C, StorageType Storage, int64_t Count,
             int64_t LowerBound)
      : DINode(C, DISubrangeKind, Storage, dwarf::DW_TAG_subrange_type, None),
        Count(Count), LowerBound(LowerBound) {}
  ~DISubrange() = default;

  static DISubrange *getImpl(LLVMContext &Context, int64_t Count,
                             int64_t LowerBound, StorageType Storage,
                             bool ShouldCreate = true);

  TempDISubrange cloneImpl() const {
    return getTemporary(getContext(), getCount(), getLowerBound());
  }

public:
  DEFINE_MDNODE_GET(DISubrange, (int64_t Count, int64_t LowerBound = 0),
                    (Count, LowerBound))

  TempDISubrange clone() const { return cloneImpl(); }

  int64_t getLowerBound() const { return LowerBound; }
  int64_t getCount() const { return Count; }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DISubrangeKind;
  }
};

/// \brief Enumeration value.
///
/// TODO: Add a pointer to the context (DW_TAG_enumeration_type) once that no
/// longer creates a type cycle.
class DIEnumerator : public DINode {
  friend class LLVMContextImpl;
  friend class MDNode;

  int64_t Value;

  DIEnumerator(LLVMContext &C, StorageType Storage, int64_t Value,
               ArrayRef<Metadata *> Ops)
      : DINode(C, DIEnumeratorKind, Storage, dwarf::DW_TAG_enumerator, Ops),
        Value(Value) {}
  ~DIEnumerator() = default;

  static DIEnumerator *getImpl(LLVMContext &Context, int64_t Value,
                               StringRef Name, StorageType Storage,
                               bool ShouldCreate = true) {
    return getImpl(Context, Value, getCanonicalMDString(Context, Name), Storage,
                   ShouldCreate);
  }
  static DIEnumerator *getImpl(LLVMContext &Context, int64_t Value,
                               MDString *Name, StorageType Storage,
                               bool ShouldCreate = true);

  TempDIEnumerator cloneImpl() const {
    return getTemporary(getContext(), getValue(), getName());
  }

public:
  DEFINE_MDNODE_GET(DIEnumerator, (int64_t Value, StringRef Name),
                    (Value, Name))
  DEFINE_MDNODE_GET(DIEnumerator, (int64_t Value, MDString *Name),
                    (Value, Name))

  TempDIEnumerator clone() const { return cloneImpl(); }

  int64_t getValue() const { return Value; }
  StringRef getName() const { return getStringOperand(0); }

  MDString *getRawName() const { return getOperandAs<MDString>(0); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DIEnumeratorKind;
  }
};

/// \brief Base class for scope-like contexts.
///
/// Base class for lexical scopes and types (which are also declaration
/// contexts).
///
/// TODO: Separate the concepts of declaration contexts and lexical scopes.
class DIScope : public DINode {
protected:
  DIScope(LLVMContext &C, unsigned ID, StorageType Storage, unsigned Tag,
          ArrayRef<Metadata *> Ops)
      : DINode(C, ID, Storage, Tag, Ops) {}
  ~DIScope() = default;

public:
  DIFile *getFile() const { return cast_or_null<DIFile>(getRawFile()); }

  inline StringRef getFilename() const;
  inline StringRef getDirectory() const;

  StringRef getName() const;
  DIScopeRef getScope() const;

  /// \brief Return the raw underlying file.
  ///
  /// An \a DIFile is an \a DIScope, but it doesn't point at a separate file
  /// (it\em is the file).  If \c this is an \a DIFile, we need to return \c
  /// this.  Otherwise, return the first operand, which is where all other
  /// subclasses store their file pointer.
  Metadata *getRawFile() const {
    return isa<DIFile>(this) ? const_cast<DIScope *>(this)
                             : static_cast<Metadata *>(getOperand(0));
  }

  static bool classof(const Metadata *MD) {
    switch (MD->getMetadataID()) {
    default:
      return false;
    case DIBasicTypeKind:
    case DIDerivedTypeKind:
    case DICompositeTypeKind:
    case DISubroutineTypeKind:
    case DIFileKind:
    case DICompileUnitKind:
    case DISubprogramKind:
    case DILexicalBlockKind:
    case DILexicalBlockFileKind:
    case DINamespaceKind:
    case DIModuleKind:
      return true;
    }
  }
};

/// \brief File.
///
/// TODO: Merge with directory/file node (including users).
/// TODO: Canonicalize paths on creation.
class DIFile : public DIScope {
  friend class LLVMContextImpl;
  friend class MDNode;

  DIFile(LLVMContext &C, StorageType Storage, ArrayRef<Metadata *> Ops)
      : DIScope(C, DIFileKind, Storage, dwarf::DW_TAG_file_type, Ops) {}
  ~DIFile() = default;

  static DIFile *getImpl(LLVMContext &Context, StringRef Filename,
                         StringRef Directory, StorageType Storage,
                         bool ShouldCreate = true) {
    return getImpl(Context, getCanonicalMDString(Context, Filename),
                   getCanonicalMDString(Context, Directory), Storage,
                   ShouldCreate);
  }
  static DIFile *getImpl(LLVMContext &Context, MDString *Filename,
                         MDString *Directory, StorageType Storage,
                         bool ShouldCreate = true);

  TempDIFile cloneImpl() const {
    return getTemporary(getContext(), getFilename(), getDirectory());
  }

public:
  DEFINE_MDNODE_GET(DIFile, (StringRef Filename, StringRef Directory),
                    (Filename, Directory))
  DEFINE_MDNODE_GET(DIFile, (MDString * Filename, MDString *Directory),
                    (Filename, Directory))

  TempDIFile clone() const { return cloneImpl(); }

  StringRef getFilename() const { return getStringOperand(0); }
  StringRef getDirectory() const { return getStringOperand(1); }

  MDString *getRawFilename() const { return getOperandAs<MDString>(0); }
  MDString *getRawDirectory() const { return getOperandAs<MDString>(1); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DIFileKind;
  }
};

StringRef DIScope::getFilename() const {
  if (auto *F = getFile())
    return F->getFilename();
  return "";
}

StringRef DIScope::getDirectory() const {
  if (auto *F = getFile())
    return F->getDirectory();
  return "";
}

/// \brief Base class for types.
///
/// TODO: Remove the hardcoded name and context, since many types don't use
/// them.
/// TODO: Split up flags.
class DIType : public DIScope {
  unsigned Line;
  unsigned Flags;
  uint64_t SizeInBits;
  uint64_t AlignInBits;
  uint64_t OffsetInBits;

protected:
  DIType(LLVMContext &C, unsigned ID, StorageType Storage, unsigned Tag,
         unsigned Line, uint64_t SizeInBits, uint64_t AlignInBits,
         uint64_t OffsetInBits, unsigned Flags, ArrayRef<Metadata *> Ops)
      : DIScope(C, ID, Storage, Tag, Ops) {
    init(Line, SizeInBits, AlignInBits, OffsetInBits, Flags);
  }
  ~DIType() = default;

  void init(unsigned Line, uint64_t SizeInBits, uint64_t AlignInBits,
            uint64_t OffsetInBits, unsigned Flags) {
    this->Line = Line;
    this->Flags = Flags;
    this->SizeInBits = SizeInBits;
    this->AlignInBits = AlignInBits;
    this->OffsetInBits = OffsetInBits;
  }

  /// Change fields in place.
  void mutate(unsigned Tag, unsigned Line, uint64_t SizeInBits,
              uint64_t AlignInBits, uint64_t OffsetInBits, unsigned Flags) {
    assert(isDistinct() && "Only distinct nodes can mutate");
    setTag(Tag);
    init(Line, SizeInBits, AlignInBits, OffsetInBits, Flags);
  }

public:
  TempDIType clone() const {
    return TempDIType(cast<DIType>(MDNode::clone().release()));
  }

  unsigned getLine() const { return Line; }
  uint64_t getSizeInBits() const { return SizeInBits; }
  uint64_t getAlignInBits() const { return AlignInBits; }
  uint64_t getOffsetInBits() const { return OffsetInBits; }
  unsigned getFlags() const { return Flags; }

  DIScopeRef getScope() const { return DIScopeRef(getRawScope()); }
  StringRef getName() const { return getStringOperand(2); }


  Metadata *getRawScope() const { return getOperand(1); }
  MDString *getRawName() const { return getOperandAs<MDString>(2); }

  void setFlags(unsigned NewFlags) {
    assert(!isUniqued() && "Cannot set flags on uniqued nodes");
    Flags = NewFlags;
  }

  bool isPrivate() const {
    return (getFlags() & FlagAccessibility) == FlagPrivate;
  }
  bool isProtected() const {
    return (getFlags() & FlagAccessibility) == FlagProtected;
  }
  bool isPublic() const {
    return (getFlags() & FlagAccessibility) == FlagPublic;
  }
  bool isForwardDecl() const { return getFlags() & FlagFwdDecl; }
  bool isAppleBlockExtension() const { return getFlags() & FlagAppleBlock; }
  bool isBlockByrefStruct() const { return getFlags() & FlagBlockByrefStruct; }
  bool isVirtual() const { return getFlags() & FlagVirtual; }
  bool isArtificial() const { return getFlags() & FlagArtificial; }
  bool isObjectPointer() const { return getFlags() & FlagObjectPointer; }
  bool isObjcClassComplete() const {
    return getFlags() & FlagObjcClassComplete;
  }
  bool isVector() const { return getFlags() & FlagVector; }
  bool isBitField() const { return getFlags() & FlagBitField; }
  bool isStaticMember() const { return getFlags() & FlagStaticMember; }
  bool isLValueReference() const { return getFlags() & FlagLValueReference; }
  bool isRValueReference() const { return getFlags() & FlagRValueReference; }
  bool isExternalTypeRef() const { return getFlags() & FlagExternalTypeRef; }

  static bool classof(const Metadata *MD) {
    switch (MD->getMetadataID()) {
    default:
      return false;
    case DIBasicTypeKind:
    case DIDerivedTypeKind:
    case DICompositeTypeKind:
    case DISubroutineTypeKind:
      return true;
    }
  }
};

/// \brief Basic type, like 'int' or 'float'.
///
/// TODO: Split out DW_TAG_unspecified_type.
/// TODO: Drop unused accessors.
class DIBasicType : public DIType {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Encoding;

  DIBasicType(LLVMContext &C, StorageType Storage, unsigned Tag,
              uint64_t SizeInBits, uint64_t AlignInBits, unsigned Encoding,
              ArrayRef<Metadata *> Ops)
      : DIType(C, DIBasicTypeKind, Storage, Tag, 0, SizeInBits, AlignInBits, 0,
               0, Ops),
        Encoding(Encoding) {}
  ~DIBasicType() = default;

  static DIBasicType *getImpl(LLVMContext &Context, unsigned Tag,
                              StringRef Name, uint64_t SizeInBits,
                              uint64_t AlignInBits, unsigned Encoding,
                              StorageType Storage, bool ShouldCreate = true) {
    return getImpl(Context, Tag, getCanonicalMDString(Context, Name),
                   SizeInBits, AlignInBits, Encoding, Storage, ShouldCreate);
  }
  static DIBasicType *getImpl(LLVMContext &Context, unsigned Tag,
                              MDString *Name, uint64_t SizeInBits,
                              uint64_t AlignInBits, unsigned Encoding,
                              StorageType Storage, bool ShouldCreate = true);

  TempDIBasicType cloneImpl() const {
    return getTemporary(getContext(), getTag(), getName(), getSizeInBits(),
                        getAlignInBits(), getEncoding());
  }

public:
  DEFINE_MDNODE_GET(DIBasicType, (unsigned Tag, StringRef Name),
                    (Tag, Name, 0, 0, 0))
  DEFINE_MDNODE_GET(DIBasicType,
                    (unsigned Tag, StringRef Name, uint64_t SizeInBits,
                     uint64_t AlignInBits, unsigned Encoding),
                    (Tag, Name, SizeInBits, AlignInBits, Encoding))
  DEFINE_MDNODE_GET(DIBasicType,
                    (unsigned Tag, MDString *Name, uint64_t SizeInBits,
                     uint64_t AlignInBits, unsigned Encoding),
                    (Tag, Name, SizeInBits, AlignInBits, Encoding))

  TempDIBasicType clone() const { return cloneImpl(); }

  unsigned getEncoding() const { return Encoding; }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DIBasicTypeKind;
  }
};

/// \brief Derived types.
///
/// This includes qualified types, pointers, references, friends, typedefs, and
/// class members.
///
/// TODO: Split out members (inheritance, fields, methods, etc.).
class DIDerivedType : public DIType {
  friend class LLVMContextImpl;
  friend class MDNode;

  DIDerivedType(LLVMContext &C, StorageType Storage, unsigned Tag,
                unsigned Line, uint64_t SizeInBits, uint64_t AlignInBits,
                uint64_t OffsetInBits, unsigned Flags, ArrayRef<Metadata *> Ops)
      : DIType(C, DIDerivedTypeKind, Storage, Tag, Line, SizeInBits,
               AlignInBits, OffsetInBits, Flags, Ops) {}
  ~DIDerivedType() = default;

  static DIDerivedType *getImpl(LLVMContext &Context, unsigned Tag,
                                StringRef Name, DIFile *File, unsigned Line,
                                DIScopeRef Scope, DITypeRef BaseType,
                                uint64_t SizeInBits, uint64_t AlignInBits,
                                uint64_t OffsetInBits, unsigned Flags,
                                Metadata *ExtraData, StorageType Storage,
                                bool ShouldCreate = true) {
    return getImpl(Context, Tag, getCanonicalMDString(Context, Name), File,
                   Line, Scope, BaseType, SizeInBits, AlignInBits, OffsetInBits,
                   Flags, ExtraData, Storage, ShouldCreate);
  }
  static DIDerivedType *getImpl(LLVMContext &Context, unsigned Tag,
                                MDString *Name, Metadata *File, unsigned Line,
                                Metadata *Scope, Metadata *BaseType,
                                uint64_t SizeInBits, uint64_t AlignInBits,
                                uint64_t OffsetInBits, unsigned Flags,
                                Metadata *ExtraData, StorageType Storage,
                                bool ShouldCreate = true);

  TempDIDerivedType cloneImpl() const {
    return getTemporary(getContext(), getTag(), getName(), getFile(), getLine(),
                        getScope(), getBaseType(), getSizeInBits(),
                        getAlignInBits(), getOffsetInBits(), getFlags(),
                        getExtraData());
  }

public:
  DEFINE_MDNODE_GET(DIDerivedType,
                    (unsigned Tag, MDString *Name, Metadata *File,
                     unsigned Line, Metadata *Scope, Metadata *BaseType,
                     uint64_t SizeInBits, uint64_t AlignInBits,
                     uint64_t OffsetInBits, unsigned Flags,
                     Metadata *ExtraData = nullptr),
                    (Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                     AlignInBits, OffsetInBits, Flags, ExtraData))
  DEFINE_MDNODE_GET(DIDerivedType,
                    (unsigned Tag, StringRef Name, DIFile *File, unsigned Line,
                     DIScopeRef Scope, DITypeRef BaseType, uint64_t SizeInBits,
                     uint64_t AlignInBits, uint64_t OffsetInBits,
                     unsigned Flags, Metadata *ExtraData = nullptr),
                    (Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                     AlignInBits, OffsetInBits, Flags, ExtraData))

  TempDIDerivedType clone() const { return cloneImpl(); }

  //// Get the base type this is derived from.
  DITypeRef getBaseType() const { return DITypeRef(getRawBaseType()); }
  Metadata *getRawBaseType() const { return getOperand(3); }

  /// \brief Get extra data associated with this derived type.
  ///
  /// Class type for pointer-to-members, objective-c property node for ivars,
  /// or global constant wrapper for static members.
  ///
  /// TODO: Separate out types that need this extra operand: pointer-to-member
  /// types and member fields (static members and ivars).
  Metadata *getExtraData() const { return getRawExtraData(); }
  Metadata *getRawExtraData() const { return getOperand(4); }

  /// \brief Get casted version of extra data.
  /// @{
  DITypeRef getClassType() const {
    assert(getTag() == dwarf::DW_TAG_ptr_to_member_type);
    return DITypeRef(getExtraData());
  }
  DIObjCProperty *getObjCProperty() const {
    return dyn_cast_or_null<DIObjCProperty>(getExtraData());
  }
  Constant *getStorageOffsetInBits() const {
    assert(getTag() == dwarf::DW_TAG_member && isBitField());
    if (auto *C = cast_or_null<ConstantAsMetadata>(getExtraData()))
      return C->getValue();
    return nullptr;
  }
  Constant *getConstant() const {
    assert(getTag() == dwarf::DW_TAG_member && isStaticMember());
    if (auto *C = cast_or_null<ConstantAsMetadata>(getExtraData()))
      return C->getValue();
    return nullptr;
  }
  /// @}

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DIDerivedTypeKind;
  }
};

/// \brief Composite types.
///
/// TODO: Detach from DerivedTypeBase (split out MDEnumType?).
/// TODO: Create a custom, unrelated node for DW_TAG_array_type.
class DICompositeType : public DIType {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned RuntimeLang;

  DICompositeType(LLVMContext &C, StorageType Storage, unsigned Tag,
                  unsigned Line, unsigned RuntimeLang, uint64_t SizeInBits,
                  uint64_t AlignInBits, uint64_t OffsetInBits, unsigned Flags,
                  ArrayRef<Metadata *> Ops)
      : DIType(C, DICompositeTypeKind, Storage, Tag, Line, SizeInBits,
               AlignInBits, OffsetInBits, Flags, Ops),
        RuntimeLang(RuntimeLang) {}
  ~DICompositeType() = default;

  /// Change fields in place.
  void mutate(unsigned Tag, unsigned Line, unsigned RuntimeLang,
              uint64_t SizeInBits, uint64_t AlignInBits, uint64_t OffsetInBits,
              unsigned Flags) {
    assert(isDistinct() && "Only distinct nodes can mutate");
    assert(getRawIdentifier() && "Only ODR-uniqued nodes should mutate");
    this->RuntimeLang = RuntimeLang;
    DIType::mutate(Tag, Line, SizeInBits, AlignInBits, OffsetInBits, Flags);
  }

  static DICompositeType *
  getImpl(LLVMContext &Context, unsigned Tag, StringRef Name, Metadata *File,
          unsigned Line, DIScopeRef Scope, DITypeRef BaseType,
          uint64_t SizeInBits, uint64_t AlignInBits, uint64_t OffsetInBits,
          uint64_t Flags, DINodeArray Elements, unsigned RuntimeLang,
          DITypeRef VTableHolder, DITemplateParameterArray TemplateParams,
          StringRef Identifier, StorageType Storage, bool ShouldCreate = true) {
    return getImpl(
        Context, Tag, getCanonicalMDString(Context, Name), File, Line, Scope,
        BaseType, SizeInBits, AlignInBits, OffsetInBits, Flags, Elements.get(),
        RuntimeLang, VTableHolder, TemplateParams.get(),
        getCanonicalMDString(Context, Identifier), Storage, ShouldCreate);
  }
  static DICompositeType *
  getImpl(LLVMContext &Context, unsigned Tag, MDString *Name, Metadata *File,
          unsigned Line, Metadata *Scope, Metadata *BaseType,
          uint64_t SizeInBits, uint64_t AlignInBits, uint64_t OffsetInBits,
          unsigned Flags, Metadata *Elements, unsigned RuntimeLang,
          Metadata *VTableHolder, Metadata *TemplateParams,
          MDString *Identifier, StorageType Storage, bool ShouldCreate = true);

  TempDICompositeType cloneImpl() const {
    return getTemporary(getContext(), getTag(), getName(), getFile(), getLine(),
                        getScope(), getBaseType(), getSizeInBits(),
                        getAlignInBits(), getOffsetInBits(), getFlags(),
                        getElements(), getRuntimeLang(), getVTableHolder(),
                        getTemplateParams(), getIdentifier());
  }

public:
  DEFINE_MDNODE_GET(DICompositeType,
                    (unsigned Tag, StringRef Name, DIFile *File, unsigned Line,
                     DIScopeRef Scope, DITypeRef BaseType, uint64_t SizeInBits,
                     uint64_t AlignInBits, uint64_t OffsetInBits,
                     unsigned Flags, DINodeArray Elements, unsigned RuntimeLang,
                     DITypeRef VTableHolder,
                     DITemplateParameterArray TemplateParams = nullptr,
                     StringRef Identifier = ""),
                    (Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                     AlignInBits, OffsetInBits, Flags, Elements, RuntimeLang,
                     VTableHolder, TemplateParams, Identifier))
  DEFINE_MDNODE_GET(DICompositeType,
                    (unsigned Tag, MDString *Name, Metadata *File,
                     unsigned Line, Metadata *Scope, Metadata *BaseType,
                     uint64_t SizeInBits, uint64_t AlignInBits,
                     uint64_t OffsetInBits, unsigned Flags, Metadata *Elements,
                     unsigned RuntimeLang, Metadata *VTableHolder,
                     Metadata *TemplateParams = nullptr,
                     MDString *Identifier = nullptr),
                    (Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                     AlignInBits, OffsetInBits, Flags, Elements, RuntimeLang,
                     VTableHolder, TemplateParams, Identifier))

  TempDICompositeType clone() const { return cloneImpl(); }

  /// Get a DICompositeType with the given ODR identifier.
  ///
  /// If \a LLVMContext::isODRUniquingDebugTypes(), gets the mapped
  /// DICompositeType for the given ODR \c Identifier.  If none exists, creates
  /// a new node.
  ///
  /// Else, returns \c nullptr.
  static DICompositeType *
  getODRType(LLVMContext &Context, MDString &Identifier, unsigned Tag,
             MDString *Name, Metadata *File, unsigned Line, Metadata *Scope,
             Metadata *BaseType, uint64_t SizeInBits, uint64_t AlignInBits,
             uint64_t OffsetInBits, unsigned Flags, Metadata *Elements,
             unsigned RuntimeLang, Metadata *VTableHolder,
             Metadata *TemplateParams);
  static DICompositeType *getODRTypeIfExists(LLVMContext &Context,
                                             MDString &Identifier);

  /// Build a DICompositeType with the given ODR identifier.
  ///
  /// Looks up the mapped DICompositeType for the given ODR \c Identifier.  If
  /// it doesn't exist, creates a new one.  If it does exist and \a
  /// isForwardDecl(), and the new arguments would be a definition, mutates the
  /// the type in place.  In either case, returns the type.
  ///
  /// If not \a LLVMContext::isODRUniquingDebugTypes(), this function returns
  /// nullptr.
  static DICompositeType *
  buildODRType(LLVMContext &Context, MDString &Identifier, unsigned Tag,
               MDString *Name, Metadata *File, unsigned Line, Metadata *Scope,
               Metadata *BaseType, uint64_t SizeInBits, uint64_t AlignInBits,
               uint64_t OffsetInBits, unsigned Flags, Metadata *Elements,
               unsigned RuntimeLang, Metadata *VTableHolder,
               Metadata *TemplateParams);

  DITypeRef getBaseType() const { return DITypeRef(getRawBaseType()); }
  DINodeArray getElements() const {
    return cast_or_null<MDTuple>(getRawElements());
  }
  DITypeRef getVTableHolder() const { return DITypeRef(getRawVTableHolder()); }
  DITemplateParameterArray getTemplateParams() const {
    return cast_or_null<MDTuple>(getRawTemplateParams());
  }
  StringRef getIdentifier() const { return getStringOperand(7); }
  unsigned getRuntimeLang() const { return RuntimeLang; }

  Metadata *getRawBaseType() const { return getOperand(3); }
  Metadata *getRawElements() const { return getOperand(4); }
  Metadata *getRawVTableHolder() const { return getOperand(5); }
  Metadata *getRawTemplateParams() const { return getOperand(6); }
  MDString *getRawIdentifier() const { return getOperandAs<MDString>(7); }

  /// \brief Replace operands.
  ///
  /// If this \a isUniqued() and not \a isResolved(), on a uniquing collision
  /// this will be RAUW'ed and deleted.  Use a \a TrackingMDRef to keep track
  /// of its movement if necessary.
  /// @{
  void replaceElements(DINodeArray Elements) {
#ifndef NDEBUG
    for (DINode *Op : getElements())
      assert(is_contained(Elements->operands(), Op) &&
             "Lost a member during member list replacement");
#endif
    replaceOperandWith(4, Elements.get());
  }
  void replaceVTableHolder(DITypeRef VTableHolder) {
    replaceOperandWith(5, VTableHolder);
  }
  void replaceTemplateParams(DITemplateParameterArray TemplateParams) {
    replaceOperandWith(6, TemplateParams.get());
  }
  /// @}

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DICompositeTypeKind;
  }
};

/// \brief Type array for a subprogram.
///
/// TODO: Fold the array of types in directly as operands.
class DISubroutineType : public DIType {
  friend class LLVMContextImpl;
  friend class MDNode;

  /// The calling convention used with DW_AT_calling_convention. Actually of
  /// type dwarf::CallingConvention.
  uint8_t CC;

  DISubroutineType(LLVMContext &C, StorageType Storage, unsigned Flags,
                   uint8_t CC, ArrayRef<Metadata *> Ops)
      : DIType(C, DISubroutineTypeKind, Storage, dwarf::DW_TAG_subroutine_type,
               0, 0, 0, 0, Flags, Ops),
        CC(CC) {}
  ~DISubroutineType() = default;

  static DISubroutineType *getImpl(LLVMContext &Context, unsigned Flags,
                                   uint8_t CC, DITypeRefArray TypeArray,
                                   StorageType Storage,
                                   bool ShouldCreate = true) {
    return getImpl(Context, Flags, CC, TypeArray.get(), Storage, ShouldCreate);
  }
  static DISubroutineType *getImpl(LLVMContext &Context, unsigned Flags,
                                   uint8_t CC, Metadata *TypeArray,
                                   StorageType Storage,
                                   bool ShouldCreate = true);

  TempDISubroutineType cloneImpl() const {
    return getTemporary(getContext(), getFlags(), getCC(), getTypeArray());
  }

public:
  DEFINE_MDNODE_GET(DISubroutineType,
                    (unsigned Flags, uint8_t CC, DITypeRefArray TypeArray),
                    (Flags, CC, TypeArray))
  DEFINE_MDNODE_GET(DISubroutineType,
                    (unsigned Flags, uint8_t CC, Metadata *TypeArray),
                    (Flags, CC, TypeArray))

  TempDISubroutineType clone() const { return cloneImpl(); }

  uint8_t getCC() const { return CC; }

  DITypeRefArray getTypeArray() const {
    return cast_or_null<MDTuple>(getRawTypeArray());
  }
  Metadata *getRawTypeArray() const { return getOperand(3); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DISubroutineTypeKind;
  }
};

/// \brief Compile unit.
class DICompileUnit : public DIScope {
  friend class LLVMContextImpl;
  friend class MDNode;
public:
  enum DebugEmissionKind : unsigned {
    NoDebug = 0,
    FullDebug,
    LineTablesOnly,
    LastEmissionKind = LineTablesOnly
  };
  static Optional<DebugEmissionKind> getEmissionKind(StringRef Str);
  static const char *EmissionKindString(DebugEmissionKind EK);

private:
  unsigned SourceLanguage;
  bool IsOptimized;
  unsigned RuntimeVersion;
  unsigned EmissionKind;
  uint64_t DWOId;
  bool SplitDebugInlining;

  DICompileUnit(LLVMContext &C, StorageType Storage, unsigned SourceLanguage,
                bool IsOptimized, unsigned RuntimeVersion,
                unsigned EmissionKind, uint64_t DWOId, bool SplitDebugInlining,
                ArrayRef<Metadata *> Ops)
      : DIScope(C, DICompileUnitKind, Storage, dwarf::DW_TAG_compile_unit, Ops),
        SourceLanguage(SourceLanguage), IsOptimized(IsOptimized),
        RuntimeVersion(RuntimeVersion), EmissionKind(EmissionKind),
        DWOId(DWOId), SplitDebugInlining(SplitDebugInlining) {
    assert(Storage != Uniqued);
  }
  ~DICompileUnit() = default;

  static DICompileUnit *
  getImpl(LLVMContext &Context, unsigned SourceLanguage, DIFile *File,
          StringRef Producer, bool IsOptimized, StringRef Flags,
          unsigned RuntimeVersion, StringRef SplitDebugFilename,
          unsigned EmissionKind, DICompositeTypeArray EnumTypes,
          DIScopeArray RetainedTypes, DIGlobalVariableArray GlobalVariables,
          DIImportedEntityArray ImportedEntities, DIMacroNodeArray Macros,
          uint64_t DWOId, bool SplitDebugInlining, StorageType Storage,
          bool ShouldCreate = true) {
    return getImpl(Context, SourceLanguage, File,
                   getCanonicalMDString(Context, Producer), IsOptimized,
                   getCanonicalMDString(Context, Flags), RuntimeVersion,
                   getCanonicalMDString(Context, SplitDebugFilename),
                   EmissionKind, EnumTypes.get(), RetainedTypes.get(),
                   GlobalVariables.get(), ImportedEntities.get(), Macros.get(),
                   DWOId, SplitDebugInlining, Storage, ShouldCreate);
  }
  static DICompileUnit *
  getImpl(LLVMContext &Context, unsigned SourceLanguage, Metadata *File,
          MDString *Producer, bool IsOptimized, MDString *Flags,
          unsigned RuntimeVersion, MDString *SplitDebugFilename,
          unsigned EmissionKind, Metadata *EnumTypes, Metadata *RetainedTypes,
          Metadata *GlobalVariables, Metadata *ImportedEntities,
          Metadata *Macros, uint64_t DWOId, bool SplitDebugInlining,
          StorageType Storage, bool ShouldCreate = true);

  TempDICompileUnit cloneImpl() const {
    return getTemporary(getContext(), getSourceLanguage(), getFile(),
                        getProducer(), isOptimized(), getFlags(),
                        getRuntimeVersion(), getSplitDebugFilename(),
                        getEmissionKind(), getEnumTypes(), getRetainedTypes(),
                        getGlobalVariables(), getImportedEntities(),
                        getMacros(), DWOId, getSplitDebugInlining());
  }

  static void get() = delete;
  static void getIfExists() = delete;

public:
  DEFINE_MDNODE_GET_DISTINCT_TEMPORARY(
      DICompileUnit,
      (unsigned SourceLanguage, DIFile *File, StringRef Producer,
       bool IsOptimized, StringRef Flags, unsigned RuntimeVersion,
       StringRef SplitDebugFilename, DebugEmissionKind EmissionKind,
       DICompositeTypeArray EnumTypes, DIScopeArray RetainedTypes,
       DIGlobalVariableArray GlobalVariables,
       DIImportedEntityArray ImportedEntities, DIMacroNodeArray Macros,
       uint64_t DWOId, bool SplitDebugInlining),
      (SourceLanguage, File, Producer, IsOptimized, Flags, RuntimeVersion,
       SplitDebugFilename, EmissionKind, EnumTypes, RetainedTypes,
       GlobalVariables, ImportedEntities, Macros, DWOId, SplitDebugInlining))
  DEFINE_MDNODE_GET_DISTINCT_TEMPORARY(
      DICompileUnit,
      (unsigned SourceLanguage, Metadata *File, MDString *Producer,
       bool IsOptimized, MDString *Flags, unsigned RuntimeVersion,
       MDString *SplitDebugFilename, unsigned EmissionKind, Metadata *EnumTypes,
       Metadata *RetainedTypes, Metadata *GlobalVariables,
       Metadata *ImportedEntities, Metadata *Macros, uint64_t DWOId,
       bool SplitDebugInlining),
      (SourceLanguage, File, Producer, IsOptimized, Flags, RuntimeVersion,
       SplitDebugFilename, EmissionKind, EnumTypes, RetainedTypes,
       GlobalVariables, ImportedEntities, Macros, DWOId, SplitDebugInlining))

  TempDICompileUnit clone() const { return cloneImpl(); }

  unsigned getSourceLanguage() const { return SourceLanguage; }
  bool isOptimized() const { return IsOptimized; }
  unsigned getRuntimeVersion() const { return RuntimeVersion; }
  DebugEmissionKind getEmissionKind() const {
    return (DebugEmissionKind)EmissionKind;
  }
  StringRef getProducer() const { return getStringOperand(1); }
  StringRef getFlags() const { return getStringOperand(2); }
  StringRef getSplitDebugFilename() const { return getStringOperand(3); }
  DICompositeTypeArray getEnumTypes() const {
    return cast_or_null<MDTuple>(getRawEnumTypes());
  }
  DIScopeArray getRetainedTypes() const {
    return cast_or_null<MDTuple>(getRawRetainedTypes());
  }
  DIGlobalVariableArray getGlobalVariables() const {
    return cast_or_null<MDTuple>(getRawGlobalVariables());
  }
  DIImportedEntityArray getImportedEntities() const {
    return cast_or_null<MDTuple>(getRawImportedEntities());
  }
  DIMacroNodeArray getMacros() const {
    return cast_or_null<MDTuple>(getRawMacros());
  }
  uint64_t getDWOId() const { return DWOId; }
  void setDWOId(uint64_t DwoId) { DWOId = DwoId; }
  bool getSplitDebugInlining() const { return SplitDebugInlining; }
  void setSplitDebugInlining(bool SplitDebugInlining) {
    this->SplitDebugInlining = SplitDebugInlining;
  }

  MDString *getRawProducer() const { return getOperandAs<MDString>(1); }
  MDString *getRawFlags() const { return getOperandAs<MDString>(2); }
  MDString *getRawSplitDebugFilename() const {
    return getOperandAs<MDString>(3);
  }
  Metadata *getRawEnumTypes() const { return getOperand(4); }
  Metadata *getRawRetainedTypes() const { return getOperand(5); }
  Metadata *getRawGlobalVariables() const { return getOperand(6); }
  Metadata *getRawImportedEntities() const { return getOperand(7); }
  Metadata *getRawMacros() const { return getOperand(8); }

  /// \brief Replace arrays.
  ///
  /// If this \a isUniqued() and not \a isResolved(), it will be RAUW'ed and
  /// deleted on a uniquing collision.  In practice, uniquing collisions on \a
  /// DICompileUnit should be fairly rare.
  /// @{
  void replaceEnumTypes(DICompositeTypeArray N) {
    replaceOperandWith(4, N.get());
  }
  void replaceRetainedTypes(DITypeArray N) {
    replaceOperandWith(5, N.get());
  }
  void replaceGlobalVariables(DIGlobalVariableArray N) {
    replaceOperandWith(6, N.get());
  }
  void replaceImportedEntities(DIImportedEntityArray N) {
    replaceOperandWith(7, N.get());
  }
  void replaceMacros(DIMacroNodeArray N) { replaceOperandWith(8, N.get()); }
  /// @}

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DICompileUnitKind;
  }
};

/// \brief A scope for locals.
///
/// A legal scope for lexical blocks, local variables, and debug info
/// locations.  Subclasses are \a DISubprogram, \a DILexicalBlock, and \a
/// DILexicalBlockFile.
class DILocalScope : public DIScope {
protected:
  DILocalScope(LLVMContext &C, unsigned ID, StorageType Storage, unsigned Tag,
               ArrayRef<Metadata *> Ops)
      : DIScope(C, ID, Storage, Tag, Ops) {}
  ~DILocalScope() = default;

public:
  /// \brief Get the subprogram for this scope.
  ///
  /// Return this if it's an \a DISubprogram; otherwise, look up the scope
  /// chain.
  DISubprogram *getSubprogram() const;

  /// Get the first non DILexicalBlockFile scope of this scope.
  ///
  /// Return this if it's not a \a DILexicalBlockFIle; otherwise, look up the
  /// scope chain.
  DILocalScope *getNonLexicalBlockFileScope() const;

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DISubprogramKind ||
           MD->getMetadataID() == DILexicalBlockKind ||
           MD->getMetadataID() == DILexicalBlockFileKind;
  }
};

/// \brief Debug location.
///
/// A debug location in source code, used for debug info and otherwise.
class DILocation : public MDNode {
  friend class LLVMContextImpl;
  friend class MDNode;

  DILocation(LLVMContext &C, StorageType Storage, unsigned Line,
             unsigned Column, ArrayRef<Metadata *> MDs);
  ~DILocation() { dropAllReferences(); }

  static DILocation *getImpl(LLVMContext &Context, unsigned Line,
                             unsigned Column, Metadata *Scope,
                             Metadata *InlinedAt, StorageType Storage,
                             bool ShouldCreate = true);
  static DILocation *getImpl(LLVMContext &Context, unsigned Line,
                             unsigned Column, DILocalScope *Scope,
                             DILocation *InlinedAt, StorageType Storage,
                             bool ShouldCreate = true) {
    return getImpl(Context, Line, Column, static_cast<Metadata *>(Scope),
                   static_cast<Metadata *>(InlinedAt), Storage, ShouldCreate);
  }

  TempDILocation cloneImpl() const {
    // Get the raw scope/inlinedAt since it is possible to invoke this on
    // a DILocation containing temporary metadata.
    return getTemporary(getContext(), getLine(), getColumn(), getRawScope(),
                        getRawInlinedAt());
  }

  // Disallow replacing operands.
  void replaceOperandWith(unsigned I, Metadata *New) = delete;

public:
  DEFINE_MDNODE_GET(DILocation,
                    (unsigned Line, unsigned Column, Metadata *Scope,
                     Metadata *InlinedAt = nullptr),
                    (Line, Column, Scope, InlinedAt))
  DEFINE_MDNODE_GET(DILocation,
                    (unsigned Line, unsigned Column, DILocalScope *Scope,
                     DILocation *InlinedAt = nullptr),
                    (Line, Column, Scope, InlinedAt))

  /// \brief Return a (temporary) clone of this.
  TempDILocation clone() const { return cloneImpl(); }

  unsigned getLine() const { return SubclassData32; }
  unsigned getColumn() const { return SubclassData16; }
  DILocalScope *getScope() const { return cast<DILocalScope>(getRawScope()); }
  DILocation *getInlinedAt() const {
    return cast_or_null<DILocation>(getRawInlinedAt());
  }

  DIFile *getFile() const { return getScope()->getFile(); }
  StringRef getFilename() const { return getScope()->getFilename(); }
  StringRef getDirectory() const { return getScope()->getDirectory(); }

  /// \brief Get the scope where this is inlined.
  ///
  /// Walk through \a getInlinedAt() and return \a getScope() from the deepest
  /// location.
  DILocalScope *getInlinedAtScope() const {
    if (auto *IA = getInlinedAt())
      return IA->getInlinedAtScope();
    return getScope();
  }

  /// \brief Check whether this can be discriminated from another location.
  ///
  /// Check \c this can be discriminated from \c RHS in a linetable entry.
  /// Scope and inlined-at chains are not recorded in the linetable, so they
  /// cannot be used to distinguish basic blocks.
  ///
  /// The current implementation is weaker than it should be, since it just
  /// checks filename and line.
  ///
  /// FIXME: Add a check for getDiscriminator().
  /// FIXME: Add a check for getColumn().
  /// FIXME: Change the getFilename() check to getFile() (or add one for
  /// getDirectory()).
  bool canDiscriminate(const DILocation &RHS) const {
    return getFilename() != RHS.getFilename() || getLine() != RHS.getLine();
  }

  /// \brief Get the DWARF discriminator.
  ///
  /// DWARF discriminators distinguish identical file locations between
  /// instructions that are on different basic blocks.
  inline unsigned getDiscriminator() const;

  Metadata *getRawScope() const { return getOperand(0); }
  Metadata *getRawInlinedAt() const {
    if (getNumOperands() == 2)
      return getOperand(1);
    return nullptr;
  }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DILocationKind;
  }
};

/// \brief Subprogram description.
///
/// TODO: Remove DisplayName.  It's always equal to Name.
/// TODO: Split up flags.
class DISubprogram : public DILocalScope {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Line;
  unsigned ScopeLine;
  unsigned VirtualIndex;

  /// In the MS ABI, the implicit 'this' parameter is adjusted in the prologue
  /// of method overrides from secondary bases by this amount. It may be
  /// negative.
  int ThisAdjustment;

  // Virtuality can only assume three values, so we can pack
  // in 2 bits (none/pure/pure_virtual).
  unsigned Virtuality : 2;

  unsigned Flags : 27;

  // These are boolean flags so one bit is enough.
  // MSVC starts a new container field every time the base
  // type changes so we can't use 'bool' to ensure these bits
  // are packed.
  unsigned IsLocalToUnit : 1;
  unsigned IsDefinition : 1;
  unsigned IsOptimized : 1;

  DISubprogram(LLVMContext &C, StorageType Storage, unsigned Line,
               unsigned ScopeLine, unsigned Virtuality, unsigned VirtualIndex,
               int ThisAdjustment, unsigned Flags, bool IsLocalToUnit,
               bool IsDefinition, bool IsOptimized, ArrayRef<Metadata *> Ops)
      : DILocalScope(C, DISubprogramKind, Storage, dwarf::DW_TAG_subprogram,
                     Ops),
        Line(Line), ScopeLine(ScopeLine), VirtualIndex(VirtualIndex),
        ThisAdjustment(ThisAdjustment), Virtuality(Virtuality), Flags(Flags),
        IsLocalToUnit(IsLocalToUnit), IsDefinition(IsDefinition),
        IsOptimized(IsOptimized) {
    static_assert(dwarf::DW_VIRTUALITY_max < 4, "Virtuality out of range");
    assert(Virtuality < 4 && "Virtuality out of range");
    assert((Flags < (1 << 27)) && "Flags out of range");
  }
  ~DISubprogram() = default;

  static DISubprogram *
  getImpl(LLVMContext &Context, DIScopeRef Scope, StringRef Name,
          StringRef LinkageName, DIFile *File, unsigned Line,
          DISubroutineType *Type, bool IsLocalToUnit, bool IsDefinition,
          unsigned ScopeLine, DITypeRef ContainingType, unsigned Virtuality,
          unsigned VirtualIndex, int ThisAdjustment, unsigned Flags,
          bool IsOptimized, DICompileUnit *Unit,
          DITemplateParameterArray TemplateParams, DISubprogram *Declaration,
          DILocalVariableArray Variables, StorageType Storage,
          bool ShouldCreate = true) {
    return getImpl(Context, Scope, getCanonicalMDString(Context, Name),
                   getCanonicalMDString(Context, LinkageName), File, Line, Type,
                   IsLocalToUnit, IsDefinition, ScopeLine, ContainingType,
                   Virtuality, VirtualIndex, ThisAdjustment, Flags, IsOptimized,
                   Unit, TemplateParams.get(), Declaration, Variables.get(),
                   Storage, ShouldCreate);
  }
  static DISubprogram *
  getImpl(LLVMContext &Context, Metadata *Scope, MDString *Name,
          MDString *LinkageName, Metadata *File, unsigned Line, Metadata *Type,
          bool IsLocalToUnit, bool IsDefinition, unsigned ScopeLine,
          Metadata *ContainingType, unsigned Virtuality, unsigned VirtualIndex,
          int ThisAdjustment, unsigned Flags, bool IsOptimized, Metadata *Unit,
          Metadata *TemplateParams, Metadata *Declaration, Metadata *Variables,
          StorageType Storage, bool ShouldCreate = true);

  TempDISubprogram cloneImpl() const {
    return getTemporary(
        getContext(), getScope(), getName(), getLinkageName(), getFile(),
        getLine(), getType(), isLocalToUnit(), isDefinition(), getScopeLine(),
        getContainingType(), getVirtuality(), getVirtualIndex(),
        getThisAdjustment(), getFlags(), isOptimized(), getUnit(),
        getTemplateParams(), getDeclaration(), getVariables());
  }

public:
  DEFINE_MDNODE_GET(DISubprogram,
                    (DIScopeRef Scope, StringRef Name, StringRef LinkageName,
                     DIFile *File, unsigned Line, DISubroutineType *Type,
                     bool IsLocalToUnit, bool IsDefinition, unsigned ScopeLine,
                     DITypeRef ContainingType, unsigned Virtuality,
                     unsigned VirtualIndex, int ThisAdjustment, unsigned Flags,
                     bool IsOptimized, DICompileUnit *Unit,
                     DITemplateParameterArray TemplateParams = nullptr,
                     DISubprogram *Declaration = nullptr,
                     DILocalVariableArray Variables = nullptr),
                    (Scope, Name, LinkageName, File, Line, Type, IsLocalToUnit,
                     IsDefinition, ScopeLine, ContainingType, Virtuality,
                     VirtualIndex, ThisAdjustment, Flags, IsOptimized, Unit,
                     TemplateParams, Declaration, Variables))
  DEFINE_MDNODE_GET(
      DISubprogram,
      (Metadata * Scope, MDString *Name, MDString *LinkageName, Metadata *File,
       unsigned Line, Metadata *Type, bool IsLocalToUnit, bool IsDefinition,
       unsigned ScopeLine, Metadata *ContainingType, unsigned Virtuality,
       unsigned VirtualIndex, int ThisAdjustment, unsigned Flags,
       bool IsOptimized, Metadata *Unit, Metadata *TemplateParams = nullptr,
       Metadata *Declaration = nullptr, Metadata *Variables = nullptr),
      (Scope, Name, LinkageName, File, Line, Type, IsLocalToUnit, IsDefinition,
       ScopeLine, ContainingType, Virtuality, VirtualIndex, ThisAdjustment,
       Flags, IsOptimized, Unit, TemplateParams, Declaration, Variables))

  TempDISubprogram clone() const { return cloneImpl(); }

public:
  unsigned getLine() const { return Line; }
  unsigned getVirtuality() const { return Virtuality; }
  unsigned getVirtualIndex() const { return VirtualIndex; }
  int getThisAdjustment() const { return ThisAdjustment; }
  unsigned getScopeLine() const { return ScopeLine; }
  unsigned getFlags() const { return Flags; }
  bool isLocalToUnit() const { return IsLocalToUnit; }
  bool isDefinition() const { return IsDefinition; }
  bool isOptimized() const { return IsOptimized; }

  unsigned isArtificial() const { return getFlags() & FlagArtificial; }
  bool isPrivate() const {
    return (getFlags() & FlagAccessibility) == FlagPrivate;
  }
  bool isProtected() const {
    return (getFlags() & FlagAccessibility) == FlagProtected;
  }
  bool isPublic() const {
    return (getFlags() & FlagAccessibility) == FlagPublic;
  }
  bool isExplicit() const { return getFlags() & FlagExplicit; }
  bool isPrototyped() const { return getFlags() & FlagPrototyped; }

  /// \brief Check if this is reference-qualified.
  ///
  /// Return true if this subprogram is a C++11 reference-qualified non-static
  /// member function (void foo() &).
  unsigned isLValueReference() const {
    return getFlags() & FlagLValueReference;
  }

  /// \brief Check if this is rvalue-reference-qualified.
  ///
  /// Return true if this subprogram is a C++11 rvalue-reference-qualified
  /// non-static member function (void foo() &&).
  unsigned isRValueReference() const {
    return getFlags() & FlagRValueReference;
  }

  /// \brief Check if this is marked as noreturn.
  ///
  /// Return true if this subprogram is C++11 noreturn or C11 _Noreturn
  unsigned isNoReturn() const {
    return getFlags() & FlagNoReturn;
  }

  DIScopeRef getScope() const { return DIScopeRef(getRawScope()); }

  StringRef getName() const { return getStringOperand(2); }
  StringRef getDisplayName() const { return getStringOperand(3); }
  StringRef getLinkageName() const { return getStringOperand(4); }

  MDString *getRawName() const { return getOperandAs<MDString>(2); }
  MDString *getRawLinkageName() const { return getOperandAs<MDString>(4); }

  DISubroutineType *getType() const {
    return cast_or_null<DISubroutineType>(getRawType());
  }
  DITypeRef getContainingType() const {
    return DITypeRef(getRawContainingType());
  }

  DICompileUnit *getUnit() const {
    return cast_or_null<DICompileUnit>(getRawUnit());
  }
  void replaceUnit(DICompileUnit *CU) {
    replaceOperandWith(7, CU);
  }
  DITemplateParameterArray getTemplateParams() const {
    return cast_or_null<MDTuple>(getRawTemplateParams());
  }
  DISubprogram *getDeclaration() const {
    return cast_or_null<DISubprogram>(getRawDeclaration());
  }
  DILocalVariableArray getVariables() const {
    return cast_or_null<MDTuple>(getRawVariables());
  }

  Metadata *getRawScope() const { return getOperand(1); }
  Metadata *getRawType() const { return getOperand(5); }
  Metadata *getRawContainingType() const { return getOperand(6); }
  Metadata *getRawUnit() const { return getOperand(7); }
  Metadata *getRawTemplateParams() const { return getOperand(8); }
  Metadata *getRawDeclaration() const { return getOperand(9); }
  Metadata *getRawVariables() const { return getOperand(10); }

  /// \brief Check if this subprogram describes the given function.
  ///
  /// FIXME: Should this be looking through bitcasts?
  bool describes(const Function *F) const;

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DISubprogramKind;
  }
};

class DILexicalBlockBase : public DILocalScope {
protected:
  DILexicalBlockBase(LLVMContext &C, unsigned ID, StorageType Storage,
                     ArrayRef<Metadata *> Ops)
      : DILocalScope(C, ID, Storage, dwarf::DW_TAG_lexical_block, Ops) {}
  ~DILexicalBlockBase() = default;

public:
  DILocalScope *getScope() const { return cast<DILocalScope>(getRawScope()); }

  Metadata *getRawScope() const { return getOperand(1); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DILexicalBlockKind ||
           MD->getMetadataID() == DILexicalBlockFileKind;
  }
};

class DILexicalBlock : public DILexicalBlockBase {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Line;
  uint16_t Column;

  DILexicalBlock(LLVMContext &C, StorageType Storage, unsigned Line,
                 unsigned Column, ArrayRef<Metadata *> Ops)
      : DILexicalBlockBase(C, DILexicalBlockKind, Storage, Ops), Line(Line),
        Column(Column) {
    assert(Column < (1u << 16) && "Expected 16-bit column");
  }
  ~DILexicalBlock() = default;

  static DILexicalBlock *getImpl(LLVMContext &Context, DILocalScope *Scope,
                                 DIFile *File, unsigned Line, unsigned Column,
                                 StorageType Storage,
                                 bool ShouldCreate = true) {
    return getImpl(Context, static_cast<Metadata *>(Scope),
                   static_cast<Metadata *>(File), Line, Column, Storage,
                   ShouldCreate);
  }

  static DILexicalBlock *getImpl(LLVMContext &Context, Metadata *Scope,
                                 Metadata *File, unsigned Line, unsigned Column,
                                 StorageType Storage, bool ShouldCreate = true);

  TempDILexicalBlock cloneImpl() const {
    return getTemporary(getContext(), getScope(), getFile(), getLine(),
                        getColumn());
  }

public:
  DEFINE_MDNODE_GET(DILexicalBlock, (DILocalScope * Scope, DIFile *File,
                                     unsigned Line, unsigned Column),
                    (Scope, File, Line, Column))
  DEFINE_MDNODE_GET(DILexicalBlock, (Metadata * Scope, Metadata *File,
                                     unsigned Line, unsigned Column),
                    (Scope, File, Line, Column))

  TempDILexicalBlock clone() const { return cloneImpl(); }

  unsigned getLine() const { return Line; }
  unsigned getColumn() const { return Column; }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DILexicalBlockKind;
  }
};

class DILexicalBlockFile : public DILexicalBlockBase {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Discriminator;

  DILexicalBlockFile(LLVMContext &C, StorageType Storage,
                     unsigned Discriminator, ArrayRef<Metadata *> Ops)
      : DILexicalBlockBase(C, DILexicalBlockFileKind, Storage, Ops),
        Discriminator(Discriminator) {}
  ~DILexicalBlockFile() = default;

  static DILexicalBlockFile *getImpl(LLVMContext &Context, DILocalScope *Scope,
                                     DIFile *File, unsigned Discriminator,
                                     StorageType Storage,
                                     bool ShouldCreate = true) {
    return getImpl(Context, static_cast<Metadata *>(Scope),
                   static_cast<Metadata *>(File), Discriminator, Storage,
                   ShouldCreate);
  }

  static DILexicalBlockFile *getImpl(LLVMContext &Context, Metadata *Scope,
                                     Metadata *File, unsigned Discriminator,
                                     StorageType Storage,
                                     bool ShouldCreate = true);

  TempDILexicalBlockFile cloneImpl() const {
    return getTemporary(getContext(), getScope(), getFile(),
                        getDiscriminator());
  }

public:
  DEFINE_MDNODE_GET(DILexicalBlockFile, (DILocalScope * Scope, DIFile *File,
                                         unsigned Discriminator),
                    (Scope, File, Discriminator))
  DEFINE_MDNODE_GET(DILexicalBlockFile,
                    (Metadata * Scope, Metadata *File, unsigned Discriminator),
                    (Scope, File, Discriminator))

  TempDILexicalBlockFile clone() const { return cloneImpl(); }

  // TODO: Remove these once they're gone from DILexicalBlockBase.
  unsigned getLine() const = delete;
  unsigned getColumn() const = delete;

  unsigned getDiscriminator() const { return Discriminator; }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DILexicalBlockFileKind;
  }
};

unsigned DILocation::getDiscriminator() const {
  if (auto *F = dyn_cast<DILexicalBlockFile>(getScope()))
    return F->getDiscriminator();
  return 0;
}

class DINamespace : public DIScope {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Line;

  DINamespace(LLVMContext &Context, StorageType Storage, unsigned Line,
              ArrayRef<Metadata *> Ops)
      : DIScope(Context, DINamespaceKind, Storage, dwarf::DW_TAG_namespace,
                Ops),
        Line(Line) {}
  ~DINamespace() = default;

  static DINamespace *getImpl(LLVMContext &Context, DIScope *Scope,
                              DIFile *File, StringRef Name, unsigned Line,
                              StorageType Storage, bool ShouldCreate = true) {
    return getImpl(Context, Scope, File, getCanonicalMDString(Context, Name),
                   Line, Storage, ShouldCreate);
  }
  static DINamespace *getImpl(LLVMContext &Context, Metadata *Scope,
                              Metadata *File, MDString *Name, unsigned Line,
                              StorageType Storage, bool ShouldCreate = true);

  TempDINamespace cloneImpl() const {
    return getTemporary(getContext(), getScope(), getFile(), getName(),
                        getLine());
  }

public:
  DEFINE_MDNODE_GET(DINamespace, (DIScope * Scope, DIFile *File, StringRef Name,
                                  unsigned Line),
                    (Scope, File, Name, Line))
  DEFINE_MDNODE_GET(DINamespace, (Metadata * Scope, Metadata *File,
                                  MDString *Name, unsigned Line),
                    (Scope, File, Name, Line))

  TempDINamespace clone() const { return cloneImpl(); }

  unsigned getLine() const { return Line; }
  DIScope *getScope() const { return cast_or_null<DIScope>(getRawScope()); }
  StringRef getName() const { return getStringOperand(2); }

  Metadata *getRawScope() const { return getOperand(1); }
  MDString *getRawName() const { return getOperandAs<MDString>(2); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DINamespaceKind;
  }
};

/// \brief A (clang) module that has been imported by the compile unit.
///
class DIModule : public DIScope {
  friend class LLVMContextImpl;
  friend class MDNode;

  DIModule(LLVMContext &Context, StorageType Storage, ArrayRef<Metadata *> Ops)
      : DIScope(Context, DIModuleKind, Storage, dwarf::DW_TAG_module, Ops) {}
  ~DIModule() {}

  static DIModule *getImpl(LLVMContext &Context, DIScope *Scope,
                           StringRef Name, StringRef ConfigurationMacros,
                           StringRef IncludePath, StringRef ISysRoot,
                           StorageType Storage, bool ShouldCreate = true) {
    return getImpl(Context, Scope, getCanonicalMDString(Context, Name),
                   getCanonicalMDString(Context, ConfigurationMacros),
                   getCanonicalMDString(Context, IncludePath),
                   getCanonicalMDString(Context, ISysRoot),
                   Storage, ShouldCreate);
  }
  static DIModule *getImpl(LLVMContext &Context, Metadata *Scope,
                           MDString *Name, MDString *ConfigurationMacros,
                           MDString *IncludePath, MDString *ISysRoot,
                           StorageType Storage, bool ShouldCreate = true);

  TempDIModule cloneImpl() const {
    return getTemporary(getContext(), getScope(), getName(),
                        getConfigurationMacros(), getIncludePath(),
                        getISysRoot());
  }

public:
  DEFINE_MDNODE_GET(DIModule, (DIScope *Scope, StringRef Name,
                               StringRef ConfigurationMacros, StringRef IncludePath,
                               StringRef ISysRoot),
                    (Scope, Name, ConfigurationMacros, IncludePath, ISysRoot))
  DEFINE_MDNODE_GET(DIModule,
                    (Metadata *Scope, MDString *Name, MDString *ConfigurationMacros,
                     MDString *IncludePath, MDString *ISysRoot),
                    (Scope, Name, ConfigurationMacros, IncludePath, ISysRoot))

  TempDIModule clone() const { return cloneImpl(); }

  DIScope *getScope() const { return cast_or_null<DIScope>(getRawScope()); }
  StringRef getName() const { return getStringOperand(1); }
  StringRef getConfigurationMacros() const { return getStringOperand(2); }
  StringRef getIncludePath() const { return getStringOperand(3); }
  StringRef getISysRoot() const { return getStringOperand(4); }

  Metadata *getRawScope() const { return getOperand(0); }
  MDString *getRawName() const { return getOperandAs<MDString>(1); }
  MDString *getRawConfigurationMacros() const { return getOperandAs<MDString>(2); }
  MDString *getRawIncludePath() const { return getOperandAs<MDString>(3); }
  MDString *getRawISysRoot() const { return getOperandAs<MDString>(4); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DIModuleKind;
  }
};

/// \brief Base class for template parameters.
class DITemplateParameter : public DINode {
protected:
  DITemplateParameter(LLVMContext &Context, unsigned ID, StorageType Storage,
                      unsigned Tag, ArrayRef<Metadata *> Ops)
      : DINode(Context, ID, Storage, Tag, Ops) {}
  ~DITemplateParameter() = default;

public:
  StringRef getName() const { return getStringOperand(0); }
  DITypeRef getType() const { return DITypeRef(getRawType()); }

  MDString *getRawName() const { return getOperandAs<MDString>(0); }
  Metadata *getRawType() const { return getOperand(1); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DITemplateTypeParameterKind ||
           MD->getMetadataID() == DITemplateValueParameterKind;
  }
};

class DITemplateTypeParameter : public DITemplateParameter {
  friend class LLVMContextImpl;
  friend class MDNode;

  DITemplateTypeParameter(LLVMContext &Context, StorageType Storage,
                          ArrayRef<Metadata *> Ops)
      : DITemplateParameter(Context, DITemplateTypeParameterKind, Storage,
                            dwarf::DW_TAG_template_type_parameter, Ops) {}
  ~DITemplateTypeParameter() = default;

  static DITemplateTypeParameter *getImpl(LLVMContext &Context, StringRef Name,
                                          DITypeRef Type, StorageType Storage,
                                          bool ShouldCreate = true) {
    return getImpl(Context, getCanonicalMDString(Context, Name), Type, Storage,
                   ShouldCreate);
  }
  static DITemplateTypeParameter *getImpl(LLVMContext &Context, MDString *Name,
                                          Metadata *Type, StorageType Storage,
                                          bool ShouldCreate = true);

  TempDITemplateTypeParameter cloneImpl() const {
    return getTemporary(getContext(), getName(), getType());
  }

public:
  DEFINE_MDNODE_GET(DITemplateTypeParameter, (StringRef Name, DITypeRef Type),
                    (Name, Type))
  DEFINE_MDNODE_GET(DITemplateTypeParameter, (MDString * Name, Metadata *Type),
                    (Name, Type))

  TempDITemplateTypeParameter clone() const { return cloneImpl(); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DITemplateTypeParameterKind;
  }
};

class DITemplateValueParameter : public DITemplateParameter {
  friend class LLVMContextImpl;
  friend class MDNode;

  DITemplateValueParameter(LLVMContext &Context, StorageType Storage,
                           unsigned Tag, ArrayRef<Metadata *> Ops)
      : DITemplateParameter(Context, DITemplateValueParameterKind, Storage, Tag,
                            Ops) {}
  ~DITemplateValueParameter() = default;

  static DITemplateValueParameter *getImpl(LLVMContext &Context, unsigned Tag,
                                           StringRef Name, DITypeRef Type,
                                           Metadata *Value, StorageType Storage,
                                           bool ShouldCreate = true) {
    return getImpl(Context, Tag, getCanonicalMDString(Context, Name), Type,
                   Value, Storage, ShouldCreate);
  }
  static DITemplateValueParameter *getImpl(LLVMContext &Context, unsigned Tag,
                                           MDString *Name, Metadata *Type,
                                           Metadata *Value, StorageType Storage,
                                           bool ShouldCreate = true);

  TempDITemplateValueParameter cloneImpl() const {
    return getTemporary(getContext(), getTag(), getName(), getType(),
                        getValue());
  }

public:
  DEFINE_MDNODE_GET(DITemplateValueParameter, (unsigned Tag, StringRef Name,
                                               DITypeRef Type, Metadata *Value),
                    (Tag, Name, Type, Value))
  DEFINE_MDNODE_GET(DITemplateValueParameter, (unsigned Tag, MDString *Name,
                                               Metadata *Type, Metadata *Value),
                    (Tag, Name, Type, Value))

  TempDITemplateValueParameter clone() const { return cloneImpl(); }

  Metadata *getValue() const { return getOperand(2); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DITemplateValueParameterKind;
  }
};

/// \brief Base class for variables.
class DIVariable : public DINode {
  unsigned Line;

protected:
  DIVariable(LLVMContext &C, unsigned ID, StorageType Storage, unsigned Line,
             ArrayRef<Metadata *> Ops)
      : DINode(C, ID, Storage, dwarf::DW_TAG_variable, Ops), Line(Line) {}
  ~DIVariable() = default;

public:
  unsigned getLine() const { return Line; }
  DIScope *getScope() const { return cast_or_null<DIScope>(getRawScope()); }
  StringRef getName() const { return getStringOperand(1); }
  DIFile *getFile() const { return cast_or_null<DIFile>(getRawFile()); }
  DITypeRef getType() const { return DITypeRef(getRawType()); }

  StringRef getFilename() const {
    if (auto *F = getFile())
      return F->getFilename();
    return "";
  }
  StringRef getDirectory() const {
    if (auto *F = getFile())
      return F->getDirectory();
    return "";
  }

  Metadata *getRawScope() const { return getOperand(0); }
  MDString *getRawName() const { return getOperandAs<MDString>(1); }
  Metadata *getRawFile() const { return getOperand(2); }
  Metadata *getRawType() const { return getOperand(3); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DILocalVariableKind ||
           MD->getMetadataID() == DIGlobalVariableKind;
  }
};

/// \brief Global variables.
///
/// TODO: Remove DisplayName.  It's always equal to Name.
class DIGlobalVariable : public DIVariable {
  friend class LLVMContextImpl;
  friend class MDNode;

  bool IsLocalToUnit;
  bool IsDefinition;

  DIGlobalVariable(LLVMContext &C, StorageType Storage, unsigned Line,
                   bool IsLocalToUnit, bool IsDefinition,
                   ArrayRef<Metadata *> Ops)
      : DIVariable(C, DIGlobalVariableKind, Storage, Line, Ops),
        IsLocalToUnit(IsLocalToUnit), IsDefinition(IsDefinition) {}
  ~DIGlobalVariable() = default;

  static DIGlobalVariable *
  getImpl(LLVMContext &Context, DIScope *Scope, StringRef Name,
          StringRef LinkageName, DIFile *File, unsigned Line, DITypeRef Type,
          bool IsLocalToUnit, bool IsDefinition, Constant *Variable,
          DIDerivedType *StaticDataMemberDeclaration, StorageType Storage,
          bool ShouldCreate = true) {
    return getImpl(Context, Scope, getCanonicalMDString(Context, Name),
                   getCanonicalMDString(Context, LinkageName), File, Line, Type,
                   IsLocalToUnit, IsDefinition,
                   Variable ? ConstantAsMetadata::get(Variable) : nullptr,
                   StaticDataMemberDeclaration, Storage, ShouldCreate);
  }
  static DIGlobalVariable *
  getImpl(LLVMContext &Context, Metadata *Scope, MDString *Name,
          MDString *LinkageName, Metadata *File, unsigned Line, Metadata *Type,
          bool IsLocalToUnit, bool IsDefinition, Metadata *Variable,
          Metadata *StaticDataMemberDeclaration, StorageType Storage,
          bool ShouldCreate = true);

  TempDIGlobalVariable cloneImpl() const {
    return getTemporary(getContext(), getScope(), getName(), getLinkageName(),
                        getFile(), getLine(), getType(), isLocalToUnit(),
                        isDefinition(), getVariable(),
                        getStaticDataMemberDeclaration());
  }

public:
  DEFINE_MDNODE_GET(DIGlobalVariable,
                    (DIScope * Scope, StringRef Name, StringRef LinkageName,
                     DIFile *File, unsigned Line, DITypeRef Type,
                     bool IsLocalToUnit, bool IsDefinition, Constant *Variable,
                     DIDerivedType *StaticDataMemberDeclaration),
                    (Scope, Name, LinkageName, File, Line, Type, IsLocalToUnit,
                     IsDefinition, Variable, StaticDataMemberDeclaration))
  DEFINE_MDNODE_GET(DIGlobalVariable,
                    (Metadata * Scope, MDString *Name, MDString *LinkageName,
                     Metadata *File, unsigned Line, Metadata *Type,
                     bool IsLocalToUnit, bool IsDefinition, Metadata *Variable,
                     Metadata *StaticDataMemberDeclaration),
                    (Scope, Name, LinkageName, File, Line, Type, IsLocalToUnit,
                     IsDefinition, Variable, StaticDataMemberDeclaration))

  TempDIGlobalVariable clone() const { return cloneImpl(); }

  bool isLocalToUnit() const { return IsLocalToUnit; }
  bool isDefinition() const { return IsDefinition; }
  StringRef getDisplayName() const { return getStringOperand(4); }
  StringRef getLinkageName() const { return getStringOperand(5); }
  Constant *getVariable() const {
    if (auto *C = cast_or_null<ConstantAsMetadata>(getRawVariable()))
      return dyn_cast<Constant>(C->getValue());
    return nullptr;
  }
  DIDerivedType *getStaticDataMemberDeclaration() const {
    return cast_or_null<DIDerivedType>(getRawStaticDataMemberDeclaration());
  }

  MDString *getRawLinkageName() const { return getOperandAs<MDString>(5); }
  Metadata *getRawVariable() const { return getOperand(6); }
  Metadata *getRawStaticDataMemberDeclaration() const { return getOperand(7); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DIGlobalVariableKind;
  }
};

/// \brief Local variable.
///
/// TODO: Split up flags.
class DILocalVariable : public DIVariable {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Arg : 16;
  unsigned Flags : 16;

  DILocalVariable(LLVMContext &C, StorageType Storage, unsigned Line,
                  unsigned Arg, unsigned Flags, ArrayRef<Metadata *> Ops)
      : DIVariable(C, DILocalVariableKind, Storage, Line, Ops), Arg(Arg),
        Flags(Flags) {
    assert(Flags < (1 << 16) && "DILocalVariable: Flags out of range");
    assert(Arg < (1 << 16) && "DILocalVariable: Arg out of range");
  }
  ~DILocalVariable() = default;

  static DILocalVariable *getImpl(LLVMContext &Context, DIScope *Scope,
                                  StringRef Name, DIFile *File, unsigned Line,
                                  DITypeRef Type, unsigned Arg, unsigned Flags,
                                  StorageType Storage,
                                  bool ShouldCreate = true) {
    return getImpl(Context, Scope, getCanonicalMDString(Context, Name), File,
                   Line, Type, Arg, Flags, Storage, ShouldCreate);
  }
  static DILocalVariable *getImpl(LLVMContext &Context, Metadata *Scope,
                                  MDString *Name, Metadata *File, unsigned Line,
                                  Metadata *Type, unsigned Arg, unsigned Flags,
                                  StorageType Storage,
                                  bool ShouldCreate = true);

  TempDILocalVariable cloneImpl() const {
    return getTemporary(getContext(), getScope(), getName(), getFile(),
                        getLine(), getType(), getArg(), getFlags());
  }

public:
  DEFINE_MDNODE_GET(DILocalVariable,
                    (DILocalScope * Scope, StringRef Name, DIFile *File,
                     unsigned Line, DITypeRef Type, unsigned Arg,
                     unsigned Flags),
                    (Scope, Name, File, Line, Type, Arg, Flags))
  DEFINE_MDNODE_GET(DILocalVariable,
                    (Metadata * Scope, MDString *Name, Metadata *File,
                     unsigned Line, Metadata *Type, unsigned Arg,
                     unsigned Flags),
                    (Scope, Name, File, Line, Type, Arg, Flags))

  TempDILocalVariable clone() const { return cloneImpl(); }

  /// \brief Get the local scope for this variable.
  ///
  /// Variables must be defined in a local scope.
  DILocalScope *getScope() const {
    return cast<DILocalScope>(DIVariable::getScope());
  }

  bool isParameter() const { return Arg; }
  unsigned getArg() const { return Arg; }
  unsigned getFlags() const { return Flags; }

  bool isArtificial() const { return getFlags() & FlagArtificial; }
  bool isObjectPointer() const { return getFlags() & FlagObjectPointer; }

  /// \brief Check that a location is valid for this variable.
  ///
  /// Check that \c DL exists, is in the same subprogram, and has the same
  /// inlined-at location as \c this.  (Otherwise, it's not a valid attachment
  /// to a \a DbgInfoIntrinsic.)
  bool isValidLocationForIntrinsic(const DILocation *DL) const {
    return DL && getScope()->getSubprogram() == DL->getScope()->getSubprogram();
  }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DILocalVariableKind;
  }
};

/// \brief DWARF expression.
///
/// This is (almost) a DWARF expression that modifies the location of a
/// variable or (or the location of a single piece of a variable).
///
/// FIXME: Instead of DW_OP_plus taking an argument, this should use DW_OP_const
/// and have DW_OP_plus consume the topmost elements on the stack.
///
/// TODO: Co-allocate the expression elements.
/// TODO: Separate from MDNode, or otherwise drop Distinct and Temporary
/// storage types.
class DIExpression : public MDNode {
  friend class LLVMContextImpl;
  friend class MDNode;

  std::vector<uint64_t> Elements;

  DIExpression(LLVMContext &C, StorageType Storage, ArrayRef<uint64_t> Elements)
      : MDNode(C, DIExpressionKind, Storage, None),
        Elements(Elements.begin(), Elements.end()) {}
  ~DIExpression() = default;

  static DIExpression *getImpl(LLVMContext &Context,
                               ArrayRef<uint64_t> Elements, StorageType Storage,
                               bool ShouldCreate = true);

  TempDIExpression cloneImpl() const {
    return getTemporary(getContext(), getElements());
  }

public:
  DEFINE_MDNODE_GET(DIExpression, (ArrayRef<uint64_t> Elements), (Elements))

  TempDIExpression clone() const { return cloneImpl(); }

  ArrayRef<uint64_t> getElements() const { return Elements; }

  unsigned getNumElements() const { return Elements.size(); }
  uint64_t getElement(unsigned I) const {
    assert(I < Elements.size() && "Index out of range");
    return Elements[I];
  }

  /// \brief Return whether this is a piece of an aggregate variable.
  bool isBitPiece() const;

  /// \brief Return the offset of this piece in bits.
  uint64_t getBitPieceOffset() const;

  /// \brief Return the size of this piece in bits.
  uint64_t getBitPieceSize() const;

  typedef ArrayRef<uint64_t>::iterator element_iterator;
  element_iterator elements_begin() const { return getElements().begin(); }
  element_iterator elements_end() const { return getElements().end(); }

  /// \brief A lightweight wrapper around an expression operand.
  ///
  /// TODO: Store arguments directly and change \a DIExpression to store a
  /// range of these.
  class ExprOperand {
    const uint64_t *Op;

  public:
    explicit ExprOperand(const uint64_t *Op) : Op(Op) {}

    const uint64_t *get() const { return Op; }

    /// \brief Get the operand code.
    uint64_t getOp() const { return *Op; }

    /// \brief Get an argument to the operand.
    ///
    /// Never returns the operand itself.
    uint64_t getArg(unsigned I) const { return Op[I + 1]; }

    unsigned getNumArgs() const { return getSize() - 1; }

    /// \brief Return the size of the operand.
    ///
    /// Return the number of elements in the operand (1 + args).
    unsigned getSize() const;
  };

  /// \brief An iterator for expression operands.
  class expr_op_iterator
      : public std::iterator<std::input_iterator_tag, ExprOperand> {
    ExprOperand Op;

  public:
    explicit expr_op_iterator(element_iterator I) : Op(I) {}

    element_iterator getBase() const { return Op.get(); }
    const ExprOperand &operator*() const { return Op; }
    const ExprOperand *operator->() const { return &Op; }

    expr_op_iterator &operator++() {
      increment();
      return *this;
    }
    expr_op_iterator operator++(int) {
      expr_op_iterator T(*this);
      increment();
      return T;
    }

    /// \brief Get the next iterator.
    ///
    /// \a std::next() doesn't work because this is technically an
    /// input_iterator, but it's a perfectly valid operation.  This is an
    /// accessor to provide the same functionality.
    expr_op_iterator getNext() const { return ++expr_op_iterator(*this); }

    bool operator==(const expr_op_iterator &X) const {
      return getBase() == X.getBase();
    }
    bool operator!=(const expr_op_iterator &X) const {
      return getBase() != X.getBase();
    }

  private:
    void increment() { Op = ExprOperand(getBase() + Op.getSize()); }
  };

  /// \brief Visit the elements via ExprOperand wrappers.
  ///
  /// These range iterators visit elements through \a ExprOperand wrappers.
  /// This is not guaranteed to be a valid range unless \a isValid() gives \c
  /// true.
  ///
  /// \pre \a isValid() gives \c true.
  /// @{
  expr_op_iterator expr_op_begin() const {
    return expr_op_iterator(elements_begin());
  }
  expr_op_iterator expr_op_end() const {
    return expr_op_iterator(elements_end());
  }
  /// @}

  bool isValid() const;

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DIExpressionKind;
  }

  /// \brief Is the first element a DW_OP_deref?.
  bool startsWithDeref() const {
    return getNumElements() > 0 && getElement(0) == dwarf::DW_OP_deref;
  }
};

class DIObjCProperty : public DINode {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Line;
  unsigned Attributes;

  DIObjCProperty(LLVMContext &C, StorageType Storage, unsigned Line,
                 unsigned Attributes, ArrayRef<Metadata *> Ops)
      : DINode(C, DIObjCPropertyKind, Storage, dwarf::DW_TAG_APPLE_property,
               Ops),
        Line(Line), Attributes(Attributes) {}
  ~DIObjCProperty() = default;

  static DIObjCProperty *
  getImpl(LLVMContext &Context, StringRef Name, DIFile *File, unsigned Line,
          StringRef GetterName, StringRef SetterName, unsigned Attributes,
          DITypeRef Type, StorageType Storage, bool ShouldCreate = true) {
    return getImpl(Context, getCanonicalMDString(Context, Name), File, Line,
                   getCanonicalMDString(Context, GetterName),
                   getCanonicalMDString(Context, SetterName), Attributes, Type,
                   Storage, ShouldCreate);
  }
  static DIObjCProperty *getImpl(LLVMContext &Context, MDString *Name,
                                 Metadata *File, unsigned Line,
                                 MDString *GetterName, MDString *SetterName,
                                 unsigned Attributes, Metadata *Type,
                                 StorageType Storage, bool ShouldCreate = true);

  TempDIObjCProperty cloneImpl() const {
    return getTemporary(getContext(), getName(), getFile(), getLine(),
                        getGetterName(), getSetterName(), getAttributes(),
                        getType());
  }

public:
  DEFINE_MDNODE_GET(DIObjCProperty,
                    (StringRef Name, DIFile *File, unsigned Line,
                     StringRef GetterName, StringRef SetterName,
                     unsigned Attributes, DITypeRef Type),
                    (Name, File, Line, GetterName, SetterName, Attributes,
                     Type))
  DEFINE_MDNODE_GET(DIObjCProperty,
                    (MDString * Name, Metadata *File, unsigned Line,
                     MDString *GetterName, MDString *SetterName,
                     unsigned Attributes, Metadata *Type),
                    (Name, File, Line, GetterName, SetterName, Attributes,
                     Type))

  TempDIObjCProperty clone() const { return cloneImpl(); }

  unsigned getLine() const { return Line; }
  unsigned getAttributes() const { return Attributes; }
  StringRef getName() const { return getStringOperand(0); }
  DIFile *getFile() const { return cast_or_null<DIFile>(getRawFile()); }
  StringRef getGetterName() const { return getStringOperand(2); }
  StringRef getSetterName() const { return getStringOperand(3); }
  DITypeRef getType() const { return DITypeRef(getRawType()); }

  StringRef getFilename() const {
    if (auto *F = getFile())
      return F->getFilename();
    return "";
  }
  StringRef getDirectory() const {
    if (auto *F = getFile())
      return F->getDirectory();
    return "";
  }

  MDString *getRawName() const { return getOperandAs<MDString>(0); }
  Metadata *getRawFile() const { return getOperand(1); }
  MDString *getRawGetterName() const { return getOperandAs<MDString>(2); }
  MDString *getRawSetterName() const { return getOperandAs<MDString>(3); }
  Metadata *getRawType() const { return getOperand(4); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DIObjCPropertyKind;
  }
};

/// \brief An imported module (C++ using directive or similar).
class DIImportedEntity : public DINode {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Line;

  DIImportedEntity(LLVMContext &C, StorageType Storage, unsigned Tag,
                   unsigned Line, ArrayRef<Metadata *> Ops)
      : DINode(C, DIImportedEntityKind, Storage, Tag, Ops), Line(Line) {}
  ~DIImportedEntity() = default;

  static DIImportedEntity *getImpl(LLVMContext &Context, unsigned Tag,
                                   DIScope *Scope, DINodeRef Entity,
                                   unsigned Line, StringRef Name,
                                   StorageType Storage,
                                   bool ShouldCreate = true) {
    return getImpl(Context, Tag, Scope, Entity, Line,
                   getCanonicalMDString(Context, Name), Storage, ShouldCreate);
  }
  static DIImportedEntity *getImpl(LLVMContext &Context, unsigned Tag,
                                   Metadata *Scope, Metadata *Entity,
                                   unsigned Line, MDString *Name,
                                   StorageType Storage,
                                   bool ShouldCreate = true);

  TempDIImportedEntity cloneImpl() const {
    return getTemporary(getContext(), getTag(), getScope(), getEntity(),
                        getLine(), getName());
  }

public:
  DEFINE_MDNODE_GET(DIImportedEntity,
                    (unsigned Tag, DIScope *Scope, DINodeRef Entity,
                     unsigned Line, StringRef Name = ""),
                    (Tag, Scope, Entity, Line, Name))
  DEFINE_MDNODE_GET(DIImportedEntity,
                    (unsigned Tag, Metadata *Scope, Metadata *Entity,
                     unsigned Line, MDString *Name),
                    (Tag, Scope, Entity, Line, Name))

  TempDIImportedEntity clone() const { return cloneImpl(); }

  unsigned getLine() const { return Line; }
  DIScope *getScope() const { return cast_or_null<DIScope>(getRawScope()); }
  DINodeRef getEntity() const { return DINodeRef(getRawEntity()); }
  StringRef getName() const { return getStringOperand(2); }

  Metadata *getRawScope() const { return getOperand(0); }
  Metadata *getRawEntity() const { return getOperand(1); }
  MDString *getRawName() const { return getOperandAs<MDString>(2); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DIImportedEntityKind;
  }
};

/// \brief Macro Info DWARF-like metadata node.
///
/// A metadata node with a DWARF macro info (i.e., a constant named
/// \c DW_MACINFO_*, defined in llvm/Support/Dwarf.h).  Called \a DIMacroNode
/// because it's potentially used for non-DWARF output.
class DIMacroNode : public MDNode {
  friend class LLVMContextImpl;
  friend class MDNode;

protected:
  DIMacroNode(LLVMContext &C, unsigned ID, StorageType Storage, unsigned MIType,
              ArrayRef<Metadata *> Ops1, ArrayRef<Metadata *> Ops2 = None)
      : MDNode(C, ID, Storage, Ops1, Ops2) {
    assert(MIType < 1u << 16);
    SubclassData16 = MIType;
  }
  ~DIMacroNode() = default;

  template <class Ty> Ty *getOperandAs(unsigned I) const {
    return cast_or_null<Ty>(getOperand(I));
  }

  StringRef getStringOperand(unsigned I) const {
    if (auto *S = getOperandAs<MDString>(I))
      return S->getString();
    return StringRef();
  }

  static MDString *getCanonicalMDString(LLVMContext &Context, StringRef S) {
    if (S.empty())
      return nullptr;
    return MDString::get(Context, S);
  }

public:
  unsigned getMacinfoType() const { return SubclassData16; }

  static bool classof(const Metadata *MD) {
    switch (MD->getMetadataID()) {
    default:
      return false;
    case DIMacroKind:
    case DIMacroFileKind:
      return true;
    }
  }
};

class DIMacro : public DIMacroNode {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Line;

  DIMacro(LLVMContext &C, StorageType Storage, unsigned MIType, unsigned Line,
          ArrayRef<Metadata *> Ops)
      : DIMacroNode(C, DIMacroKind, Storage, MIType, Ops), Line(Line) {}
  ~DIMacro() = default;

  static DIMacro *getImpl(LLVMContext &Context, unsigned MIType, unsigned Line,
                          StringRef Name, StringRef Value, StorageType Storage,
                          bool ShouldCreate = true) {
    return getImpl(Context, MIType, Line, getCanonicalMDString(Context, Name),
                   getCanonicalMDString(Context, Value), Storage, ShouldCreate);
  }
  static DIMacro *getImpl(LLVMContext &Context, unsigned MIType, unsigned Line,
                          MDString *Name, MDString *Value, StorageType Storage,
                          bool ShouldCreate = true);

  TempDIMacro cloneImpl() const {
    return getTemporary(getContext(), getMacinfoType(), getLine(), getName(),
                        getValue());
  }

public:
  DEFINE_MDNODE_GET(DIMacro, (unsigned MIType, unsigned Line, StringRef Name,
                              StringRef Value = ""),
                    (MIType, Line, Name, Value))
  DEFINE_MDNODE_GET(DIMacro, (unsigned MIType, unsigned Line, MDString *Name,
                              MDString *Value),
                    (MIType, Line, Name, Value))

  TempDIMacro clone() const { return cloneImpl(); }

  unsigned getLine() const { return Line; }

  StringRef getName() const { return getStringOperand(0); }
  StringRef getValue() const { return getStringOperand(1); }

  MDString *getRawName() const { return getOperandAs<MDString>(0); }
  MDString *getRawValue() const { return getOperandAs<MDString>(1); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DIMacroKind;
  }
};

class DIMacroFile : public DIMacroNode {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Line;

  DIMacroFile(LLVMContext &C, StorageType Storage, unsigned MIType,
              unsigned Line, ArrayRef<Metadata *> Ops)
      : DIMacroNode(C, DIMacroFileKind, Storage, MIType, Ops), Line(Line) {}
  ~DIMacroFile() = default;

  static DIMacroFile *getImpl(LLVMContext &Context, unsigned MIType,
                              unsigned Line, DIFile *File,
                              DIMacroNodeArray Elements, StorageType Storage,
                              bool ShouldCreate = true) {
    return getImpl(Context, MIType, Line, static_cast<Metadata *>(File),
                   Elements.get(), Storage, ShouldCreate);
  }

  static DIMacroFile *getImpl(LLVMContext &Context, unsigned MIType,
                              unsigned Line, Metadata *File, Metadata *Elements,
                              StorageType Storage, bool ShouldCreate = true);

  TempDIMacroFile cloneImpl() const {
    return getTemporary(getContext(), getMacinfoType(), getLine(), getFile(),
                        getElements());
  }

public:
  DEFINE_MDNODE_GET(DIMacroFile, (unsigned MIType, unsigned Line, DIFile *File,
                                  DIMacroNodeArray Elements),
                    (MIType, Line, File, Elements))
  DEFINE_MDNODE_GET(DIMacroFile, (unsigned MIType, unsigned Line,
                                  Metadata *File, Metadata *Elements),
                    (MIType, Line, File, Elements))

  TempDIMacroFile clone() const { return cloneImpl(); }

  void replaceElements(DIMacroNodeArray Elements) {
#ifndef NDEBUG
    for (DIMacroNode *Op : getElements())
      assert(is_contained(Elements->operands(), Op) &&
             "Lost a macro node during macro node list replacement");
#endif
    replaceOperandWith(1, Elements.get());
  }

  unsigned getLine() const { return Line; }
  DIFile *getFile() const { return cast_or_null<DIFile>(getRawFile()); }

  DIMacroNodeArray getElements() const {
    return cast_or_null<MDTuple>(getRawElements());
  }

  Metadata *getRawFile() const { return getOperand(0); }
  Metadata *getRawElements() const { return getOperand(1); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == DIMacroFileKind;
  }
};

} // end namespace llvm

#undef DEFINE_MDNODE_GET_UNPACK_IMPL
#undef DEFINE_MDNODE_GET_UNPACK
#undef DEFINE_MDNODE_GET

#endif
