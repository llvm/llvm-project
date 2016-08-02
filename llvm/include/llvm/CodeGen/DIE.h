//===--- lib/CodeGen/DIE.h - DWARF Info Entries -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Data structures for DWARF info entries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DIE_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DIE_H

#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/DwarfStringPoolEntry.h"
#include "llvm/Support/Dwarf.h"
#include <vector>

namespace llvm {
class AsmPrinter;
class MCExpr;
class MCSymbol;
class raw_ostream;
class DwarfTypeUnit;

// AsmStreamerBase - A base abstract interface class defines methods that
// can be implemented to stream objects or can be implemented to
// calculate the size of the streamed objects.
// The derived classes will use an AsmPrinter to implement the methods.
//
// TODO: complete this interface and use it to merge EmitValue and SizeOf
//       methods in the DIE classes below.
class AsmStreamerBase {
protected:
  const AsmPrinter *AP;
  AsmStreamerBase(const AsmPrinter *AP) : AP(AP) {}

public:
  virtual ~AsmStreamerBase() {}
  virtual unsigned emitULEB128(uint64_t Value, const char *Desc = nullptr,
                               unsigned PadTo = 0) = 0;
  virtual unsigned emitInt8(unsigned char Value) = 0;
  virtual unsigned emitBytes(StringRef Data) = 0;
};

/// EmittingAsmStreamer - Implements AbstractAsmStreamer to stream objects.
/// Notice that the return value is not the actual size of the streamed object.
/// For size calculation use SizeReporterAsmStreamer.
class EmittingAsmStreamer : public AsmStreamerBase {
public:
  EmittingAsmStreamer(const AsmPrinter *AP) : AsmStreamerBase(AP) {}
  unsigned emitULEB128(uint64_t Value, const char *Desc = nullptr,
                       unsigned PadTo = 0) override;
  unsigned emitInt8(unsigned char Value) override;
  unsigned emitBytes(StringRef Data) override;
};

/// SizeReporterAsmStreamer - Only reports the size of the streamed objects.
class SizeReporterAsmStreamer : public AsmStreamerBase {
public:
  SizeReporterAsmStreamer(const AsmPrinter *AP) : AsmStreamerBase(AP) {}
  unsigned emitULEB128(uint64_t Value, const char *Desc = nullptr,
                       unsigned PadTo = 0) override;
  unsigned emitInt8(unsigned char Value) override;
  unsigned emitBytes(StringRef Data) override;
};

//===--------------------------------------------------------------------===//
/// DIEAbbrevData - Dwarf abbreviation data, describes one attribute of a
/// Dwarf abbreviation.
class DIEAbbrevData {
  /// Attribute - Dwarf attribute code.
  ///
  dwarf::Attribute Attribute;

  /// Form - Dwarf form code.
  ///
  dwarf::Form Form;

public:
  DIEAbbrevData(dwarf::Attribute A, dwarf::Form F) : Attribute(A), Form(F) {}

  // Accessors.
  dwarf::Attribute getAttribute() const { return Attribute; }
  dwarf::Form getForm() const { return Form; }

  /// Profile - Used to gather unique data for the abbreviation folding set.
  ///
  void Profile(FoldingSetNodeID &ID) const;
};

//===--------------------------------------------------------------------===//
/// DIEAbbrev - Dwarf abbreviation, describes the organization of a debug
/// information object.
class DIEAbbrev : public FoldingSetNode {
  /// Unique number for node.
  ///
  unsigned Number;

  /// Tag - Dwarf tag code.
  ///
  dwarf::Tag Tag;

  /// Children - Whether or not this node has children.
  ///
  // This cheats a bit in all of the uses since the values in the standard
  // are 0 and 1 for no children and children respectively.
  bool Children;

  /// Data - Raw data bytes for abbreviation.
  ///
  SmallVector<DIEAbbrevData, 12> Data;

public:
  DIEAbbrev(dwarf::Tag T, bool C) : Tag(T), Children(C), Data() {}

  // Accessors.
  dwarf::Tag getTag() const { return Tag; }
  unsigned getNumber() const { return Number; }
  bool hasChildren() const { return Children; }
  const SmallVectorImpl<DIEAbbrevData> &getData() const { return Data; }
  void setChildrenFlag(bool hasChild) { Children = hasChild; }
  void setNumber(unsigned N) { Number = N; }

  /// AddAttribute - Adds another set of attribute information to the
  /// abbreviation.
  void AddAttribute(dwarf::Attribute Attribute, dwarf::Form Form) {
    Data.push_back(DIEAbbrevData(Attribute, Form));
  }

  /// Profile - Used to gather unique data for the abbreviation folding set.
  ///
  void Profile(FoldingSetNodeID &ID) const;

  /// Emit - Print the abbreviation using the specified asm printer.
  ///
  void Emit(const AsmPrinter *AP) const;

  void print(raw_ostream &O);
  void dump();
};

//===--------------------------------------------------------------------===//
/// DIEInteger - An integer value DIE.
///
class DIEInteger {
  uint64_t Integer;

public:
  explicit DIEInteger(uint64_t I) : Integer(I) {}

  /// BestForm - Choose the best form for integer.
  ///
  static dwarf::Form BestForm(bool IsSigned, uint64_t Int) {
    if (IsSigned) {
      const int64_t SignedInt = Int;
      if ((char)Int == SignedInt)
        return dwarf::DW_FORM_data1;
      if ((short)Int == SignedInt)
        return dwarf::DW_FORM_data2;
      if ((int)Int == SignedInt)
        return dwarf::DW_FORM_data4;
    } else {
      if ((unsigned char)Int == Int)
        return dwarf::DW_FORM_data1;
      if ((unsigned short)Int == Int)
        return dwarf::DW_FORM_data2;
      if ((unsigned int)Int == Int)
        return dwarf::DW_FORM_data4;
    }
    return dwarf::DW_FORM_data8;
  }

  uint64_t getValue() const { return Integer; }
  void setValue(uint64_t Val) { Integer = Val; }

  void EmitValue(const AsmPrinter *AP, dwarf::Form Form) const;
  unsigned SizeOf(const AsmPrinter *AP, dwarf::Form Form) const;

  void print(raw_ostream &O) const;
};

//===--------------------------------------------------------------------===//
/// DIEExpr - An expression DIE.
//
class DIEExpr {
  const MCExpr *Expr;

public:
  explicit DIEExpr(const MCExpr *E) : Expr(E) {}

  /// getValue - Get MCExpr.
  ///
  const MCExpr *getValue() const { return Expr; }

  void EmitValue(const AsmPrinter *AP, dwarf::Form Form) const;
  unsigned SizeOf(const AsmPrinter *AP, dwarf::Form Form) const;

  void print(raw_ostream &O) const;
};

//===--------------------------------------------------------------------===//
/// DIELabel - A label DIE.
//
class DIELabel {
  const MCSymbol *Label;

public:
  explicit DIELabel(const MCSymbol *L) : Label(L) {}

  /// getValue - Get MCSymbol.
  ///
  const MCSymbol *getValue() const { return Label; }

  void EmitValue(const AsmPrinter *AP, dwarf::Form Form) const;
  unsigned SizeOf(const AsmPrinter *AP, dwarf::Form Form) const;

  void print(raw_ostream &O) const;
};

//===--------------------------------------------------------------------===//
/// DIEDelta - A simple label difference DIE.
///
class DIEDelta {
  const MCSymbol *LabelHi;
  const MCSymbol *LabelLo;

public:
  DIEDelta(const MCSymbol *Hi, const MCSymbol *Lo) : LabelHi(Hi), LabelLo(Lo) {}

  void EmitValue(const AsmPrinter *AP, dwarf::Form Form) const;
  unsigned SizeOf(const AsmPrinter *AP, dwarf::Form Form) const;

  void print(raw_ostream &O) const;
};

//===--------------------------------------------------------------------===//
/// DIEString - A container for string values.
///
class DIEString {
  DwarfStringPoolEntryRef S;

public:
  DIEString(DwarfStringPoolEntryRef S) : S(S) {}

  /// getString - Grab the string out of the object.
  StringRef getString() const { return S.getString(); }

  void EmitValue(const AsmPrinter *AP, dwarf::Form Form) const;
  unsigned SizeOf(const AsmPrinter *AP, dwarf::Form Form) const;

  void print(raw_ostream &O) const;
};

//===--------------------------------------------------------------------===//
/// DIEEntry - A pointer to another debug information entry.  An instance of
/// this class can also be used as a proxy for a debug information entry not
/// yet defined (ie. types.)
class DIE;
class DIEEntry {
  DIE *Entry;

  DIEEntry() = delete;

public:
  explicit DIEEntry(DIE &E) : Entry(&E) {}

  DIE &getEntry() const { return *Entry; }

  /// Returns size of a ref_addr entry.
  static unsigned getRefAddrSize(const AsmPrinter *AP);

  void EmitValue(const AsmPrinter *AP, dwarf::Form Form) const;
  unsigned SizeOf(const AsmPrinter *AP, dwarf::Form Form) const {
    return Form == dwarf::DW_FORM_ref_addr ? getRefAddrSize(AP)
                                           : sizeof(int32_t);
  }

  void print(raw_ostream &O) const;
};

//===--------------------------------------------------------------------===//
/// \brief A signature reference to a type unit.
class DIETypeSignature {
  const DwarfTypeUnit *Unit;

  DIETypeSignature() = delete;

public:
  explicit DIETypeSignature(const DwarfTypeUnit &Unit) : Unit(&Unit) {}

  void EmitValue(const AsmPrinter *AP, dwarf::Form Form) const;
  unsigned SizeOf(const AsmPrinter *AP, dwarf::Form Form) const {
    assert(Form == dwarf::DW_FORM_ref_sig8);
    return 8;
  }

  void print(raw_ostream &O) const;
};

//===--------------------------------------------------------------------===//
/// DIELocList - Represents a pointer to a location list in the debug_loc
/// section.
//
class DIELocList {
  // Index into the .debug_loc vector.
  size_t Index;

public:
  DIELocList(size_t I) : Index(I) {}

  /// getValue - Grab the current index out.
  size_t getValue() const { return Index; }

  void EmitValue(const AsmPrinter *AP, dwarf::Form Form) const;
  unsigned SizeOf(const AsmPrinter *AP, dwarf::Form Form) const;

  void print(raw_ostream &O) const;
};

//===--------------------------------------------------------------------===//
/// DIEValue - A debug information entry value. Some of these roughly correlate
/// to DWARF attribute classes.
///
class DIEBlock;
class DIELoc;
class DIEValue {
public:
  enum Type {
    isNone,
#define HANDLE_DIEVALUE(T) is##T,
#include "llvm/CodeGen/DIEValue.def"
  };

private:
  /// Ty - Type of data stored in the value.
  ///
  Type Ty = isNone;
  dwarf::Attribute Attribute = (dwarf::Attribute)0;
  dwarf::Form Form = (dwarf::Form)0;

  /// Storage for the value.
  ///
  /// All values that aren't standard layout (or are larger than 8 bytes)
  /// should be stored by reference instead of by value.
  typedef AlignedCharArrayUnion<DIEInteger, DIEString, DIEExpr, DIELabel,
                                DIEDelta *, DIEEntry, DIETypeSignature,
                                DIEBlock *, DIELoc *, DIELocList> ValTy;
  static_assert(sizeof(ValTy) <= sizeof(uint64_t) ||
                    sizeof(ValTy) <= sizeof(void *),
                "Expected all large types to be stored via pointer");

  /// Underlying stored value.
  ValTy Val;

  template <class T> void construct(T V) {
    static_assert(std::is_standard_layout<T>::value ||
                      std::is_pointer<T>::value,
                  "Expected standard layout or pointer");
    new (reinterpret_cast<void *>(Val.buffer)) T(V);
  }

  template <class T> T *get() { return reinterpret_cast<T *>(Val.buffer); }
  template <class T> const T *get() const {
    return reinterpret_cast<const T *>(Val.buffer);
  }
  template <class T> void destruct() { get<T>()->~T(); }

  /// Destroy the underlying value.
  ///
  /// This should get optimized down to a no-op.  We could skip it if we could
  /// add a static assert on \a std::is_trivially_copyable(), but we currently
  /// support versions of GCC that don't understand that.
  void destroyVal() {
    switch (Ty) {
    case isNone:
      return;
#define HANDLE_DIEVALUE_SMALL(T)                                               \
  case is##T:                                                                  \
    destruct<DIE##T>();
    return;
#define HANDLE_DIEVALUE_LARGE(T)                                               \
  case is##T:                                                                  \
    destruct<const DIE##T *>();
    return;
#include "llvm/CodeGen/DIEValue.def"
    }
  }

  /// Copy the underlying value.
  ///
  /// This should get optimized down to a simple copy.  We need to actually
  /// construct the value, rather than calling memcpy, to satisfy strict
  /// aliasing rules.
  void copyVal(const DIEValue &X) {
    switch (Ty) {
    case isNone:
      return;
#define HANDLE_DIEVALUE_SMALL(T)                                               \
  case is##T:                                                                  \
    construct<DIE##T>(*X.get<DIE##T>());                                       \
    return;
#define HANDLE_DIEVALUE_LARGE(T)                                               \
  case is##T:                                                                  \
    construct<const DIE##T *>(*X.get<const DIE##T *>());                       \
    return;
#include "llvm/CodeGen/DIEValue.def"
    }
  }

public:
  DIEValue() = default;
  DIEValue(const DIEValue &X) : Ty(X.Ty), Attribute(X.Attribute), Form(X.Form) {
    copyVal(X);
  }
  DIEValue &operator=(const DIEValue &X) {
    destroyVal();
    Ty = X.Ty;
    Attribute = X.Attribute;
    Form = X.Form;
    copyVal(X);
    return *this;
  }
  ~DIEValue() { destroyVal(); }

#define HANDLE_DIEVALUE_SMALL(T)                                               \
  DIEValue(dwarf::Attribute Attribute, dwarf::Form Form, const DIE##T &V)      \
      : Ty(is##T), Attribute(Attribute), Form(Form) {                          \
    construct<DIE##T>(V);                                                      \
  }
#define HANDLE_DIEVALUE_LARGE(T)                                               \
  DIEValue(dwarf::Attribute Attribute, dwarf::Form Form, const DIE##T *V)      \
      : Ty(is##T), Attribute(Attribute), Form(Form) {                          \
    assert(V && "Expected valid value");                                       \
    construct<const DIE##T *>(V);                                              \
  }
#include "llvm/CodeGen/DIEValue.def"

  // Accessors
  Type getType() const { return Ty; }
  dwarf::Attribute getAttribute() const { return Attribute; }
  dwarf::Form getForm() const { return Form; }
  explicit operator bool() const { return Ty; }

#define HANDLE_DIEVALUE_SMALL(T)                                               \
  const DIE##T &getDIE##T() const {                                            \
    assert(getType() == is##T && "Expected " #T);                              \
    return *get<DIE##T>();                                                     \
  }
#define HANDLE_DIEVALUE_LARGE(T)                                               \
  const DIE##T &getDIE##T() const {                                            \
    assert(getType() == is##T && "Expected " #T);                              \
    return **get<const DIE##T *>();                                            \
  }
#include "llvm/CodeGen/DIEValue.def"

  /// EmitValue - Emit value via the Dwarf writer.
  ///
  void EmitValue(const AsmPrinter *AP) const;

  /// SizeOf - Return the size of a value in bytes.
  ///
  unsigned SizeOf(const AsmPrinter *AP) const;

  void print(raw_ostream &O) const;
  void dump() const;
};

struct IntrusiveBackListNode {
  PointerIntPair<IntrusiveBackListNode *, 1> Next;
  IntrusiveBackListNode() : Next(this, true) {}

  IntrusiveBackListNode *getNext() const {
    return Next.getInt() ? nullptr : Next.getPointer();
  }
};

struct IntrusiveBackListBase {
  typedef IntrusiveBackListNode Node;
  Node *Last = nullptr;

  bool empty() const { return !Last; }
  void push_back(Node &N) {
    assert(N.Next.getPointer() == &N && "Expected unlinked node");
    assert(N.Next.getInt() == true && "Expected unlinked node");

    if (Last) {
      N.Next = Last->Next;
      Last->Next.setPointerAndInt(&N, false);
    }
    Last = &N;
  }
};

template <class T> class IntrusiveBackList : IntrusiveBackListBase {
public:
  using IntrusiveBackListBase::empty;
  void push_back(T &N) { IntrusiveBackListBase::push_back(N); }
  T &back() { return *static_cast<T *>(Last); }
  const T &back() const { return *static_cast<T *>(Last); }

  class const_iterator;
  class iterator
      : public iterator_facade_base<iterator, std::forward_iterator_tag, T> {
    friend class const_iterator;
    Node *N = nullptr;

  public:
    iterator() = default;
    explicit iterator(T *N) : N(N) {}

    iterator &operator++() {
      N = N->getNext();
      return *this;
    }

    explicit operator bool() const { return N; }
    T &operator*() const { return *static_cast<T *>(N); }

    bool operator==(const iterator &X) const { return N == X.N; }
    bool operator!=(const iterator &X) const { return N != X.N; }
  };

  class const_iterator
      : public iterator_facade_base<const_iterator, std::forward_iterator_tag,
                                    const T> {
    const Node *N = nullptr;

  public:
    const_iterator() = default;
    // Placate MSVC by explicitly scoping 'iterator'.
    const_iterator(typename IntrusiveBackList<T>::iterator X) : N(X.N) {}
    explicit const_iterator(const T *N) : N(N) {}

    const_iterator &operator++() {
      N = N->getNext();
      return *this;
    }

    explicit operator bool() const { return N; }
    const T &operator*() const { return *static_cast<const T *>(N); }

    bool operator==(const const_iterator &X) const { return N == X.N; }
    bool operator!=(const const_iterator &X) const { return N != X.N; }
  };

  iterator begin() {
    return Last ? iterator(static_cast<T *>(Last->Next.getPointer())) : end();
  }
  const_iterator begin() const {
    return const_cast<IntrusiveBackList *>(this)->begin();
  }
  iterator end() { return iterator(); }
  const_iterator end() const { return const_iterator(); }

  static iterator toIterator(T &N) { return iterator(&N); }
  static const_iterator toIterator(const T &N) { return const_iterator(&N); }
};

/// A list of DIE values.
///
/// This is a singly-linked list, but instead of reversing the order of
/// insertion, we keep a pointer to the back of the list so we can push in
/// order.
///
/// There are two main reasons to choose a linked list over a customized
/// vector-like data structure.
///
///  1. For teardown efficiency, we want DIEs to be BumpPtrAllocated.  Using a
///     linked list here makes this way easier to accomplish.
///  2. Carrying an extra pointer per \a DIEValue isn't expensive.  45% of DIEs
///     have 2 or fewer values, and 90% have 5 or fewer.  A vector would be
///     over-allocated by 50% on average anyway, the same cost as the
///     linked-list node.
class DIEValueList {
  struct Node : IntrusiveBackListNode {
    DIEValue V;
    explicit Node(DIEValue V) : V(V) {}
  };

  typedef IntrusiveBackList<Node> ListTy;
  ListTy List;

public:
  class const_value_iterator;
  class value_iterator
      : public iterator_adaptor_base<value_iterator, ListTy::iterator,
                                     std::forward_iterator_tag, DIEValue> {
    friend class const_value_iterator;
    typedef iterator_adaptor_base<value_iterator, ListTy::iterator,
                                  std::forward_iterator_tag,
                                  DIEValue> iterator_adaptor;

  public:
    value_iterator() = default;
    explicit value_iterator(ListTy::iterator X) : iterator_adaptor(X) {}

    explicit operator bool() const { return bool(wrapped()); }
    DIEValue &operator*() const { return wrapped()->V; }
  };

  class const_value_iterator : public iterator_adaptor_base<
                                   const_value_iterator, ListTy::const_iterator,
                                   std::forward_iterator_tag, const DIEValue> {
    typedef iterator_adaptor_base<const_value_iterator, ListTy::const_iterator,
                                  std::forward_iterator_tag,
                                  const DIEValue> iterator_adaptor;

  public:
    const_value_iterator() = default;
    const_value_iterator(DIEValueList::value_iterator X)
        : iterator_adaptor(X.wrapped()) {}
    explicit const_value_iterator(ListTy::const_iterator X)
        : iterator_adaptor(X) {}

    explicit operator bool() const { return bool(wrapped()); }
    const DIEValue &operator*() const { return wrapped()->V; }
  };

  typedef iterator_range<value_iterator> value_range;
  typedef iterator_range<const_value_iterator> const_value_range;

  value_iterator addValue(BumpPtrAllocator &Alloc, DIEValue V) {
    List.push_back(*new (Alloc) Node(V));
    return value_iterator(ListTy::toIterator(List.back()));
  }
  template <class T>
  value_iterator addValue(BumpPtrAllocator &Alloc, dwarf::Attribute Attribute,
                    dwarf::Form Form, T &&Value) {
    return addValue(Alloc, DIEValue(Attribute, Form, std::forward<T>(Value)));
  }

  value_range values() {
    return llvm::make_range(value_iterator(List.begin()),
                            value_iterator(List.end()));
  }
  const_value_range values() const {
    return llvm::make_range(const_value_iterator(List.begin()),
                            const_value_iterator(List.end()));
  }
};

//===--------------------------------------------------------------------===//
/// DIE - A structured debug information entry.  Has an abbreviation which
/// describes its organization.
class DIE : IntrusiveBackListNode, public DIEValueList {
  friend class IntrusiveBackList<DIE>;

  /// Offset - Offset in debug info section.
  ///
  unsigned Offset;

  /// Size - Size of instance + children.
  ///
  unsigned Size;

  unsigned AbbrevNumber = ~0u;

  /// Tag - Dwarf tag code.
  ///
  dwarf::Tag Tag = (dwarf::Tag)0;

  /// Children DIEs.
  IntrusiveBackList<DIE> Children;

  DIE *Parent = nullptr;

  DIE() = delete;
  explicit DIE(dwarf::Tag Tag) : Offset(0), Size(0), Tag(Tag) {}

public:
  static DIE *get(BumpPtrAllocator &Alloc, dwarf::Tag Tag) {
    return new (Alloc) DIE(Tag);
  }

  // Accessors.
  unsigned getAbbrevNumber() const { return AbbrevNumber; }
  dwarf::Tag getTag() const { return Tag; }
  unsigned getOffset() const { return Offset; }
  unsigned getSize() const { return Size; }
  bool hasChildren() const { return !Children.empty(); }

  typedef IntrusiveBackList<DIE>::iterator child_iterator;
  typedef IntrusiveBackList<DIE>::const_iterator const_child_iterator;
  typedef iterator_range<child_iterator> child_range;
  typedef iterator_range<const_child_iterator> const_child_range;

  child_range children() {
    return llvm::make_range(Children.begin(), Children.end());
  }
  const_child_range children() const {
    return llvm::make_range(Children.begin(), Children.end());
  }

  DIE *getParent() const { return Parent; }

  /// Generate the abbreviation for this DIE.
  ///
  /// Calculate the abbreviation for this, which should be uniqued and
  /// eventually used to call \a setAbbrevNumber().
  DIEAbbrev generateAbbrev() const;

  /// Set the abbreviation number for this DIE.
  void setAbbrevNumber(unsigned I) { AbbrevNumber = I; }

  /// Climb up the parent chain to get the compile or type unit DIE this DIE
  /// belongs to.
  const DIE *getUnit() const;
  /// Similar to getUnit, returns null when DIE is not added to an
  /// owner yet.
  const DIE *getUnitOrNull() const;
  void setOffset(unsigned O) { Offset = O; }
  void setSize(unsigned S) { Size = S; }

  /// Add a child to the DIE.
  DIE &addChild(DIE *Child) {
    assert(!Child->getParent() && "Child should be orphaned");
    Child->Parent = this;
    Children.push_back(*Child);
    return Children.back();
  }

  /// Find a value in the DIE with the attribute given.
  ///
  /// Returns a default-constructed DIEValue (where \a DIEValue::getType()
  /// gives \a DIEValue::isNone) if no such attribute exists.
  DIEValue findAttribute(dwarf::Attribute Attribute) const;

  void print(raw_ostream &O, unsigned IndentCount = 0) const;
  void dump();
};

//===--------------------------------------------------------------------===//
/// DIELoc - Represents an expression location.
//
class DIELoc : public DIEValueList {
  mutable unsigned Size; // Size in bytes excluding size header.

public:
  DIELoc() : Size(0) {}

  /// ComputeSize - Calculate the size of the location expression.
  ///
  unsigned ComputeSize(const AsmPrinter *AP) const;

  /// BestForm - Choose the best form for data.
  ///
  dwarf::Form BestForm(unsigned DwarfVersion) const {
    if (DwarfVersion > 3)
      return dwarf::DW_FORM_exprloc;
    // Pre-DWARF4 location expressions were blocks and not exprloc.
    if ((unsigned char)Size == Size)
      return dwarf::DW_FORM_block1;
    if ((unsigned short)Size == Size)
      return dwarf::DW_FORM_block2;
    if ((unsigned int)Size == Size)
      return dwarf::DW_FORM_block4;
    return dwarf::DW_FORM_block;
  }

  void EmitValue(const AsmPrinter *AP, dwarf::Form Form) const;
  unsigned SizeOf(const AsmPrinter *AP, dwarf::Form Form) const;

  void print(raw_ostream &O) const;
};

//===--------------------------------------------------------------------===//
/// DIEBlock - Represents a block of values.
//
class DIEBlock : public DIEValueList {
  mutable unsigned Size; // Size in bytes excluding size header.

public:
  DIEBlock() : Size(0) {}

  /// ComputeSize - Calculate the size of the location expression.
  ///
  unsigned ComputeSize(const AsmPrinter *AP) const;

  /// BestForm - Choose the best form for data.
  ///
  dwarf::Form BestForm() const {
    if ((unsigned char)Size == Size)
      return dwarf::DW_FORM_block1;
    if ((unsigned short)Size == Size)
      return dwarf::DW_FORM_block2;
    if ((unsigned int)Size == Size)
      return dwarf::DW_FORM_block4;
    return dwarf::DW_FORM_block;
  }

  void EmitValue(const AsmPrinter *AP, dwarf::Form Form) const;
  unsigned SizeOf(const AsmPrinter *AP, dwarf::Form Form) const;

  void print(raw_ostream &O) const;
};

} // end llvm namespace

#endif
