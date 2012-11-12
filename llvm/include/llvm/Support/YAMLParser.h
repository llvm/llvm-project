//===--- YAMLParser.h - Simple YAML parser --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This is a YAML 1.2 parser.
//
//  See http://www.yaml.org/spec/1.2/spec.html for the full standard.
//
//  This currently does not implement the following:
//    * Multi-line literal folding.
//    * Tag resolution.
//    * UTF-16.
//    * BOMs anywhere other than the first Unicode scalar value in the file.
//
//  The most important class here is Stream. This represents a YAML stream with
//  0, 1, or many documents.
//
//  SourceMgr sm;
//  StringRef input = getInput();
//  yaml::Stream stream(input, sm);
//
//  for (yaml::document_iterator di = stream.begin(), de = stream.end();
//       di != de; ++di) {
//    yaml::Node *n = di->getRoot();
//    if (n) {
//      // Do something with n...
//    } else
//      break;
//  }
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_YAML_PARSER_H
#define LLVM_SUPPORT_YAML_PARSER_H

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/SMLoc.h"

#include <limits>
#include <utility>

namespace llvm {
class MemoryBuffer;
class SourceMgr;
class raw_ostream;
class Twine;

namespace yaml {

class document_iterator;
class Document;
class Node;
class Scanner;
struct Token;

/// @brief Dump all the tokens in this stream to OS.
/// @returns true if there was an error, false otherwise.
bool dumpTokens(StringRef Input, raw_ostream &);

/// @brief Scans all tokens in input without outputting anything. This is used
///        for benchmarking the tokenizer.
/// @returns true if there was an error, false otherwise.
bool scanTokens(StringRef Input);

/// @brief Escape \a Input for a double quoted scalar.
std::string escape(StringRef Input);

/// @brief This class represents a YAML stream potentially containing multiple
///        documents.
class Stream {
public:
  Stream(StringRef Input, SourceMgr &);
  ~Stream();

  document_iterator begin();
  document_iterator end();
  void skip();
  bool failed();
  bool validate() {
    skip();
    return !failed();
  }

  void printError(Node *N, const Twine &Msg);

private:
  OwningPtr<Scanner> scanner;
  OwningPtr<Document> CurrentDoc;

  friend class Document;

  /// @brief Validate a %YAML x.x directive.
  void handleYAMLDirective(const Token &);
};

/// @brief Abstract base class for all Nodes.
class Node {
public:
  enum NodeKind {
    NK_Null,
    NK_Scalar,
    NK_KeyValue,
    NK_Mapping,
    NK_Sequence,
    NK_Alias
  };

  Node(unsigned int Type, OwningPtr<Document>&, StringRef Anchor);

  /// @brief Get the value of the anchor attached to this node. If it does not
  ///        have one, getAnchor().size() will be 0.
  StringRef getAnchor() const { return Anchor; }

  SMRange getSourceRange() const { return SourceRange; }
  void setSourceRange(SMRange SR) { SourceRange = SR; }

  // These functions forward to Document and Scanner.
  Token &peekNext();
  Token getNext();
  Node *parseBlockNode();
  BumpPtrAllocator &getAllocator();
  void setError(const Twine &Message, Token &Location) const;
  bool failed() const;

  virtual void skip() {}

  unsigned int getType() const { return TypeID; }

  void *operator new ( size_t Size
                     , BumpPtrAllocator &Alloc
                     , size_t Alignment = 16) throw() {
    return Alloc.Allocate(Size, Alignment);
  }

  void operator delete(void *Ptr, BumpPtrAllocator &Alloc, size_t) throw() {
    Alloc.Deallocate(Ptr);
  }

protected:
  OwningPtr<Document> &Doc;
  SMRange SourceRange;

  void operator delete(void *) throw() {}

  virtual ~Node() {}

private:
  unsigned int TypeID;
  StringRef Anchor;
};

/// @brief A null value.
///
/// Example:
///   !!null null
class NullNode : public Node {
public:
  NullNode(OwningPtr<Document> &D) : Node(NK_Null, D, StringRef()) {}

  static inline bool classof(const Node *N) {
    return N->getType() == NK_Null;
  }
};

/// @brief A scalar node is an opaque datum that can be presented as a
///        series of zero or more Unicode scalar values.
///
/// Example:
///   Adena
class ScalarNode : public Node {
public:
  ScalarNode(OwningPtr<Document> &D, StringRef Anchor, StringRef Val)
    : Node(NK_Scalar, D, Anchor)
    , Value(Val) {
    SMLoc Start = SMLoc::getFromPointer(Val.begin());
    SMLoc End = SMLoc::getFromPointer(Val.end() - 1);
    SourceRange = SMRange(Start, End);
  }

  // Return Value without any escaping or folding or other fun YAML stuff. This
  // is the exact bytes that are contained in the file (after conversion to
  // utf8).
  StringRef getRawValue() const { return Value; }

  /// @brief Gets the value of this node as a StringRef.
  ///
  /// @param Storage is used to store the content of the returned StringRef iff
  ///        it requires any modification from how it appeared in the source.
  ///        This happens with escaped characters and multi-line literals.
  StringRef getValue(SmallVectorImpl<char> &Storage) const;

  static inline bool classof(const Node *N) {
    return N->getType() == NK_Scalar;
  }

private:
  StringRef Value;

  StringRef unescapeDoubleQuoted( StringRef UnquotedValue
                                , StringRef::size_type Start
                                , SmallVectorImpl<char> &Storage) const;
};

/// @brief A key and value pair. While not technically a Node under the YAML
///        representation graph, it is easier to treat them this way.
///
/// TODO: Consider making this not a child of Node.
///
/// Example:
///   Section: .text
class KeyValueNode : public Node {
public:
  KeyValueNode(OwningPtr<Document> &D)
    : Node(NK_KeyValue, D, StringRef())
    , Key(0)
    , Value(0)
  {}

  /// @brief Parse and return the key.
  ///
  /// This may be called multiple times.
  ///
  /// @returns The key, or nullptr if failed() == true.
  Node *getKey();

  /// @brief Parse and return the value.
  ///
  /// This may be called multiple times.
  ///
  /// @returns The value, or nullptr if failed() == true.
  Node *getValue();

  virtual void skip() LLVM_OVERRIDE {
    getKey()->skip();
    getValue()->skip();
  }

  static inline bool classof(const Node *N) {
    return N->getType() == NK_KeyValue;
  }

private:
  Node *Key;
  Node *Value;
};

/// @brief This is an iterator abstraction over YAML collections shared by both
///        sequences and maps.
///
/// BaseT must have a ValueT* member named CurrentEntry and a member function
/// increment() which must set CurrentEntry to 0 to create an end iterator.
template <class BaseT, class ValueT>
class basic_collection_iterator
  : public std::iterator<std::forward_iterator_tag, ValueT> {
public:
  basic_collection_iterator() : Base(0) {}
  basic_collection_iterator(BaseT *B) : Base(B) {}

  ValueT *operator ->() const {
    assert(Base && Base->CurrentEntry && "Attempted to access end iterator!");
    return Base->CurrentEntry;
  }

  ValueT &operator *() const {
    assert(Base && Base->CurrentEntry &&
           "Attempted to dereference end iterator!");
    return *Base->CurrentEntry;
  }

  operator ValueT*() const {
    assert(Base && Base->CurrentEntry && "Attempted to access end iterator!");
    return Base->CurrentEntry;
  }

  bool operator !=(const basic_collection_iterator &Other) const {
    if(Base != Other.Base)
      return true;
    return (Base && Other.Base) && Base->CurrentEntry
                                   != Other.Base->CurrentEntry;
  }

  basic_collection_iterator &operator++() {
    assert(Base && "Attempted to advance iterator past end!");
    Base->increment();
    // Create an end iterator.
    if (Base->CurrentEntry == 0)
      Base = 0;
    return *this;
  }

private:
  BaseT *Base;
};

// The following two templates are used for both MappingNode and Sequence Node.
template <class CollectionType>
typename CollectionType::iterator begin(CollectionType &C) {
  assert(C.IsAtBeginning && "You may only iterate over a collection once!");
  C.IsAtBeginning = false;
  typename CollectionType::iterator ret(&C);
  ++ret;
  return ret;
}

template <class CollectionType>
void skip(CollectionType &C) {
  // TODO: support skipping from the middle of a parsed collection ;/
  assert((C.IsAtBeginning || C.IsAtEnd) && "Cannot skip mid parse!");
  if (C.IsAtBeginning)
    for (typename CollectionType::iterator i = begin(C), e = C.end();
                                           i != e; ++i)
      i->skip();
}

/// @brief Represents a YAML map created from either a block map for a flow map.
///
/// This parses the YAML stream as increment() is called.
///
/// Example:
///   Name: _main
///   Scope: Global
class MappingNode : public Node {
public:
  enum MappingType {
    MT_Block,
    MT_Flow,
    MT_Inline ///< An inline mapping node is used for "[key: value]".
  };

  MappingNode(OwningPtr<Document> &D, StringRef Anchor, MappingType MT)
    : Node(NK_Mapping, D, Anchor)
    , Type(MT)
    , IsAtBeginning(true)
    , IsAtEnd(false)
    , CurrentEntry(0)
  {}

  friend class basic_collection_iterator<MappingNode, KeyValueNode>;
  typedef basic_collection_iterator<MappingNode, KeyValueNode> iterator;
  template <class T> friend typename T::iterator yaml::begin(T &);
  template <class T> friend void yaml::skip(T &);

  iterator begin() {
    return yaml::begin(*this);
  }

  iterator end() { return iterator(); }

  virtual void skip() LLVM_OVERRIDE {
    yaml::skip(*this);
  }

  static inline bool classof(const Node *N) {
    return N->getType() == NK_Mapping;
  }

private:
  MappingType Type;
  bool IsAtBeginning;
  bool IsAtEnd;
  KeyValueNode *CurrentEntry;

  void increment();
};

/// @brief Represents a YAML sequence created from either a block sequence for a
///        flow sequence.
///
/// This parses the YAML stream as increment() is called.
///
/// Example:
///   - Hello
///   - World
class SequenceNode : public Node {
public:
  enum SequenceType {
    ST_Block,
    ST_Flow,
    // Use for:
    //
    // key:
    // - val1
    // - val2
    //
    // As a BlockMappingEntry and BlockEnd are not created in this case.
    ST_Indentless
  };

  SequenceNode(OwningPtr<Document> &D, StringRef Anchor, SequenceType ST)
    : Node(NK_Sequence, D, Anchor)
    , SeqType(ST)
    , IsAtBeginning(true)
    , IsAtEnd(false)
    , WasPreviousTokenFlowEntry(true) // Start with an imaginary ','.
    , CurrentEntry(0)
  {}

  friend class basic_collection_iterator<SequenceNode, Node>;
  typedef basic_collection_iterator<SequenceNode, Node> iterator;
  template <class T> friend typename T::iterator yaml::begin(T &);
  template <class T> friend void yaml::skip(T &);

  void increment();

  iterator begin() {
    return yaml::begin(*this);
  }

  iterator end() { return iterator(); }

  virtual void skip() LLVM_OVERRIDE {
    yaml::skip(*this);
  }

  static inline bool classof(const Node *N) {
    return N->getType() == NK_Sequence;
  }

private:
  SequenceType SeqType;
  bool IsAtBeginning;
  bool IsAtEnd;
  bool WasPreviousTokenFlowEntry;
  Node *CurrentEntry;
};

/// @brief Represents an alias to a Node with an anchor.
///
/// Example:
///   *AnchorName
class AliasNode : public Node {
public:
  AliasNode(OwningPtr<Document> &D, StringRef Val)
    : Node(NK_Alias, D, StringRef()), Name(Val) {}

  StringRef getName() const { return Name; }
  Node *getTarget();

  static inline bool classof(const Node *N) {
    return N->getType() == NK_Alias;
  }

private:
  StringRef Name;
};

/// @brief A YAML Stream is a sequence of Documents. A document contains a root
///        node.
class Document {
public:
  /// @brief Root for parsing a node. Returns a single node.
  Node *parseBlockNode();

  Document(Stream &ParentStream);

  /// @brief Finish parsing the current document and return true if there are
  ///        more. Return false otherwise.
  bool skip();

  /// @brief Parse and return the root level node.
  Node *getRoot() {
    if (Root)
      return Root;
    return Root = parseBlockNode();
  }

private:
  friend class Node;
  friend class document_iterator;

  /// @brief Stream to read tokens from.
  Stream &stream;

  /// @brief Used to allocate nodes to. All are destroyed without calling their
  ///        destructor when the document is destroyed.
  BumpPtrAllocator NodeAllocator;

  /// @brief The root node. Used to support skipping a partially parsed
  ///        document.
  Node *Root;

  Token &peekNext();
  Token getNext();
  void setError(const Twine &Message, Token &Location) const;
  bool failed() const;

  void handleTagDirective(const Token &Tag) {
    // TODO: Track tags.
  }

  /// @brief Parse %BLAH directives and return true if any were encountered.
  bool parseDirectives();

  /// @brief Consume the next token and error if it is not \a TK.
  bool expectToken(int TK);
};

/// @brief Iterator abstraction for Documents over a Stream.
class document_iterator {
public:
  document_iterator() : Doc(0) {}
  document_iterator(OwningPtr<Document> &D) : Doc(&D) {}

  bool operator ==(const document_iterator &Other) {
    if (isAtEnd() || Other.isAtEnd())
      return isAtEnd() && Other.isAtEnd();

    return *Doc == *Other.Doc;
  }
  bool operator !=(const document_iterator &Other) {
    return !(*this == Other);
  }

  document_iterator operator ++() {
    assert(Doc != 0 && "incrementing iterator past the end.");
    if (!(*Doc)->skip()) {
      Doc->reset(0);
    } else {
      Stream &S = (*Doc)->stream;
      Doc->reset(new Document(S));
    }
    return *this;
  }

  Document &operator *() {
    return *Doc->get();
  }

  OwningPtr<Document> &operator ->() {
    return *Doc;
  }

private:
  bool isAtEnd() const {
    return Doc == 0 || *Doc == 0;
  }

  OwningPtr<Document> *Doc;
};

}
}

#endif
