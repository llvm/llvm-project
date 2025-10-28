//===-- llvm/ADT/RadixTree.h - Radix Tree implementation --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file implements a Radix Tree.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_RADIXTREE_H
#define LLVM_ADT_RADIXTREE_H

#include "llvm/ADT/ADL.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include <cassert>
#include <cstddef>
#include <iterator>
#include <list>
#include <optional>
#include <utility>
#include <vector>

namespace llvm {

/// \brief A Radix Tree implementation.
///
/// A Radix Tree (also known as a compact prefix tree or radix trie) is a
/// data structure that stores a dynamic set or associative array where keys
/// are strings and values are associated with these keys. Unlike a regular
/// trie, the edges of a radix tree can be labeled with sequences of characters
/// as well as single characters. This makes radix trees more efficient for
/// storing sparse data sets, where many nodes in a regular trie would have
/// only one child.
///
/// This implementation supports arbitrary key types that can be iterated over
/// (e.g., `std::string`, `std::vector<char>`, `ArrayRef<char>`). The key type
/// must provide `begin()` and `end()` for iteration.
///
/// The tree stores `std::pair<const KeyType, T>` as its value type.
///
/// Example usage:
/// \code
///   llvm::RadixTree<StringRef, int> Tree;
///   Tree.emplace("apple", 1);
///   Tree.emplace("grapefruit", 2);
///   Tree.emplace("grape", 3);
///
///   // Find prefixes
///   for (const auto &[Key, Value] : Tree.find_prefixes("grapefruit juice")) {
///     // pair will be {"grape", 3}
///     // pair will be {"grapefruit", 2}
///     llvm::outs() << Key << ": " << Value << "\n";
///   }
///
///   // Iterate over all elements
///   for (const auto &[Key, Value] : Tree)
///     llvm::outs() << Key << ": " << Value << "\n";
/// \endcode
///
/// \note
/// The `RadixTree` takes ownership of the `KeyType` and `T` objects
/// inserted into it. When an element is removed or the tree is destroyed,
/// these objects will be destructed.
/// However, if `KeyType` is a reference-like type, e.g., StringRef or range,
/// the user must guarantee that the referenced data has a lifetime longer than
/// the tree.
template <typename KeyType, typename T> class RadixTree {
public:
  using key_type = KeyType;
  using mapped_type = T;
  using value_type = std::pair<const KeyType, mapped_type>;

private:
  using KeyConstIteratorType =
      decltype(adl_begin(std::declval<const key_type &>()));
  using KeyConstIteratorRangeType = iterator_range<KeyConstIteratorType>;
  using KeyValueType =
      remove_cvref_t<decltype(*adl_begin(std::declval<key_type &>()))>;
  using ContainerType = std::list<value_type>;

  /// Represents an internal node in the Radix Tree.
  struct Node {
    KeyConstIteratorRangeType Key{KeyConstIteratorType{},
                                  KeyConstIteratorType{}};
    std::vector<Node> Children;

    /// An iterator to the value associated with this node.
    ///
    /// If this node does not have a value (i.e., it's an internal node that
    /// only serves as a path to other values), this iterator will be equal
    /// to default constructed `ContainerType::iterator()`.
    std::optional<typename ContainerType::iterator> Value;

    /// The first character of the Key. Used for fast child lookup.
    KeyValueType KeyFront;

    Node() = default;
    Node(const KeyConstIteratorRangeType &Key)
        : Key(Key), KeyFront(*Key.begin()) {
      assert(!Key.empty());
    }

    Node(Node &&) = default;
    Node &operator=(Node &&) = default;

    Node(const Node &) = delete;
    Node &operator=(const Node &) = delete;

    const Node *findChild(const KeyConstIteratorRangeType &Key) const {
      if (Key.empty())
        return nullptr;
      for (const Node &Child : Children) {
        assert(!Child.Key.empty()); // Only root can be empty.
        if (Child.KeyFront == *Key.begin())
          return &Child;
      }
      return nullptr;
    }

    Node *findChild(const KeyConstIteratorRangeType &Query) {
      const Node *This = this;
      return const_cast<Node *>(This->findChild(Query));
    }

    size_t countNodes() const {
      size_t R = 1;
      for (const Node &C : Children)
        R += C.countNodes();
      return R;
    }

    ///
    /// Splits the current node into two.
    ///
    /// This function is used when a new key needs to be inserted that shares
    /// a common prefix with the current node's key, but then diverges.
    /// The current `Key` is truncated to the common prefix, and a new child
    /// node is created for the remainder of the original node's `Key`.
    ///
    /// \param SplitPoint An iterator pointing to the character in the current
    ///                   `Key` where the split should occur.
    void split(KeyConstIteratorType SplitPoint) {
      Node Child(make_range(SplitPoint, Key.end()));
      Key = make_range(Key.begin(), SplitPoint);

      Children.swap(Child.Children);
      std::swap(Value, Child.Value);

      Children.emplace_back(std::move(Child));
    }
  };

  /// Root always corresponds to the empty key, which is the shortest possible
  /// prefix for everything.
  Node Root;
  ContainerType KeyValuePairs;

  /// Finds or creates a new tail or leaf node corresponding to the `Key`.
  Node &findOrCreate(KeyConstIteratorRangeType Key) {
    Node *Curr = &Root;
    if (Key.empty())
      return *Curr;

    for (;;) {
      auto [I1, I2] = llvm::mismatch(Key, Curr->Key);
      Key = make_range(I1, Key.end());

      if (I2 != Curr->Key.end()) {
        // Match is partial. Either query is too short, or there is mismatching
        // character. Split either way, and put new node in between of the
        // current and its children.
        Curr->split(I2);

        // Split was caused by mismatch, so `findChild` would fail.
        break;
      }

      Node *Child = Curr->findChild(Key);
      if (!Child)
        break;

      // Move to child with the same first character.
      Curr = Child;
    }

    if (Key.empty()) {
      // The current node completely matches the key, return it.
      return *Curr;
    }

    // `Key` is a suffix of original `Key` unmatched by path from the `Root` to
    // the `Curr`, and we have no candidate in the children to match more.
    // Create a new one.
    return Curr->Children.emplace_back(Key);
  }

  ///
  /// An iterator for traversing prefixes search results.
  ///
  /// This iterator is used by `find_prefixes` to traverse the tree and find
  /// elements that are prefixes to the given key. It's a forward iterator.
  ///
  /// \tparam MappedType The type of the value pointed to by the iterator.
  ///                    This will be `value_type` for non-const iterators
  ///                    and `const value_type` for const iterators.
  template <typename MappedType>
  class IteratorImpl
      : public iterator_facade_base<IteratorImpl<MappedType>,
                                    std::forward_iterator_tag, MappedType> {
    const Node *Curr = nullptr;
    KeyConstIteratorRangeType Query{KeyConstIteratorType{},
                                    KeyConstIteratorType{}};

    void findNextValid() {
      while (Curr && !Curr->Value.has_value())
        advance();
    }

    void advance() {
      assert(Curr);
      if (Query.empty()) {
        Curr = nullptr;
        return;
      }

      Curr = Curr->findChild(Query);
      if (!Curr) {
        Curr = nullptr;
        return;
      }

      auto [I1, I2] = llvm::mismatch(Query, Curr->Key);
      if (I2 != Curr->Key.end()) {
        Curr = nullptr;
        return;
      }
      Query = make_range(I1, Query.end());
    }

    friend class RadixTree;
    IteratorImpl(const Node *C, const KeyConstIteratorRangeType &Q)
        : Curr(C), Query(Q) {
      findNextValid();
    }

  public:
    IteratorImpl() = default;

    MappedType &operator*() const { return **Curr->Value; }

    IteratorImpl &operator++() {
      advance();
      findNextValid();
      return *this;
    }

    bool operator==(const IteratorImpl &Other) const {
      return Curr == Other.Curr;
    }
  };

public:
  RadixTree() = default;
  RadixTree(RadixTree &&) = default;
  RadixTree &operator=(RadixTree &&) = default;

  using prefix_iterator = IteratorImpl<value_type>;
  using const_prefix_iterator = IteratorImpl<const value_type>;

  using iterator = typename ContainerType::iterator;
  using const_iterator = typename ContainerType::const_iterator;

  /// Returns true if the tree is empty.
  bool empty() const { return KeyValuePairs.empty(); }

  /// Returns the number of elements in the tree.
  size_t size() const { return KeyValuePairs.size(); }

  /// Returns the number of nodes in the tree.
  ///
  /// This function counts all internal nodes in the tree. It can be useful for
  /// understanding the memory footprint or complexity of the tree structure.
  size_t countNodes() const { return Root.countNodes(); }

  /// Returns an iterator to the first element.
  iterator begin() { return KeyValuePairs.begin(); }
  const_iterator begin() const { return KeyValuePairs.begin(); }

  /// Returns an iterator to the end of the tree.
  iterator end() { return KeyValuePairs.end(); }
  const_iterator end() const { return KeyValuePairs.end(); }

  /// Constructs and inserts a new element into the tree.
  ///
  /// This function constructs an element in place within the tree. If an
  /// element with the same key already exists, the insertion fails and the
  /// function returns an iterator to the existing element along with `false`.
  /// Otherwise, the new element is inserted and the function returns an
  /// iterator to the new element along with `true`.
  ///
  /// \param Key The key of the element to construct.
  /// \param Args Arguments to forward to the constructor of the mapped_type.
  /// \return A pair consisting of an iterator to the inserted element (or to
  ///         the element that prevented insertion) and a boolean value
  ///         indicating whether the insertion took place.
  template <typename... Ts>
  std::pair<iterator, bool> emplace(key_type &&Key, Ts &&...Args) {
    // We want to make new `Node` to refer key in the container, not the one
    // from the argument.
    // FIXME: Determine that we need a new node, before expanding
    // `KeyValuePairs`.
    const value_type &NewValue = KeyValuePairs.emplace_front(
        std::move(Key), T(std::forward<Ts>(Args)...));
    Node &Node = findOrCreate(NewValue.first);
    bool HasValue = Node.Value.has_value();
    if (!HasValue)
      Node.Value = KeyValuePairs.begin();
    else
      KeyValuePairs.pop_front();
    return {*Node.Value, !HasValue};
  }

  ///
  /// Finds all elements whose keys are prefixes of the given `Key`.
  ///
  /// This function returns an iterator range over all elements in the tree
  /// whose keys are prefixes of the provided `Key`. For example, if the tree
  /// contains "abcde", "abc", "abcdefgh", and `Key` is "abcde", this function
  /// would return iterators to "abcde" and "abc".
  ///
  /// \param Key The key to search for prefixes of.
  /// \return An `iterator_range` of `const_prefix_iterator`s, allowing
  ///         iteration over the found prefix elements.
  /// \note The returned iterators reference the `Key` provided by the caller.
  ///       The caller must ensure that `Key` remains valid for the lifetime
  ///       of the iterators.
  iterator_range<const_prefix_iterator>
  find_prefixes(const key_type &Key) const {
    return iterator_range<const_prefix_iterator>{
        const_prefix_iterator(&Root, KeyConstIteratorRangeType(Key)),
        const_prefix_iterator{}};
  }
};

} // namespace llvm

#endif // LLVM_ADT_RADIXTREE_H
