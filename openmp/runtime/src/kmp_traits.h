//===----------- Traits.h - OpenMP context traits -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of OpenMP context traits.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_TRAITS_H
#define OPENMP_TRAITS_H

#include "kmp.h"
#include "kmp_adt.h"

namespace kmp_traits {

extern "C" int omp_get_num_devices();
extern "C" const char *omp_get_uid_from_device(int device_num);

class kmp_trait {
protected:
  enum trait_type { WILDCARD_T, LITERAL_T, UID_T };
  trait_type _type;

  kmp_trait(trait_type type) : _type(type) {}

public:
  virtual ~kmp_trait() = default;

  kmp_trait(const kmp_trait &) = delete;
  kmp_trait(kmp_trait &&) = delete;
  kmp_trait &operator=(const kmp_trait &) = delete;
  kmp_trait &operator=(kmp_trait &&) = delete;

  virtual bool match(int device) const = 0;

  // Use KMP_INTERNAL_MALLOC/KMP_INTERNAL_FREE for memory management.
  void *operator new(size_t size) { return KMP_INTERNAL_MALLOC(size); }
  void operator delete(void *ptr) { KMP_INTERNAL_FREE(ptr); }

  virtual bool operator==(const kmp_trait &other) const {
    return _type == other._type;
  }
};

/// Represents a wildcard trait that matches any device.
class kmp_wildcard_trait final : public kmp_trait {
public:
  kmp_wildcard_trait() : kmp_trait(WILDCARD_T) {}

  bool match([[maybe_unused]] int device) const override { return true; }

  bool operator==(const kmp_trait &other) const override {
    return kmp_trait::operator==(other);
  }
};

/// Represents a specific device number.
class kmp_literal_trait final : public kmp_trait {
  int device_num;

public:
  kmp_literal_trait(int device_num)
      : kmp_trait(LITERAL_T), device_num(device_num) {
    assert(device_num >= 0 && "Device number must be non-negative");
  }

  bool match(int device) const override { return device_num == device; }

  bool operator==(const kmp_trait &other) const override {
    return kmp_trait::operator==(other) &&
           device_num ==
               static_cast<const kmp_literal_trait &>(other).device_num;
  }
};

/// Represents a specific UID.
/// UID is deliberately not resolved at construction time since libomptarget
/// might not be initialized yet. This is why we delay calls to
/// omp_get_uid_from_device / omp_get_device_from_uid until the trait is
/// evaluated.
class kmp_uid_trait final : public kmp_trait {
  char *uid;
  // Can be used by unit tests to mock omp_get_uid_from_device.
  const char *(*get_uid_from_device)(int device) = omp_get_uid_from_device;

public:
  kmp_uid_trait(kmp_str_ref uid) : kmp_trait(UID_T), uid(uid.copy()) {}

  ~kmp_uid_trait() override {
    if (uid)
      KMP_INTERNAL_FREE(uid);
  }

  bool match(int device) const override {
    const char *device_uid = get_uid_from_device(device);
    if (!device_uid || !uid)
      return false;
    return strcmp(device_uid, uid) == 0;
  }

  // For testing purposes only: set the function that returns the UID from a
  // device.
  void set_uid_from_device(const char *(*uid_from_device)(int)) {
    get_uid_from_device = uid_from_device;
  }

  bool operator==(const kmp_trait &other) const override {
    if (!kmp_trait::operator==(other))
      return false;
    const char *other_uid = static_cast<const kmp_uid_trait &>(other).uid;
    return uid && other_uid ? strcmp(uid, other_uid) == 0 : uid == other_uid;
  }
};

/// Abstract class representing either a single trait expression or a collection
/// of trait expressions that are ANDed or ORed together.
class kmp_trait_expr {
protected:
  enum expr_type { SINGLE_T, GROUP_T };
  expr_type _type;
  // Determines if the expression is negated (true) or not (false).
  bool negated = false;
  // Can be used by unit tests to mock omp_get_num_devices.
  int (*get_num_devices)() = omp_get_num_devices;

  kmp_trait_expr(expr_type type) : _type(type) {}
  kmp_trait_expr(expr_type type, bool negated)
      : _type(type), negated(negated) {}

  virtual bool match_impl(int device, int num_devices) const = 0;

public:
  virtual ~kmp_trait_expr() = default;

  kmp_trait_expr(const kmp_trait_expr &) = delete;
  kmp_trait_expr(kmp_trait_expr &&) = delete;
  kmp_trait_expr &operator=(const kmp_trait_expr &) = delete;
  kmp_trait_expr &operator=(kmp_trait_expr &&) = delete;

  bool is_negated() const { return negated; }

  // Check if the device matches the expression.
  bool match(int device, int num_devices = -1) const {
    if (num_devices == -1)
      num_devices = get_num_devices();
    if (device < 0 || device >= num_devices)
      return false;
    return match_impl(device, num_devices);
  }

  void set_negated(bool neg = true) { negated = neg; }

  // For testing purposes only: set the function that returns the number of
  // devices.
  void set_num_devices(int (*num_devices)()) { get_num_devices = num_devices; }

  // Use KMP_INTERNAL_MALLOC/KMP_INTERNAL_FREE for memory management.
  void *operator new(size_t size) { return KMP_INTERNAL_MALLOC(size); }
  void operator delete(void *ptr) { KMP_INTERNAL_FREE(ptr); }

  virtual bool operator==(const kmp_trait_expr &other) const {
    return _type == other._type && negated == other.negated;
  }
};

/// Represents a single (possibly negated) trait.
class kmp_trait_expr_single final : public kmp_trait_expr {
  kmp_trait *trait = nullptr;

protected:
  bool match_impl(int device, [[maybe_unused]] int num_devices) const override {
    assert(trait);
    bool result = trait->match(device);
    return negated ? !result : result;
  }

public:
  kmp_trait_expr_single() : kmp_trait_expr(SINGLE_T) {}
  kmp_trait_expr_single(bool negated) : kmp_trait_expr(SINGLE_T, negated) {}
  kmp_trait_expr_single(kmp_trait *trait)
      : kmp_trait_expr(SINGLE_T), trait(trait) {
    assert(trait && "kmp_trait_expr_single requires a non-null trait");
  }
  ~kmp_trait_expr_single() override { delete trait; }

  void set_trait(kmp_trait *new_trait) {
    assert(new_trait);
    if (trait)
      delete trait;
    trait = new_trait;
  }

  bool operator==(const kmp_trait_expr &other) const override {
    if (!kmp_trait_expr::operator==(other))
      return false;
    const kmp_trait_expr_single &other_single =
        static_cast<const kmp_trait_expr_single &>(other);
    return trait && other_single.trait ? *trait == *other_single.trait
                                       : trait == other_single.trait;
  }
};

/// Represents a (possibly negated) collection of traits that are either ANDed
/// or ORed together.
class kmp_trait_expr_group final : public kmp_trait_expr {
public:
  enum group_type { AND, OR };

private:
  kmp_vector<kmp_trait_expr *> exprs;
  // Determines if all traits have to match (true) or any of them (false).
  group_type type = OR;

protected:
  bool match_impl(int device, int num_devices) const override {
    size_t matched = 0;
    for (const kmp_trait_expr *expr : exprs) {
      if (expr->match(device, num_devices))
        matched++;
    }
    // Note: AND evaluates to true for an empty group.
    bool result = type == AND ? matched == exprs.size() : matched > 0;
    return negated ? !result : result;
  }

public:
  kmp_trait_expr_group() : kmp_trait_expr(GROUP_T) {}
  kmp_trait_expr_group(bool negated) : kmp_trait_expr(GROUP_T, negated) {}
  ~kmp_trait_expr_group() override {
    for (kmp_trait_expr *expr : exprs)
      delete expr;
  }

  void add_expr(kmp_trait *trait) {
    assert(trait);
    add_expr(new kmp_trait_expr_single(trait));
  }
  void add_expr(kmp_trait_expr *expr) {
    assert(expr);
    exprs.push_back(expr);
    // Propagate get_num_devices to the expression.
    expr->set_num_devices(get_num_devices);
  }

  group_type get_group_type() const { return type; }

  void set_group_type(group_type new_type) { type = new_type; }

  void set_num_devices(int (*num_devices)()) {
    kmp_trait_expr::set_num_devices(num_devices);
    for (kmp_trait_expr *expr : exprs)
      expr->set_num_devices(num_devices);
  }

  bool operator==(const kmp_trait_expr &other) const override {
    if (!kmp_trait_expr::operator==(other))
      return false;
    const kmp_trait_expr_group &other_group =
        static_cast<const kmp_trait_expr_group &>(other);
    return exprs.is_set_equal(
        other_group.exprs, [](kmp_trait_expr *const &a,
                              kmp_trait_expr *const &b) { return *a == *b; });
  }
};

class kmp_trait_clause final {
  kmp_trait_expr *expr = nullptr;

public:
  kmp_trait_clause() = default;
  ~kmp_trait_clause() { delete expr; }

  kmp_trait_clause(const kmp_trait_clause &) = delete;
  kmp_trait_clause(kmp_trait_clause &&) = delete;
  kmp_trait_clause &operator=(const kmp_trait_clause &) = delete;
  kmp_trait_clause &operator=(kmp_trait_clause &&) = delete;

  kmp_trait_expr *get_expr() { return expr; }

  bool match(int device, int num_devices = -1) const {
    assert(expr);
    return expr->match(device, num_devices);
  }

  void set_expr(kmp_trait *trait) {
    assert(trait);
    if (expr)
      delete expr;
    expr = new kmp_trait_expr_single(trait);
  }
  void set_expr(kmp_trait_expr *new_expr) {
    assert(new_expr);
    if (expr)
      delete expr;
    expr = new_expr;
  }

  // Use KMP_INTERNAL_MALLOC/KMP_INTERNAL_FREE for memory management.
  void *operator new(size_t size) { return KMP_INTERNAL_MALLOC(size); }
  void operator delete(void *ptr) { KMP_INTERNAL_FREE(ptr); }

  bool operator==(const kmp_trait_clause &other) const {
    return expr && other.expr ? *expr == *other.expr : expr == other.expr;
  }
};

} // namespace kmp_traits

class kmp_trait_context final {
  using kmp_trait_clause = kmp_traits::kmp_trait_clause;
  using kmp_trait_expr = kmp_traits::kmp_trait_expr;

  kmp_vector<kmp_trait_clause *> clauses;
  // List of devices that have been evaluated.
  kmp_vector<int> devices;
  bool evaluated = false;
  // Can be used by unit tests to mock omp_get_num_devices.
  int (*get_num_devices)() = kmp_traits::omp_get_num_devices;

  void _evaluate() {
    devices.clear();
    for (int d = 0; d < get_num_devices(); ++d) {
      if (_match(d))
        devices.push_back(d);
    }
    evaluated = true;
  }

  bool _match(int device) const {
    if (device < 0 || device >= get_num_devices())
      return false;
    for (kmp_trait_clause *clause : clauses) {
      if (clause->match(device))
        return true;
    }
    return false;
  }

public:
  kmp_trait_context() = default;
  ~kmp_trait_context() {
    for (kmp_trait_clause *clause : clauses)
      delete clause;
  }

  kmp_trait_context(const kmp_trait_context &) = delete;
  kmp_trait_context(kmp_trait_context &&) = delete;
  kmp_trait_context &operator=(const kmp_trait_context &) = delete;
  kmp_trait_context &operator=(kmp_trait_context &&) = delete;

  // Parse a trait specification from a string.
  // If dbg_name is provided, it will be used in error messages to identify the
  // source of the trait specification.
  static kmp_trait_context *parse_from_spec(kmp_str_ref spec,
                                            const char *dbg_name = nullptr);

  // Parse only a single device number from the spec.
  // This is useful for backward compatibility with legacy code.
  // If dbg_name is provided, it will be used in error messages to identify the
  // source of the device number.
  static int parse_single_device(kmp_str_ref spec, int device_num_limit,
                                 const char *dbg_name = nullptr);

  void add_clause(kmp_trait_clause *clause) {
    assert(clause);
    clauses.push_back(clause);
    // Propagate get_num_devices to the clause.
    if (kmp_trait_expr *expr = clause->get_expr())
      expr->set_num_devices(get_num_devices);
  }

  // Returns the list of devices that match the trait specification represented
  // by the context. The list contains devices numbers forming a set and sorted
  // in ascending order.
  // Note to future developers: if we want to add an option to force
  // re-evaluation, we need to consider that the devices vector and thus the
  // context iterators are invalidated.
  const kmp_vector<int> &evaluate() {
    trigger_evaluation();
    return devices;
  }

  const kmp_vector<int> &evaluate() const {
    assert(evaluated && "kmp_trait_context not evaluated");
    return devices;
  }

  // Check if the device matches the trait specification represented by the
  // context.
  bool match(int device) { return evaluate().contains(device); }

  bool match(int device) const {
    assert(evaluated && "kmp_trait_context not evaluated");
    return devices.contains(device);
  }

  // For testing purposes only: set the function that returns the number of
  // devices.
  void set_num_devices(int (*num_devices)()) {
    get_num_devices = num_devices;
    for (kmp_trait_clause *clause : clauses) {
      if (kmp_trait_expr *expr = clause->get_expr())
        expr->set_num_devices(num_devices);
    }
  }

  // Triggers lazy evaluation if not already evaluated.
  void trigger_evaluation() {
    if (!evaluated)
      _evaluate();
  }

  // Use KMP_INTERNAL_MALLOC/KMP_INTERNAL_FREE for memory management.
  void *operator new(size_t size) { return KMP_INTERNAL_MALLOC(size); }
  void operator delete(void *ptr) { KMP_INTERNAL_FREE(ptr); }

  bool operator==(const kmp_trait_context &other) const {
    auto clause_comp = [](kmp_trait_clause *const &a,
                          kmp_trait_clause *const &b) { return *a == *b; };
    return clauses.is_set_equal(other.clauses, clause_comp);
  }

  // Iterator support (returns the iterators of the devices vector; triggers
  // lazy evaluation if not already evaluated and if the context is not const).
  const int *begin() { return evaluate().begin(); }
  const int *end() { return evaluate().end(); }
  const int *begin() const { return evaluate().begin(); }
  const int *end() const { return evaluate().end(); }
};

#endif // OPENMP_TRAITS_H
