//===- Support/GICHelper.h -- Helper functions for ISL --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Helper functions for isl objects.
//
//===----------------------------------------------------------------------===//
//
#ifndef POLLY_SUPPORT_GIC_HELPER_H
#define POLLY_SUPPORT_GIC_HELPER_H

#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"
#include "isl/aff.h"
#include "isl/ctx.h"
#include "isl/map.h"
#include "isl/options.h"
#include "isl/set.h"
#include "isl/union_map.h"
#include "isl/union_set.h"
#include <functional>
#include <string>

struct isl_schedule;
struct isl_multi_aff;

namespace llvm {
class Value;
} // namespace llvm

namespace polly {

/// Translate an llvm::APInt to an isl_val.
///
/// Translate the bitsequence without sign information as provided by APInt into
/// a signed isl_val type. Depending on the value of @p IsSigned @p Int is
/// interpreted as unsigned value or as signed value in two's complement
/// representation.
///
/// Input IsSigned                 Output
///
///     0        0           ->    0
///     1        0           ->    1
///    00        0           ->    0
///    01        0           ->    1
///    10        0           ->    2
///    11        0           ->    3
///
///     0        1           ->    0
///     1        1           ->   -1
///    00        1           ->    0
///    01        1           ->    1
///    10        1           ->   -2
///    11        1           ->   -1
///
/// @param Ctx      The isl_ctx to create the isl_val in.
/// @param Int      The integer value to translate.
/// @param IsSigned If the APInt should be interpreted as signed or unsigned
///                 value.
///
/// @return The isl_val corresponding to @p Int.
__isl_give isl_val *isl_valFromAPInt(isl_ctx *Ctx, const llvm::APInt Int,
                                     bool IsSigned);

/// Translate isl_val to llvm::APInt.
///
/// This function can only be called on isl_val values which are integers.
/// Calling this function with a non-integral rational, NaN or infinity value
/// is not allowed.
///
/// As the input isl_val may be negative, the APInt that this function returns
/// must always be interpreted as signed two's complement value. The bitwidth of
/// the generated APInt is always the minimal bitwidth necessary to model the
/// provided integer when interpreting the bitpattern as signed value.
///
/// Some example conversions are:
///
///   Input      Bits    Signed  Bitwidth
///       0 ->      0         0         1
///      -1 ->      1        -1         1
///       1 ->     01         1         2
///      -2 ->     10        -2         2
///       2 ->    010         2         3
///      -3 ->    101        -3         3
///       3 ->    011         3         3
///      -4 ->    100        -4         3
///       4 ->   0100         4         4
///
/// @param Val The isl val to translate.
///
/// @return The APInt value corresponding to @p Val.
llvm::APInt APIntFromVal(__isl_take isl_val *Val);

/// Get c++ string from Isl objects.
//@{
std::string stringFromIslObj(__isl_keep isl_map *map);
std::string stringFromIslObj(__isl_keep isl_union_map *umap);
std::string stringFromIslObj(__isl_keep isl_set *set);
std::string stringFromIslObj(__isl_keep isl_union_set *uset);
std::string stringFromIslObj(__isl_keep isl_schedule *schedule);
std::string stringFromIslObj(__isl_keep isl_multi_aff *maff);
std::string stringFromIslObj(__isl_keep isl_pw_multi_aff *pma);
std::string stringFromIslObj(__isl_keep isl_multi_pw_aff *mpa);
std::string stringFromIslObj(__isl_keep isl_union_pw_multi_aff *upma);
std::string stringFromIslObj(__isl_keep isl_aff *aff);
std::string stringFromIslObj(__isl_keep isl_pw_aff *pwaff);
std::string stringFromIslObj(__isl_keep isl_space *space);
//@}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     __isl_keep isl_union_map *Map) {
  OS << polly::stringFromIslObj(Map);
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     __isl_keep isl_map *Map) {
  OS << polly::stringFromIslObj(Map);
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     __isl_keep isl_set *Set) {
  OS << polly::stringFromIslObj(Set);
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     __isl_keep isl_pw_aff *Map) {
  OS << polly::stringFromIslObj(Map);
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     __isl_keep isl_pw_multi_aff *PMA) {
  OS << polly::stringFromIslObj(PMA);
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     __isl_keep isl_multi_aff *MA) {
  OS << polly::stringFromIslObj(MA);
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     __isl_keep isl_union_pw_multi_aff *UPMA) {
  OS << polly::stringFromIslObj(UPMA);
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     __isl_keep isl_schedule *Schedule) {
  OS << polly::stringFromIslObj(Schedule);
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     __isl_keep isl_space *Space) {
  OS << polly::stringFromIslObj(Space);
  return OS;
}

/// Return @p Prefix + @p Val->getName() + @p Suffix but Isl compatible.
std::string getIslCompatibleName(const std::string &Prefix,
                                 const llvm::Value *Val,
                                 const std::string &Suffix);

std::string getIslCompatibleName(const std::string &Prefix,
                                 const std::string &Middle,
                                 const std::string &Suffix);

/// IslObjTraits<isl_*> is a static class to invoke common functions that all
/// ISL objects have: isl_*_copy, isl_*_free, isl_*_get_ctx and isl_*_to_str.
/// These functions follow a common naming scheme, but not a base class
/// hierarchy (as ISL is written in C). As such, the functions are accessible
/// only by constructing the function name using the preprocessor. This class
/// serves to make these names accessible to a C++ template scheme.
///
/// There is an isl_obj polymorphism layer, but its implementation is
/// incomplete.
template <typename T> class IslObjTraits;

#define DECLARE_TRAITS(TYPE)                                                   \
  template <> class IslObjTraits<isl_##TYPE> {                                 \
  public:                                                                      \
    static __isl_give isl_##TYPE *copy(__isl_keep isl_##TYPE *Obj) {           \
      return isl_##TYPE##_copy(Obj);                                           \
    }                                                                          \
    static void free(__isl_take isl_##TYPE *Obj) { isl_##TYPE##_free(Obj); }   \
    static isl_ctx *get_ctx(__isl_keep isl_##TYPE *Obj) {                      \
      return isl_##TYPE##_get_ctx(Obj);                                        \
    }                                                                          \
    static std::string to_str(__isl_keep isl_##TYPE *Obj) {                    \
      if (!Obj)                                                                \
        return "null";                                                         \
      char *cstr = isl_##TYPE##_to_str(Obj);                                   \
      if (!cstr)                                                               \
        return "null";                                                         \
      std::string Result{cstr};                                                \
      ::free(cstr);                                                            \
      return Result;                                                           \
    }                                                                          \
  };

DECLARE_TRAITS(id)
DECLARE_TRAITS(val)
DECLARE_TRAITS(space)
DECLARE_TRAITS(basic_map)
DECLARE_TRAITS(map)
DECLARE_TRAITS(union_map)
DECLARE_TRAITS(basic_set)
DECLARE_TRAITS(set)
DECLARE_TRAITS(union_set)
DECLARE_TRAITS(aff)
DECLARE_TRAITS(multi_aff)
DECLARE_TRAITS(pw_aff)
DECLARE_TRAITS(pw_multi_aff)
DECLARE_TRAITS(multi_pw_aff)
DECLARE_TRAITS(union_pw_aff)
DECLARE_TRAITS(multi_union_pw_aff)
DECLARE_TRAITS(union_pw_multi_aff)

template <typename T> class NonowningIslPtr;

/// Smart pointer to an ISL object.
///
/// An object of this class owns an reference of an ISL object, meaning if will
/// free it when destroyed. Most ISL objects are reference counted such that we
/// gain an automatic memory management.
///
/// Function parameters in the ISL API are annotated using either __isl_keep
/// __isl_take. Return values that are objects are annotated using __is_give,
/// meaning the caller is responsible for releasing the object. When annotated
/// with __isl_keep, use the keep() function to pass a plain pointer to the ISL
/// object. For __isl_take-annotated parameters, use either copy() to increase
/// the reference counter by one, or take() to pass the ownership to the called
/// function. When IslPtr loses ownership, it cannot be used anymore and won't
/// free the object when destroyed. Use the give() function to wrap the
/// ownership of a returned isl_* object into an IstPtr<isl_*>.
///
/// There is purposefully no implicit conversion from/to plain isl_* pointers to
/// avoid difficult to find bugs because keep/copy/take would have been
/// required.
template <typename T> class IslPtr {
  typedef IslPtr<T> ThisTy;
  typedef IslObjTraits<T> Traits;

private:
  T *Obj;

  explicit IslPtr(__isl_take T *Obj) : Obj(Obj) {}

public:
  IslPtr() : Obj(nullptr) {}
  /* implicit */ IslPtr(std::nullptr_t That) : IslPtr() {}

  /* implicit */ IslPtr(const ThisTy &That)
      : IslPtr(IslObjTraits<T>::copy(That.Obj)) {}
  /* implicit */ IslPtr(ThisTy &&That) : IslPtr(That.Obj) {
    That.Obj = nullptr;
  }
  /* implicit */ IslPtr(NonowningIslPtr<T> That) : IslPtr(That.copy()) {}
  ~IslPtr() {
    if (Obj)
      Traits::free(Obj);
  }

  ThisTy &operator=(const ThisTy &That) {
    if (Obj)
      Traits::free(Obj);
    this->Obj = Traits::copy(That.Obj);
    return *this;
  }
  ThisTy &operator=(ThisTy &&That) {
    swap(*this, That);
    return *this;
  }

  explicit operator bool() const { return Obj; }

  static void swap(ThisTy &LHS, ThisTy &RHS) { std::swap(LHS.Obj, RHS.Obj); }

  static ThisTy give(__isl_take T *Obj) { return ThisTy(Obj); }
  T *keep() const { return Obj; }
  __isl_give T *take() {
    auto *Result = Obj;
    Obj = nullptr;
    return Result;
  }
  __isl_give T *copy() const { return Traits::copy(Obj); }

  isl_ctx *getCtx() const { return Traits::get_ctx(Obj); }
  std::string toStr() const { return Traits::to_str(Obj); }

  /// Print a string representation of this ISL object to stderr.
  ///
  /// This function is meant to be called from a debugger and therefore must
  /// not be declared inline: The debugger needs a valid function pointer to
  /// call, even if the method is not used.
  ///
  /// Note that the string representation of isl_*_dump is different than the
  /// one for isl_printer/isl_*_to_str().
  void dump() const;
};

template <typename T> static IslPtr<T> give(__isl_take T *Obj) {
  return IslPtr<T>::give(Obj);
}

template <typename T>
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const IslPtr<T> &Obj) {
  OS << IslObjTraits<T>::to_str(Obj.keep());
  return OS;
}

/// Smart pointer to an ISL object, but does not release it when destroyed.
///
/// This is meant to be used as function parameter type. The caller guarantees
/// that the reference is alive during the function's execution and hence
/// doesn't need to add a reference. Therefore, it is equivalent to the
/// __isl_keep annotation (IslPtr being equivalent to __isl_take which can be
/// either copied or moved).
///
/// Just as IslPtr, it has keep() and copy() methods. The take() method is
/// missing as this would steal the reference from the owner (the caller).
template <typename T> class NonowningIslPtr {
  typedef NonowningIslPtr<T> ThisTy;
  typedef IslObjTraits<T> Traits;

private:
  T *Obj;

  /* implicit */ NonowningIslPtr(__isl_keep T *Obj) : Obj(Obj) {}

public:
  NonowningIslPtr() : Obj(nullptr) {}
  /* implicit */ NonowningIslPtr(std::nullptr_t That) : NonowningIslPtr() {}

  /* implicit */ NonowningIslPtr(const IslPtr<T> &That)
      : NonowningIslPtr(That.keep()) {}

  explicit operator bool() const { return Obj; }

  static void swap(ThisTy &LHS, ThisTy &RHS) { std::swap(LHS.Obj, RHS.Obj); }

  T *keep() const { return Obj; }
  __isl_give T *copy() const { return Traits::copy(Obj); }

  isl_ctx *getCtx() const { return Traits::get_ctx(Obj); }
  std::string toStr() const { return Traits::to_str(Obj); }

  /// Print a string representation of this ISL object to stderr.
  ///
  /// @see IslPtr<T>::dump()
  void dump() const;
};

template <typename T>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     NonowningIslPtr<T> Obj) {
  OS << IslObjTraits<T>::to_str(Obj.keep());
  return OS;
}

/// Enumerate all isl_basic_maps of an isl_map.
///
/// This basically wraps isl_map_foreach_basic_map() and allows to call back
/// C++11 closures.
void foreachElt(NonowningIslPtr<isl_map> Map,
                const std::function<void(IslPtr<isl_basic_map>)> &F);

/// Enumerate all isl_basic_sets of an isl_set.
///
/// This basically wraps isl_set_foreach_basic_set() and allows to call back
/// C++11 closures.
void foreachElt(NonowningIslPtr<isl_set> Set,
                const std::function<void(IslPtr<isl_basic_set>)> &F);

/// Enumerate all isl_maps of an isl_union_map.
///
/// This basically wraps isl_union_map_foreach_map() and allows to call back
/// C++11 closures.
void foreachElt(NonowningIslPtr<isl_union_map> UMap,
                const std::function<void(IslPtr<isl_map> Map)> &F);

/// Enumerate all isl_sets of an isl_union_set.
///
/// This basically wraps isl_union_set_foreach_set() and allows to call back
/// C++11 closures.
void foreachElt(NonowningIslPtr<isl_union_set> USet,
                const std::function<void(IslPtr<isl_set> Set)> &F);

/// Enumerate all isl_pw_aff of an isl_union_pw_aff.
///
/// This basically wraps isl_union_pw_aff(), but also allows to call back C++11
/// closures.
void foreachElt(NonowningIslPtr<isl_union_pw_aff> UPwAff,
                const std::function<void(IslPtr<isl_pw_aff>)> &F);

/// Enumerate all polyhedra of an isl_map.
///
/// This is a wrapper for isl_map_foreach_basic_map() that allows to call back
/// C++ closures. The callback has the possibility to interrupt (break) the
/// enumeration by returning isl_stat_error. A return value of isl_stat_ok will
/// continue enumerations, if any more elements are left.
///
/// @param UMap Collection to enumerate.
/// @param F    The callback function, lambda or closure.
///
/// @return The isl_stat returned by the last callback invocation; isl_stat_ok
///         if the collection was empty.
isl_stat
foreachEltWithBreak(NonowningIslPtr<isl_map> Map,
                    const std::function<isl_stat(IslPtr<isl_basic_map>)> &F);

/// Enumerate all isl_maps of an isl_union_map.
///
/// This is a wrapper for isl_union_map_foreach_map() that allows to call back
/// C++ closures. In contrast to the variant without "_with_break", the callback
/// has the possibility to interrupt (break) the enumeration by returning
/// isl_stat_error. A return value of isl_stat_ok will continue enumerations, if
/// any more elements are left.
///
/// @param UMap Collection to enumerate.
/// @param F    The callback function, lambda or closure.
///
/// @return The isl_stat returned by the last callback invocation; isl_stat_ok
///         if the collection was initially empty.
isl_stat
foreachEltWithBreak(NonowningIslPtr<isl_union_map> UMap,
                    const std::function<isl_stat(IslPtr<isl_map> Map)> &F);

/// Enumerate all pieces of an isl_pw_aff.
///
/// This is a wrapper around isl_pw_aff_foreach_piece() that allows to call back
/// C++11 closures. The callback has the possibility to interrupt (break) the
/// enumeration by returning isl_stat_error. A return value of isl_stat_ok will
/// continue enumerations, if any more elements are left.
///
/// @param UMap Collection to enumerate.
/// @param F    The callback function, lambda or closure.
///
/// @return The isl_stat returned by the last callback invocation; isl_stat_ok
///         if the collection was initially empty.
isl_stat foreachPieceWithBreak(
    NonowningIslPtr<isl_pw_aff> PwAff,
    const std::function<isl_stat(IslPtr<isl_set>, IslPtr<isl_aff>)> &F);

/// Scoped limit of ISL operations.
///
/// Limits the number of ISL operations during the lifetime of this object. The
/// idea is to use this as an RAII guard for the scope where the code is aware
/// that ISL can return errors even when all input is valid. After leaving the
/// scope, it will return to the error setting as it was before. That also means
/// that the error setting should not be changed while in that scope.
///
/// Such scopes are not allowed to be nested because the previous operations
/// counter cannot be reset to the previous state, or one that adds the
/// operations while being in the nested scope. Use therefore is only allowed
/// while currently a no operations-limit is active.
class IslMaxOperationsGuard {
private:
  /// The ISL context to set the operations limit.
  ///
  /// If set to nullptr, there is no need for any action at the end of the
  /// scope.
  isl_ctx *IslCtx;

  /// Old OnError setting; to reset to when the scope ends.
  int OldOnError;

public:
  /// Enter a max operations scope.
  ///
  /// @param IslCtx      The ISL context to set the operations limit for.
  /// @param LocalMaxOps Maximum number of operations allowed in the
  ///                    scope. If set to zero, no operations limit is enforced.
  IslMaxOperationsGuard(isl_ctx *IslCtx, unsigned long LocalMaxOps)
      : IslCtx(IslCtx) {
    assert(IslCtx);
    assert(isl_ctx_get_max_operations(IslCtx) == 0 &&
           "Nested max operations not supported");

    if (LocalMaxOps == 0) {
      // No limit on operations; also disable restoring on_error/max_operations.
      this->IslCtx = nullptr;
      return;
    }

    // Save previous state.
    OldOnError = isl_options_get_on_error(IslCtx);

    // Activate the new setting.
    isl_ctx_set_max_operations(IslCtx, LocalMaxOps);
    isl_ctx_reset_operations(IslCtx);
    isl_options_set_on_error(IslCtx, ISL_ON_ERROR_CONTINUE);
  }

  /// Leave the max operations scope.
  ~IslMaxOperationsGuard() {
    if (!IslCtx)
      return;

    assert(isl_options_get_on_error(IslCtx) == ISL_ON_ERROR_CONTINUE &&
           "Unexpected change of the on_error setting");

    // Return to the previous error setting.
    isl_ctx_set_max_operations(IslCtx, 0);
    isl_options_set_on_error(IslCtx, OldOnError);
  }
};

} // end namespace polly

#endif
