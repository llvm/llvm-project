//===- Format.h - Utilities for String Format -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for formatting strings. They are specially
// tailored to the needs of TableGen'ing op definitions and rewrite rules,
// so they are not expected to be used as widely applicable utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_FORMAT_H_
#define MLIR_TABLEGEN_FORMAT_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SMLoc.h"

namespace mlir {
namespace tblgen {
/// An owned snapshot of a binding. Names omit the leading '$'.
struct FmtBinding {
  std::string name;
  std::string replacement;
};

/// Source context attached to a formatting failure. Location arrays retain
/// Record::getLoc() ordering: the definition followed by multiclass locations.
/// Source buffers must outlive
/// diagnostic emission; the arrays and descriptions themselves are owned.
struct FmtSite {
  SmallVector<llvm::SMLoc, 1> locations;
  std::string description;
};

/// An owned definition and its use sites, attached as an error propagates.
class FmtOrigin {
public:
  static FmtOrigin at(ArrayRef<llvm::SMLoc> locations,
                      const Twine &description);
  static FmtOrigin describe(const Twine &description) {
    return at({}, description);
  }
  FmtOrigin &withUse(ArrayRef<llvm::SMLoc> locations, const Twine &description);
  const FmtSite &getDefinition() const { return definition; }
  ArrayRef<FmtSite> getUses() const { return uses; }

private:
  FmtSite definition;
  SmallVector<FmtSite, 1> uses;
};

/// Why a placeholder could not be expanded in a formatting invocation.
enum class FmtIssueKind {
  MissingNamedBinding,
  PositionalOutOfRange,
  RangeStartOutOfRange,
  InvalidPositionalIndex
};

/// One failed occurrence. Offsets count bytes in the original template.
struct FmtIssue {
  FmtIssueKind kind;
  std::string placeholder;
  size_t offset;
};

/// A failed invocation, including all failed occurrences in template order.
/// All textual data is owned, independently of the template and context.
class FmtError : public llvm::ErrorInfo<FmtError> {
public:
  static char ID;
  FmtError(StringRef format, SmallVector<FmtIssue> issues,
           SmallVector<FmtBinding> bindings, size_t numPositionalArgs);
  StringRef getTemplate() const { return format; }
  ArrayRef<FmtIssue> getIssues() const { return issues; }
  ArrayRef<FmtBinding> getBindings() const { return bindings; }
  size_t getNumPositionalArgs() const { return numPositionalArgs; }
  ArrayRef<FmtOrigin> getOrigins() const { return origins; }
  void appendOrigin(const FmtOrigin &origin) { origins.push_back(origin); }
  void printBindings(raw_ostream &os) const;
  void log(raw_ostream &os) const override;
  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }

private:
  std::string format;
  SmallVector<FmtIssue> issues;
  SmallVector<FmtBinding> bindings;
  size_t numPositionalArgs;
  SmallVector<FmtOrigin> origins;
};

/// Append owned context without changing the error's type. Other errors pass
/// through unchanged, including non-formatting leaves in an ErrorList.
llvm::Error addFmtOrigin(llvm::Error error, const FmtOrigin &origin);

/// Format context containing substitutions for special placeholders.
///
/// Builtin placeholders have convenience setters and share storage and lookup
/// with custom placeholders. They can be used in all dialects. Their names
/// follow the convention of `$_<name>`. The rationale of the leading underscore
/// is to avoid confusion and name collision: op arguments/attributes/results
/// are named as $<name>, and we can potentially support referencing those
/// entities directly in the format template in the future.
///
/// Custom placeholders are registered by dialect-specific TableGen backends.
/// Registration and lookup names omit the leading '$'.
class FmtContext {
public:
  // Placeholder kinds
  enum class PHKind : char {
    None,
    Custom,  // For custom placeholders
    Builder, // For the $_builder placeholder
    Self,    // For the $_self placeholder
  };

  FmtContext() = default;

  // Create a format context with a list of substitutions.
  FmtContext(ArrayRef<std::pair<StringRef, StringRef>> subs);

  // Setter for custom placeholders
  FmtContext &addSubst(StringRef placeholder, const Twine &subst);

  // Setters for builtin placeholders
  FmtContext &withBuilder(Twine subst);
  FmtContext &withSelf(Twine subst);

  std::optional<StringRef> getSubstFor(PHKind placeholder) const;
  std::optional<StringRef> getSubstFor(StringRef placeholder) const;

  static PHKind getPlaceHolderKind(StringRef str);

  /// Sorted, owned snapshots, including builtin bindings and empty
  /// replacements.
  SmallVector<FmtBinding> getBindings() const;
  void printBindings(raw_ostream &os) const;

private:
  llvm::StringMap<std::string> substMap;
};

/// Struct representing a replacement segment for the formatted string. It can
/// be a segment of the formatting template (for `Literal`) or a replacement
/// parameter (for `PositionalPH`, `PositionalRangePH` and `SpecialPH`).
struct FmtReplacement {
  enum class Type {
    Empty,
    Literal,
    PositionalPH,
    PositionalRangePH,
    InvalidPH,
    SpecialPH
  };

  FmtReplacement() = default;
  explicit FmtReplacement(StringRef literal)
      : type(Type::Literal), spec(literal) {}
  FmtReplacement(StringRef spec, size_t index)
      : type(Type::PositionalPH), spec(spec), index(index) {}
  FmtReplacement(StringRef spec, size_t index, size_t end)
      : type(Type::PositionalRangePH), spec(spec), index(index), end(end) {}
  FmtReplacement(StringRef spec, FmtContext::PHKind placeholder)
      : type(Type::SpecialPH), spec(spec), placeholder(placeholder) {}

  Type type = Type::Empty;
  StringRef spec;
  size_t index = 0;
  size_t end = kUnset;
  FmtContext::PHKind placeholder = FmtContext::PHKind::None;

  static constexpr size_t kUnset = -1;
};

class FmtObjectBase {
private:
  static std::pair<FmtReplacement, StringRef> splitFmtSegment(StringRef fmt);
  static std::vector<FmtReplacement> parseFormatString(StringRef fmt);

protected:
  // The parameters are stored in a std::tuple, which does not provide runtime
  // indexing capabilities.  In order to enable runtime indexing, we use this
  // structure to put the parameters into a std::vector.  Since the parameters
  // are not all the same type, we use some type-erasure by wrapping the
  // parameters in a template class that derives from a non-template superclass.
  // Essentially, we are converting a std::tuple<Derived<Ts...>> to a
  // std::vector<Base*>.
  struct CreateAdapters {
    template <typename... Ts>
    std::vector<llvm::support::detail::FormatFunctorRef>
    operator()(Ts &...items) {
      return std::vector<llvm::support::detail::FormatFunctorRef>{items...};
    }
  };

  StringRef fmt;
  const FmtContext *context;
  std::vector<llvm::support::detail::FormatFunctorRef> adapters;
  std::vector<FmtReplacement> replacements;

public:
  FmtObjectBase(StringRef fmt, const FmtContext *ctx, size_t numParams)
      : fmt(fmt), context(ctx), replacements(parseFormatString(fmt)) {}

  FmtObjectBase(const FmtObjectBase &that) = delete;

  FmtObjectBase(FmtObjectBase &&that)
      : fmt(that.fmt), context(that.context),
        adapters(), // adapters are initialized by FmtObject
        replacements(std::move(that.replacements)) {}

  void format(llvm::raw_ostream &s) const;

  /// Validate the complete invocation before rendering any positional adapter.
  /// Unlike str(), this never encodes failure in the returned string.
  llvm::Expected<std::string> strChecked() const;

  std::string str() const {
    std::string result;
    llvm::raw_string_ostream s(result);
    format(s);
    return s.str();
  }

  template <unsigned N>
  SmallString<N> sstr() const {
    SmallString<N> result;
    llvm::raw_svector_ostream s(result);
    format(s);
    return result;
  }

  template <unsigned N>
  operator SmallString<N>() const {
    return sstr<N>();
  }

  operator std::string() const { return str(); }
};

template <typename Tuple>
class FmtObject : public FmtObjectBase {
  // Storage for the parameter adapters.  Since the base class erases the type
  // of the parameters, we have to own the storage for the parameters here, and
  // have the base class store type-erased pointers into this tuple.
  Tuple parameters;

public:
  FmtObject(StringRef fmt, const FmtContext *ctx, Tuple &&params)
      : FmtObjectBase(fmt, ctx, std::tuple_size<Tuple>::value),
        parameters(std::move(params)) {
    adapters.reserve(std::tuple_size<Tuple>::value);
    adapters = std::apply(CreateAdapters(), parameters);
  }

  FmtObject(FmtObject const &that) = delete;

  FmtObject(FmtObject &&that)
      : FmtObjectBase(std::move(that)), parameters(std::move(that.parameters)) {
    adapters.reserve(that.adapters.size());
    adapters = std::apply(CreateAdapters(), parameters);
  }
};

class FmtStrVecObject : public FmtObjectBase {
public:
  using StrFormatAdapter = llvm::support::detail::FormatFunctor<std::string>;

  FmtStrVecObject(StringRef fmt, const FmtContext *ctx,
                  ArrayRef<std::string> params);
  FmtStrVecObject(FmtStrVecObject const &that) = delete;
  FmtStrVecObject(FmtStrVecObject &&that);

private:
  SmallVector<StrFormatAdapter, 16> parameters;
};

/// Formats text by substituting placeholders in format string with replacement
/// parameters.
///
/// There are two categories of placeholders accepted, both led by a '$' sign:
///
/// 1.a Positional placeholder: $[0-9]+
/// 1.b Positional range placeholder: $[0-9]+...
/// 2. Special placeholder:    $[a-zA-Z_][a-zA-Z0-9_]*
///
/// Replacement parameters for positional placeholders are supplied as the
/// `vals` parameter pack with 1:1 mapping. That is, $0 will be replaced by the
/// first parameter in `vals`, $1 by the second one, and so on. Note that you
/// can use the positional placeholders in any order and repeat any times, for
/// example, "$2 $1 $1 $0" is accepted.
///
/// Replace parameters for positional range placeholders are supplied as if
/// positional placeholders were specified with commas separating them.
///
/// Replacement parameters for special placeholders are supplied using the `ctx`
/// format context.
///
/// The `fmt` is recorded as a `StringRef` inside the returned `FmtObject`.
/// The caller needs to make sure the underlying data is available when the
/// `FmtObject` is used.
///
/// `ctx` accepts a nullptr if there is no special placeholder is used.
///
/// If no substitution is provided for a placeholder or any error happens during
/// format string parsing or replacement, the placeholder will be outputted
/// as-is with an additional marker '<no-subst-found>', to aid debugging.
///
/// To print a '$' literally, escape it with '$$'.
/// Replacement text is emitted literally, without recursive expansion.
///
/// This utility function is inspired by LLVM formatv(), with modifications
/// specially tailored for TableGen C++ generation usage:
///
/// 1. This utility use '$' instead of '{' and '}' for denoting the placeholder
///    because '{' and '}' are frequently used in C++ code.
/// 2. This utility does not support format layout because it is rarely needed
///    in C++ code generation.
template <typename... Ts>
inline auto tgfmt(StringRef fmt, const FmtContext *ctx, Ts &&...vals)
    -> FmtObject<decltype(std::make_tuple(
        llvm::support::detail::FormatFunctor<Ts>(std::forward<Ts>(vals))...))> {
  using ParamTuple = decltype(std::make_tuple(
      llvm::support::detail::FormatFunctor<Ts>(std::forward<Ts>(vals))...));
  return FmtObject<ParamTuple>(
      fmt, ctx,
      std::make_tuple(
          llvm::support::detail::FormatFunctor<Ts>(std::forward<Ts>(vals))...));
}

/// Like tgfmt, but take replacement parameters as an array of strings instead.
inline FmtStrVecObject tgfmtv(StringRef fmt, const FmtContext *ctx,
                              ArrayRef<std::string> params) {
  return FmtStrVecObject(fmt, ctx, params);
}

/// Eager formatting that validates every placeholder before invoking positional
/// formatting providers. Returns FmtError if:
/// - a named placeholder has no binding in ctx;
/// - a decimal positional index cannot be represented as size_t;
/// - a positional placeholder's index is at least the argument count; or
/// - a positional range starts at or beyond the argument count, including an
///   empty range.
///
/// Empty replacements and unused bindings or arguments are allowed. Replacement
/// text is emitted literally, without recursive expansion. Failures inside
/// positional formatting providers are outside this validation.
///
/// On failure, FmtError owns the template, all failed occurrences and their
/// byte offsets, a snapshot of named bindings, and the positional argument
/// count. No partial output is returned on validation failure. Successful
/// results own their text; neither result nor error borrows the input template,
/// context, or arguments.
template <typename... Ts>
llvm::Expected<std::string> tgfmtChecked(StringRef fmt, const FmtContext &ctx,
                                         Ts &&...vals) {
  return tgfmt(fmt, &ctx, std::forward<Ts>(vals)...).strChecked();
}

/// Array form of tgfmtChecked, with the same validation and ownership.
inline llvm::Expected<std::string> tgfmtvChecked(StringRef fmt,
                                                 const FmtContext &ctx,
                                                 ArrayRef<std::string> params) {
  return tgfmtv(fmt, &ctx, params).strChecked();
}
} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_FORMAT_H_
