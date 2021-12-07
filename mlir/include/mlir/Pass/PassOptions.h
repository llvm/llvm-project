//===- PassOptions.h - Pass Option Utilities --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utilities for registering options with compiler passes and
// pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PASS_PASSOPTIONS_H_
#define MLIR_PASS_PASSOPTIONS_H_

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include <memory>

namespace mlir {
namespace detail {
/// Base container class and manager for all pass options.
class PassOptions : protected llvm::cl::SubCommand {
private:
  /// This is the type-erased option base class. This provides some additional
  /// hooks into the options that are not available via llvm::cl::Option.
  class OptionBase {
  public:
    virtual ~OptionBase() = default;

    /// Out of line virtual function to provide home for the class.
    virtual void anchor();

    /// Print the name and value of this option to the given stream.
    virtual void print(raw_ostream &os) = 0;

    /// Return the argument string of this option.
    StringRef getArgStr() const { return getOption()->ArgStr; }

    /// Returns true if this option has any value assigned to it.
    bool hasValue() const { return optHasValue; }

  protected:
    /// Return the main option instance.
    virtual const llvm::cl::Option *getOption() const = 0;

    /// Copy the value from the given option into this one.
    virtual void copyValueFrom(const OptionBase &other) = 0;

    /// Flag indicating if this option has a value.
    bool optHasValue = false;

    /// Allow access to private methods.
    friend PassOptions;
  };

  /// This is the parser that is used by pass options that use literal options.
  /// This is a thin wrapper around the llvm::cl::parser, that exposes some
  /// additional methods.
  template <typename DataType>
  struct GenericOptionParser : public llvm::cl::parser<DataType> {
    using llvm::cl::parser<DataType>::parser;

    /// Returns an argument name that maps to the specified value.
    Optional<StringRef> findArgStrForValue(const DataType &value) {
      for (auto &it : this->Values)
        if (it.V.compare(value))
          return it.Name;
      return llvm::None;
    }
  };

  /// Utility methods for printing option values.
  template <typename DataT>
  static void printValue(raw_ostream &os, GenericOptionParser<DataT> &parser,
                         const DataT &value) {
    if (Optional<StringRef> argStr = parser.findArgStrForValue(value))
      os << argStr;
    else
      llvm_unreachable("unknown data value for option");
  }
  template <typename DataT, typename ParserT>
  static void printValue(raw_ostream &os, ParserT &parser, const DataT &value) {
    os << value;
  }
  template <typename ParserT>
  static void printValue(raw_ostream &os, ParserT &parser, const bool &value) {
    os << (value ? StringRef("true") : StringRef("false"));
  }

public:
  /// The specific parser to use depending on llvm::cl parser used. This is only
  /// necessary because we need to provide additional methods for certain data
  /// type parsers.
  /// TODO: We should upstream the methods in GenericOptionParser to avoid the
  /// need to do this.
  template <typename DataType>
  using OptionParser =
      std::conditional_t<std::is_base_of<llvm::cl::generic_parser_base,
                                         llvm::cl::parser<DataType>>::value,
                         GenericOptionParser<DataType>,
                         llvm::cl::parser<DataType>>;

  /// This class represents a specific pass option, with a provided data type.
  template <typename DataType, typename OptionParser = OptionParser<DataType>>
  class Option
      : public llvm::cl::opt<DataType, /*ExternalStorage=*/false, OptionParser>,
        public OptionBase {
  public:
    template <typename... Args>
    Option(PassOptions &parent, StringRef arg, Args &&... args)
        : llvm::cl::opt<DataType, /*ExternalStorage=*/false, OptionParser>(
              arg, llvm::cl::sub(parent), std::forward<Args>(args)...) {
      assert(!this->isPositional() && !this->isSink() &&
             "sink and positional options are not supported");
      parent.options.push_back(this);

      // Set a callback to track if this option has a value.
      this->setCallback([this](const auto &) { this->optHasValue = true; });
    }
    ~Option() override = default;
    using llvm::cl::opt<DataType, /*ExternalStorage=*/false,
                        OptionParser>::operator=;
    Option &operator=(const Option &other) {
      *this = other.getValue();
      return *this;
    }

  private:
    /// Return the main option instance.
    const llvm::cl::Option *getOption() const final { return this; }

    /// Print the name and value of this option to the given stream.
    void print(raw_ostream &os) final {
      os << this->ArgStr << '=';
      printValue(os, this->getParser(), this->getValue());
    }

    /// Copy the value from the given option into this one.
    void copyValueFrom(const OptionBase &other) final {
      this->setValue(static_cast<const Option<DataType, OptionParser> &>(other)
                         .getValue());
      optHasValue = other.optHasValue;
    }
  };

  /// This class represents a specific pass option that contains a list of
  /// values of the provided data type.
  template <typename DataType, typename OptionParser = OptionParser<DataType>>
  class ListOption
      : public llvm::cl::list<DataType, /*StorageClass=*/bool, OptionParser>,
        public OptionBase {
  public:
    template <typename... Args>
    ListOption(PassOptions &parent, StringRef arg, Args &&... args)
        : llvm::cl::list<DataType, /*StorageClass=*/bool, OptionParser>(
              arg, llvm::cl::sub(parent), std::forward<Args>(args)...) {
      assert(!this->isPositional() && !this->isSink() &&
             "sink and positional options are not supported");
      parent.options.push_back(this);

      // Set a callback to track if this option has a value.
      this->setCallback([this](const auto &) { this->optHasValue = true; });
    }
    ~ListOption() override = default;
    ListOption<DataType, OptionParser> &
    operator=(const ListOption<DataType, OptionParser> &other) {
      *this = ArrayRef<DataType>(other);
      this->optHasValue = other.optHasValue;
      return *this;
    }

    /// Allow assigning from an ArrayRef.
    ListOption<DataType, OptionParser> &operator=(ArrayRef<DataType> values) {
      ((std::vector<DataType> &)*this).assign(values.begin(), values.end());
      optHasValue = true;
      return *this;
    }

    /// Allow accessing the data held by this option.
    MutableArrayRef<DataType> operator*() {
      return static_cast<std::vector<DataType> &>(*this);
    }
    ArrayRef<DataType> operator*() const {
      return static_cast<const std::vector<DataType> &>(*this);
    }

  private:
    /// Return the main option instance.
    const llvm::cl::Option *getOption() const final { return this; }

    /// Print the name and value of this option to the given stream.
    void print(raw_ostream &os) final {
      // Don't print the list if empty. An empty option value can be treated as
      // an element of the list in certain cases (e.g. ListOption<std::string>).
      if ((**this).empty())
        return;

      os << this->ArgStr << '=';
      auto printElementFn = [&](const DataType &value) {
        printValue(os, this->getParser(), value);
      };
      llvm::interleave(*this, os, printElementFn, ",");
    }

    /// Copy the value from the given option into this one.
    void copyValueFrom(const OptionBase &other) final {
      *this = static_cast<const ListOption<DataType, OptionParser> &>(other);
    }
  };

  PassOptions() = default;
  /// Delete the copy constructor to avoid copying the internal options map.
  PassOptions(const PassOptions &) = delete;
  PassOptions(PassOptions &&) = delete;

  /// Copy the option values from 'other' into 'this', where 'other' has the
  /// same options as 'this'.
  void copyOptionValuesFrom(const PassOptions &other);

  /// Parse options out as key=value pairs that can then be handed off to the
  /// `llvm::cl` command line passing infrastructure. Everything is space
  /// separated.
  LogicalResult parseFromString(StringRef options);

  /// Print the options held by this struct in a form that can be parsed via
  /// 'parseFromString'.
  void print(raw_ostream &os);

  /// Print the help string for the options held by this struct. `descIndent` is
  /// the indent that the descriptions should be aligned.
  void printHelp(size_t indent, size_t descIndent) const;

  /// Return the maximum width required when printing the help string.
  size_t getOptionWidth() const;

private:
  /// A list of all of the opaque options.
  std::vector<OptionBase *> options;
};
} // namespace detail

//===----------------------------------------------------------------------===//
// PassPipelineOptions
//===----------------------------------------------------------------------===//

/// Subclasses of PassPipelineOptions provide a set of options that can be used
/// to initialize a pass pipeline. See PassPipelineRegistration for usage
/// details.
///
/// Usage:
///
/// struct MyPipelineOptions : PassPipelineOptions<MyPassOptions> {
///   ListOption<int> someListFlag{
///        *this, "flag-name", llvm::cl::MiscFlags::CommaSeparated,
///        llvm::cl::desc("...")};
/// };
template <typename T> class PassPipelineOptions : public detail::PassOptions {
public:
  /// Factory that parses the provided options and returns a unique_ptr to the
  /// struct.
  static std::unique_ptr<T> createFromString(StringRef options) {
    auto result = std::make_unique<T>();
    if (failed(result->parseFromString(options)))
      return nullptr;
    return result;
  }
};

/// A default empty option struct to be used for passes that do not need to take
/// any options.
struct EmptyPipelineOptions : public PassPipelineOptions<EmptyPipelineOptions> {
};

} // namespace mlir

#endif // MLIR_PASS_PASSOPTIONS_H_

