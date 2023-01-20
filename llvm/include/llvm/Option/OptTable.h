//===- OptTable.h - Option Table --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPTION_OPTTABLE_H
#define LLVM_OPTION_OPTTABLE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/OptSpecifier.h"
#include "llvm/Support/StringSaver.h"
#include <cassert>
#include <string>
#include <vector>

namespace llvm {

class raw_ostream;
template <typename Fn> class function_ref;

namespace opt {

class Arg;
class ArgList;
class InputArgList;
class Option;

/// Provide access to the Option info table.
///
/// The OptTable class provides a layer of indirection which allows Option
/// instance to be created lazily. In the common case, only a few options will
/// be needed at runtime; the OptTable class maintains enough information to
/// parse command lines without instantiating Options, while letting other
/// parts of the driver still use Option instances where convenient.
class OptTable {
public:
  /// Entry for a single option instance in the option data table.
  struct Info {
    /// A null terminated array of prefix strings to apply to name while
    /// matching.
    ArrayRef<StringLiteral> Prefixes;
    StringRef Name;
    const char *HelpText;
    const char *MetaVar;
    unsigned ID;
    unsigned char Kind;
    unsigned char Param;
    unsigned int Flags;
    unsigned short GroupID;
    unsigned short AliasID;
    const char *AliasArgs;
    const char *Values;
  };

private:
  /// The option information table.
  ArrayRef<Info> OptionInfos;
  bool IgnoreCase;
  bool GroupedShortOptions = false;
  const char *EnvVar = nullptr;

  unsigned InputOptionID = 0;
  unsigned UnknownOptionID = 0;

protected:
  /// The index of the first option which can be parsed (i.e., is not a
  /// special option like 'input' or 'unknown', and is not an option group).
  unsigned FirstSearchableIndex = 0;

  /// The union of the first element of all option prefixes.
  SmallString<8> PrefixChars;

  /// The union of all option prefixes. If an argument does not begin with
  /// one of these, it is an input.
  virtual ArrayRef<StringLiteral> getPrefixesUnion() const = 0;

private:
  const Info &getInfo(OptSpecifier Opt) const {
    unsigned id = Opt.getID();
    assert(id > 0 && id - 1 < getNumOptions() && "Invalid Option ID.");
    return OptionInfos[id - 1];
  }

  std::unique_ptr<Arg> parseOneArgGrouped(InputArgList &Args,
                                          unsigned &Index) const;

protected:
  /// Initialize OptTable using Tablegen'ed OptionInfos. Child class must
  /// manually call \c buildPrefixChars once they are fully constructed.
  OptTable(ArrayRef<Info> OptionInfos, bool IgnoreCase = false);

  /// Build (or rebuild) the PrefixChars member.
  void buildPrefixChars();

public:
  virtual ~OptTable();

  /// Return the total number of option classes.
  unsigned getNumOptions() const { return OptionInfos.size(); }

  /// Get the given Opt's Option instance, lazily creating it
  /// if necessary.
  ///
  /// \return The option, or null for the INVALID option id.
  const Option getOption(OptSpecifier Opt) const;

  /// Lookup the name of the given option.
  StringRef getOptionName(OptSpecifier id) const { return getInfo(id).Name; }

  /// Get the kind of the given option.
  unsigned getOptionKind(OptSpecifier id) const {
    return getInfo(id).Kind;
  }

  /// Get the group id for the given option.
  unsigned getOptionGroupID(OptSpecifier id) const {
    return getInfo(id).GroupID;
  }

  /// Get the help text to use to describe this option.
  const char *getOptionHelpText(OptSpecifier id) const {
    return getInfo(id).HelpText;
  }

  /// Get the meta-variable name to use when describing
  /// this options values in the help text.
  const char *getOptionMetaVar(OptSpecifier id) const {
    return getInfo(id).MetaVar;
  }

  /// Specify the environment variable where initial options should be read.
  void setInitialOptionsFromEnvironment(const char *E) { EnvVar = E; }

  /// Support grouped short options. e.g. -ab represents -a -b.
  void setGroupedShortOptions(bool Value) { GroupedShortOptions = Value; }

  /// Find possible value for given flags. This is used for shell
  /// autocompletion.
  ///
  /// \param [in] Option - Key flag like "-stdlib=" when "-stdlib=l"
  /// was passed to clang.
  ///
  /// \param [in] Arg - Value which we want to autocomplete like "l"
  /// when "-stdlib=l" was passed to clang.
  ///
  /// \return The vector of possible values.
  std::vector<std::string> suggestValueCompletions(StringRef Option,
                                                   StringRef Arg) const;

  /// Find flags from OptTable which starts with Cur.
  ///
  /// \param [in] Cur - String prefix that all returned flags need
  //  to start with.
  ///
  /// \return The vector of flags which start with Cur.
  std::vector<std::string> findByPrefix(StringRef Cur,
                                        unsigned int DisableFlags) const;

  /// Find the OptTable option that most closely matches the given string.
  ///
  /// \param [in] Option - A string, such as "-stdlibs=l", that represents user
  /// input of an option that may not exist in the OptTable. Note that the
  /// string includes prefix dashes "-" as well as values "=l".
  /// \param [out] NearestString - The nearest option string found in the
  /// OptTable.
  /// \param [in] FlagsToInclude - Only find options with any of these flags.
  /// Zero is the default, which includes all flags.
  /// \param [in] FlagsToExclude - Don't find options with this flag. Zero
  /// is the default, and means exclude nothing.
  /// \param [in] MinimumLength - Don't find options shorter than this length.
  /// For example, a minimum length of 3 prevents "-x" from being considered
  /// near to "-S".
  /// \param [in] MaximumDistance - Don't find options whose distance is greater
  /// than this value.
  ///
  /// \return The edit distance of the nearest string found.
  unsigned findNearest(StringRef Option, std::string &NearestString,
                       unsigned FlagsToInclude = 0, unsigned FlagsToExclude = 0,
                       unsigned MinimumLength = 4,
                       unsigned MaximumDistance = UINT_MAX - 1) const;

  bool findExact(StringRef Option, std::string &ExactString,
                 unsigned FlagsToInclude = 0,
                 unsigned FlagsToExclude = 0) const {
    return findNearest(Option, ExactString, FlagsToInclude, FlagsToExclude, 4,
                       0) == 0;
  }

  /// Parse a single argument; returning the new argument and
  /// updating Index.
  ///
  /// \param [in,out] Index - The current parsing position in the argument
  /// string list; on return this will be the index of the next argument
  /// string to parse.
  /// \param [in] FlagsToInclude - Only parse options with any of these flags.
  /// Zero is the default which includes all flags.
  /// \param [in] FlagsToExclude - Don't parse options with this flag.  Zero
  /// is the default and means exclude nothing.
  ///
  /// \return The parsed argument, or 0 if the argument is missing values
  /// (in which case Index still points at the conceptual next argument string
  /// to parse).
  std::unique_ptr<Arg> ParseOneArg(const ArgList &Args, unsigned &Index,
                                   unsigned FlagsToInclude = 0,
                                   unsigned FlagsToExclude = 0) const;

  /// Parse an list of arguments into an InputArgList.
  ///
  /// The resulting InputArgList will reference the strings in [\p ArgBegin,
  /// \p ArgEnd), and their lifetime should extend past that of the returned
  /// InputArgList.
  ///
  /// The only error that can occur in this routine is if an argument is
  /// missing values; in this case \p MissingArgCount will be non-zero.
  ///
  /// \param MissingArgIndex - On error, the index of the option which could
  /// not be parsed.
  /// \param MissingArgCount - On error, the number of missing options.
  /// \param FlagsToInclude - Only parse options with any of these flags.
  /// Zero is the default which includes all flags.
  /// \param FlagsToExclude - Don't parse options with this flag.  Zero
  /// is the default and means exclude nothing.
  /// \return An InputArgList; on error this will contain all the options
  /// which could be parsed.
  InputArgList ParseArgs(ArrayRef<const char *> Args, unsigned &MissingArgIndex,
                         unsigned &MissingArgCount, unsigned FlagsToInclude = 0,
                         unsigned FlagsToExclude = 0) const;

  /// A convenience helper which handles optional initial options populated from
  /// an environment variable, expands response files recursively and parses
  /// options.
  ///
  /// \param ErrorFn - Called on a formatted error message for missing arguments
  /// or unknown options.
  /// \return An InputArgList; on error this will contain all the options which
  /// could be parsed.
  InputArgList parseArgs(int Argc, char *const *Argv, OptSpecifier Unknown,
                         StringSaver &Saver,
                         function_ref<void(StringRef)> ErrorFn) const;

  /// Render the help text for an option table.
  ///
  /// \param OS - The stream to write the help text to.
  /// \param Usage - USAGE: Usage
  /// \param Title - OVERVIEW: Title
  /// \param FlagsToInclude - If non-zero, only include options with any
  ///                         of these flags set.
  /// \param FlagsToExclude - Exclude options with any of these flags set.
  /// \param ShowAllAliases - If true, display all options including aliases
  ///                         that don't have help texts. By default, we display
  ///                         only options that are not hidden and have help
  ///                         texts.
  void printHelp(raw_ostream &OS, const char *Usage, const char *Title,
                 unsigned FlagsToInclude, unsigned FlagsToExclude,
                 bool ShowAllAliases) const;

  void printHelp(raw_ostream &OS, const char *Usage, const char *Title,
                 bool ShowHidden = false, bool ShowAllAliases = false) const;
};

/// Specialization of OptTable
class GenericOptTable : public OptTable {
  SmallVector<StringLiteral> PrefixesUnionBuffer;

protected:
  GenericOptTable(ArrayRef<Info> OptionInfos, bool IgnoreCase = false);
  ArrayRef<StringLiteral> getPrefixesUnion() const final {
    return PrefixesUnionBuffer;
  }
};

class PrecomputedOptTable : public OptTable {
  ArrayRef<StringLiteral> PrefixesUnion;

protected:
  PrecomputedOptTable(ArrayRef<Info> OptionInfos,
                      ArrayRef<StringLiteral> PrefixesTable,
                      bool IgnoreCase = false)
      : OptTable(OptionInfos, IgnoreCase), PrefixesUnion(PrefixesTable) {
    buildPrefixChars();
  }
  ArrayRef<StringLiteral> getPrefixesUnion() const final {
    return PrefixesUnion;
  }
};

} // end namespace opt

} // end namespace llvm

#endif // LLVM_OPTION_OPTTABLE_H
