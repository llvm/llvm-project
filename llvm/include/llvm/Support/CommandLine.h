//===- llvm/Support/CommandLine.h - Command line handler --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements a command line argument processor that is useful when
// creating a tool.  It provides a simple, minimalistic interface that is easily
// extensible and supports nonlocal (library) command line options.
//
// Note that rather than trying to figure out what this code does, you should
// read the library documentation located in docs/CommandLine.html or looks at
// the many example usages in tools/*/*.cpp
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_COMMANDLINE_H
#define LLVM_SUPPORT_COMMANDLINE_H

#include "llvm/Support/type_traits.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include <cassert>
#include <climits>
#include <cstdarg>
#include <utility>
#include <vector>

namespace llvm {

/// cl Namespace - This namespace contains all of the command line option
/// processing machinery.  It is intentionally a short name to make qualified
/// usage concise.
namespace cl {

//===----------------------------------------------------------------------===//
// ParseCommandLineOptions - Command line option processing entry point.
//
void ParseCommandLineOptions(int argc, const char * const *argv,
                             const char *Overview = 0);

//===----------------------------------------------------------------------===//
// ParseEnvironmentOptions - Environment variable option processing alternate
//                           entry point.
//
void ParseEnvironmentOptions(const char *progName, const char *envvar,
                             const char *Overview = 0);

///===---------------------------------------------------------------------===//
/// SetVersionPrinter - Override the default (LLVM specific) version printer
///                     used to print out the version when --version is given
///                     on the command line. This allows other systems using the
///                     CommandLine utilities to print their own version string.
void SetVersionPrinter(void (*func)());

///===---------------------------------------------------------------------===//
/// AddExtraVersionPrinter - Add an extra printer to use in addition to the
///                          default one. This can be called multiple times,
///                          and each time it adds a new function to the list
///                          which will be called after the basic LLVM version
///                          printing is complete. Each can then add additional
///                          information specific to the tool.
void AddExtraVersionPrinter(void (*func)());


// PrintOptionValues - Print option values.
// With -print-options print the difference between option values and defaults.
// With -print-all-options print all option values.
// (Currently not perfect, but best-effort.)
void PrintOptionValues();

// MarkOptionsChanged - Internal helper function.
void MarkOptionsChanged();

//===----------------------------------------------------------------------===//
// Flags permitted to be passed to command line arguments
//

enum NumOccurrencesFlag {      // Flags for the number of occurrences allowed
  Optional        = 0x00,      // Zero or One occurrence
  ZeroOrMore      = 0x01,      // Zero or more occurrences allowed
  Required        = 0x02,      // One occurrence required
  OneOrMore       = 0x03,      // One or more occurrences required

  // ConsumeAfter - Indicates that this option is fed anything that follows the
  // last positional argument required by the application (it is an error if
  // there are zero positional arguments, and a ConsumeAfter option is used).
  // Thus, for example, all arguments to LLI are processed until a filename is
  // found.  Once a filename is found, all of the succeeding arguments are
  // passed, unprocessed, to the ConsumeAfter option.
  //
  ConsumeAfter    = 0x04
};

enum ValueExpected {           // Is a value required for the option?
  // zero reserved for the unspecified value
  ValueOptional   = 0x01,      // The value can appear... or not
  ValueRequired   = 0x02,      // The value is required to appear!
  ValueDisallowed = 0x03       // A value may not be specified (for flags)
};

enum OptionHidden {            // Control whether -help shows this option
  NotHidden       = 0x00,      // Option included in -help & -help-hidden
  Hidden          = 0x01,      // -help doesn't, but -help-hidden does
  ReallyHidden    = 0x02       // Neither -help nor -help-hidden show this arg
};

// Formatting flags - This controls special features that the option might have
// that cause it to be parsed differently...
//
// Prefix - This option allows arguments that are otherwise unrecognized to be
// matched by options that are a prefix of the actual value.  This is useful for
// cases like a linker, where options are typically of the form '-lfoo' or
// '-L../../include' where -l or -L are the actual flags.  When prefix is
// enabled, and used, the value for the flag comes from the suffix of the
// argument.
//
// Grouping - With this option enabled, multiple letter options are allowed to
// bunch together with only a single hyphen for the whole group.  This allows
// emulation of the behavior that ls uses for example: ls -la === ls -l -a
//

enum FormattingFlags {
  NormalFormatting = 0x00,     // Nothing special
  Positional       = 0x01,     // Is a positional argument, no '-' required
  Prefix           = 0x02,     // Can this option directly prefix its value?
  Grouping         = 0x03      // Can this option group with other options?
};

enum MiscFlags {               // Miscellaneous flags to adjust argument
  CommaSeparated     = 0x01,  // Should this cl::list split between commas?
  PositionalEatsArgs = 0x02,  // Should this positional cl::list eat -args?
  Sink               = 0x04   // Should this cl::list eat all unknown options?
};



//===----------------------------------------------------------------------===//
// Option Base class
//
class alias;
class Option {
  friend class alias;

  // handleOccurrences - Overriden by subclasses to handle the value passed into
  // an argument.  Should return true if there was an error processing the
  // argument and the program should exit.
  //
  virtual bool handleOccurrence(unsigned pos, StringRef ArgName,
                                StringRef Arg) = 0;

  virtual enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueOptional;
  }

  // Out of line virtual function to provide home for the class.
  virtual void anchor();

  int NumOccurrences;     // The number of times specified
  // Occurrences, HiddenFlag, and Formatting are all enum types but to avoid
  // problems with signed enums in bitfields.
  unsigned Occurrences : 3; // enum NumOccurrencesFlag
  // not using the enum type for 'Value' because zero is an implementation
  // detail representing the non-value
  unsigned Value : 2;
  unsigned HiddenFlag : 2; // enum OptionHidden
  unsigned Formatting : 2; // enum FormattingFlags
  unsigned Misc : 3;
  unsigned Position;      // Position of last occurrence of the option
  unsigned AdditionalVals;// Greater than 0 for multi-valued option.
  Option *NextRegistered; // Singly linked list of registered options.
public:
  const char *ArgStr;     // The argument string itself (ex: "help", "o")
  const char *HelpStr;    // The descriptive text message for -help
  const char *ValueStr;   // String describing what the value of this option is

  inline enum NumOccurrencesFlag getNumOccurrencesFlag() const {
    return (enum NumOccurrencesFlag)Occurrences;
  }
  inline enum ValueExpected getValueExpectedFlag() const {
    return Value ? ((enum ValueExpected)Value)
              : getValueExpectedFlagDefault();
  }
  inline enum OptionHidden getOptionHiddenFlag() const {
    return (enum OptionHidden)HiddenFlag;
  }
  inline enum FormattingFlags getFormattingFlag() const {
    return (enum FormattingFlags)Formatting;
  }
  inline unsigned getMiscFlags() const {
    return Misc;
  }
  inline unsigned getPosition() const { return Position; }
  inline unsigned getNumAdditionalVals() const { return AdditionalVals; }

  // hasArgStr - Return true if the argstr != ""
  bool hasArgStr() const { return ArgStr[0] != 0; }

  //-------------------------------------------------------------------------===
  // Accessor functions set by OptionModifiers
  //
  void setArgStr(const char *S) { ArgStr = S; }
  void setDescription(const char *S) { HelpStr = S; }
  void setValueStr(const char *S) { ValueStr = S; }
  void setNumOccurrencesFlag(enum NumOccurrencesFlag Val) {
    Occurrences = Val;
  }
  void setValueExpectedFlag(enum ValueExpected Val) { Value = Val; }
  void setHiddenFlag(enum OptionHidden Val) { HiddenFlag = Val; }
  void setFormattingFlag(enum FormattingFlags V) { Formatting = V; }
  void setMiscFlag(enum MiscFlags M) { Misc |= M; }
  void setPosition(unsigned pos) { Position = pos; }
protected:
  explicit Option(enum NumOccurrencesFlag OccurrencesFlag,
                  enum OptionHidden Hidden)
    : NumOccurrences(0), Occurrences(OccurrencesFlag), Value(0),
      HiddenFlag(Hidden), Formatting(NormalFormatting), Misc(0),
      Position(0), AdditionalVals(0), NextRegistered(0),
      ArgStr(""), HelpStr(""), ValueStr("") {
  }

  inline void setNumAdditionalVals(unsigned n) { AdditionalVals = n; }
public:
  // addArgument - Register this argument with the commandline system.
  //
  void addArgument();

  Option *getNextRegisteredOption() const { return NextRegistered; }

  // Return the width of the option tag for printing...
  virtual size_t getOptionWidth() const = 0;

  // printOptionInfo - Print out information about this option.  The
  // to-be-maintained width is specified.
  //
  virtual void printOptionInfo(size_t GlobalWidth) const = 0;

  virtual void printOptionValue(size_t GlobalWidth, bool Force) const = 0;

  virtual void getExtraOptionNames(SmallVectorImpl<const char*> &) {}

  // addOccurrence - Wrapper around handleOccurrence that enforces Flags.
  //
  bool addOccurrence(unsigned pos, StringRef ArgName,
                     StringRef Value, bool MultiArg = false);

  // Prints option name followed by message.  Always returns true.
  bool error(const Twine &Message, StringRef ArgName = StringRef());

public:
  inline int getNumOccurrences() const { return NumOccurrences; }
  virtual ~Option() {}
};


//===----------------------------------------------------------------------===//
// Command line option modifiers that can be used to modify the behavior of
// command line option parsers...
//

// desc - Modifier to set the description shown in the -help output...
struct desc {
  const char *Desc;
  desc(const char *Str) : Desc(Str) {}
  void apply(Option &O) const { O.setDescription(Desc); }
};

// value_desc - Modifier to set the value description shown in the -help
// output...
struct value_desc {
  const char *Desc;
  value_desc(const char *Str) : Desc(Str) {}
  void apply(Option &O) const { O.setValueStr(Desc); }
};

// init - Specify a default (initial) value for the command line argument, if
// the default constructor for the argument type does not give you what you
// want.  This is only valid on "opt" arguments, not on "list" arguments.
//
template<class Ty>
struct initializer {
  const Ty &Init;
  initializer(const Ty &Val) : Init(Val) {}

  template<class Opt>
  void apply(Opt &O) const { O.setInitialValue(Init); }
};

template<class Ty>
initializer<Ty> init(const Ty &Val) {
  return initializer<Ty>(Val);
}


// location - Allow the user to specify which external variable they want to
// store the results of the command line argument processing into, if they don't
// want to store it in the option itself.
//
template<class Ty>
struct LocationClass {
  Ty &Loc;
  LocationClass(Ty &L) : Loc(L) {}

  template<class Opt>
  void apply(Opt &O) const { O.setLocation(O, Loc); }
};

template<class Ty>
LocationClass<Ty> location(Ty &L) { return LocationClass<Ty>(L); }


//===----------------------------------------------------------------------===//
// OptionValue class

// Support value comparison outside the template.
struct GenericOptionValue {
  virtual ~GenericOptionValue() {}
  virtual bool compare(const GenericOptionValue &V) const = 0;
private:
  virtual void anchor();
};

template<class DataType> struct OptionValue;

// The default value safely does nothing. Option value printing is only
// best-effort.
template<class DataType, bool isClass>
struct OptionValueBase : public GenericOptionValue {
  // Temporary storage for argument passing.
  typedef OptionValue<DataType> WrapperType;

  bool hasValue() const { return false; }

  const DataType &getValue() const { llvm_unreachable("no default value"); }

  // Some options may take their value from a different data type.
  template<class DT>
  void setValue(const DT& /*V*/) {}

  bool compare(const DataType &/*V*/) const { return false; }

  virtual bool compare(const GenericOptionValue& /*V*/) const { return false; }
};

// Simple copy of the option value.
template<class DataType>
class OptionValueCopy : public GenericOptionValue {
  DataType Value;
  bool Valid;
public:
  OptionValueCopy() : Valid(false) {}

  bool hasValue() const { return Valid; }

  const DataType &getValue() const {
    assert(Valid && "invalid option value");
    return Value;
  }

  void setValue(const DataType &V) { Valid = true; Value = V; }

  bool compare(const DataType &V) const {
    return Valid && (Value != V);
  }

  virtual bool compare(const GenericOptionValue &V) const {
    const OptionValueCopy<DataType> &VC =
      static_cast< const OptionValueCopy<DataType>& >(V);
    if (!VC.hasValue()) return false;
    return compare(VC.getValue());
  }
};

// Non-class option values.
template<class DataType>
struct OptionValueBase<DataType, false> : OptionValueCopy<DataType> {
  typedef DataType WrapperType;
};

// Top-level option class.
template<class DataType>
struct OptionValue : OptionValueBase<DataType, is_class<DataType>::value> {
  OptionValue() {}

  OptionValue(const DataType& V) {
    this->setValue(V);
  }
  // Some options may take their value from a different data type.
  template<class DT>
  OptionValue<DataType> &operator=(const DT& V) {
    this->setValue(V);
    return *this;
  }
};

// Other safe-to-copy-by-value common option types.
enum boolOrDefault { BOU_UNSET, BOU_TRUE, BOU_FALSE };
template<>
struct OptionValue<cl::boolOrDefault> : OptionValueCopy<cl::boolOrDefault> {
  typedef cl::boolOrDefault WrapperType;

  OptionValue() {}

  OptionValue(const cl::boolOrDefault& V) {
    this->setValue(V);
  }
  OptionValue<cl::boolOrDefault> &operator=(const cl::boolOrDefault& V) {
    setValue(V);
    return *this;
  }
private:
  virtual void anchor();
};

template<>
struct OptionValue<std::string> : OptionValueCopy<std::string> {
  typedef StringRef WrapperType;

  OptionValue() {}

  OptionValue(const std::string& V) {
    this->setValue(V);
  }
  OptionValue<std::string> &operator=(const std::string& V) {
    setValue(V);
    return *this;
  }
private:
  virtual void anchor();
};

//===----------------------------------------------------------------------===//
// Enum valued command line option
//
#define clEnumVal(ENUMVAL, DESC) #ENUMVAL, int(ENUMVAL), DESC
#define clEnumValN(ENUMVAL, FLAGNAME, DESC) FLAGNAME, int(ENUMVAL), DESC
#define clEnumValEnd (reinterpret_cast<void*>(0))

// values - For custom data types, allow specifying a group of values together
// as the values that go into the mapping that the option handler uses.  Note
// that the values list must always have a 0 at the end of the list to indicate
// that the list has ended.
//
template<class DataType>
class ValuesClass {
  // Use a vector instead of a map, because the lists should be short,
  // the overhead is less, and most importantly, it keeps them in the order
  // inserted so we can print our option out nicely.
  SmallVector<std::pair<const char *, std::pair<int, const char *> >,4> Values;
  void processValues(va_list Vals);
public:
  ValuesClass(const char *EnumName, DataType Val, const char *Desc,
              va_list ValueArgs) {
    // Insert the first value, which is required.
    Values.push_back(std::make_pair(EnumName, std::make_pair(Val, Desc)));

    // Process the varargs portion of the values...
    while (const char *enumName = va_arg(ValueArgs, const char *)) {
      DataType EnumVal = static_cast<DataType>(va_arg(ValueArgs, int));
      const char *EnumDesc = va_arg(ValueArgs, const char *);
      Values.push_back(std::make_pair(enumName,      // Add value to value map
                                      std::make_pair(EnumVal, EnumDesc)));
    }
  }

  template<class Opt>
  void apply(Opt &O) const {
    for (unsigned i = 0, e = static_cast<unsigned>(Values.size());
         i != e; ++i)
      O.getParser().addLiteralOption(Values[i].first, Values[i].second.first,
                                     Values[i].second.second);
  }
};

template<class DataType>
ValuesClass<DataType> END_WITH_NULL values(const char *Arg, DataType Val,
                                           const char *Desc, ...) {
    va_list ValueArgs;
    va_start(ValueArgs, Desc);
    ValuesClass<DataType> Vals(Arg, Val, Desc, ValueArgs);
    va_end(ValueArgs);
    return Vals;
}

//===----------------------------------------------------------------------===//
// parser class - Parameterizable parser for different data types.  By default,
// known data types (string, int, bool) have specialized parsers, that do what
// you would expect.  The default parser, used for data types that are not
// built-in, uses a mapping table to map specific options to values, which is
// used, among other things, to handle enum types.

//--------------------------------------------------
// generic_parser_base - This class holds all the non-generic code that we do
// not need replicated for every instance of the generic parser.  This also
// allows us to put stuff into CommandLine.cpp
//
class generic_parser_base {
protected:
  class GenericOptionInfo {
  public:
    GenericOptionInfo(const char *name, const char *helpStr) :
      Name(name), HelpStr(helpStr) {}
    const char *Name;
    const char *HelpStr;
  };
public:
  virtual ~generic_parser_base() {}  // Base class should have virtual-dtor

  // getNumOptions - Virtual function implemented by generic subclass to
  // indicate how many entries are in Values.
  //
  virtual unsigned getNumOptions() const = 0;

  // getOption - Return option name N.
  virtual const char *getOption(unsigned N) const = 0;

  // getDescription - Return description N
  virtual const char *getDescription(unsigned N) const = 0;

  // Return the width of the option tag for printing...
  virtual size_t getOptionWidth(const Option &O) const;

  virtual const GenericOptionValue &getOptionValue(unsigned N) const = 0;

  // printOptionInfo - Print out information about this option.  The
  // to-be-maintained width is specified.
  //
  virtual void printOptionInfo(const Option &O, size_t GlobalWidth) const;

  void printGenericOptionDiff(const Option &O, const GenericOptionValue &V,
                              const GenericOptionValue &Default,
                              size_t GlobalWidth) const;

  // printOptionDiff - print the value of an option and it's default.
  //
  // Template definition ensures that the option and default have the same
  // DataType (via the same AnyOptionValue).
  template<class AnyOptionValue>
  void printOptionDiff(const Option &O, const AnyOptionValue &V,
                       const AnyOptionValue &Default,
                       size_t GlobalWidth) const {
    printGenericOptionDiff(O, V, Default, GlobalWidth);
  }

  void initialize(Option &O) {
    // All of the modifiers for the option have been processed by now, so the
    // argstr field should be stable, copy it down now.
    //
    hasArgStr = O.hasArgStr();
  }

  void getExtraOptionNames(SmallVectorImpl<const char*> &OptionNames) {
    // If there has been no argstr specified, that means that we need to add an
    // argument for every possible option.  This ensures that our options are
    // vectored to us.
    if (!hasArgStr)
      for (unsigned i = 0, e = getNumOptions(); i != e; ++i)
        OptionNames.push_back(getOption(i));
  }


  enum ValueExpected getValueExpectedFlagDefault() const {
    // If there is an ArgStr specified, then we are of the form:
    //
    //    -opt=O2   or   -opt O2  or  -optO2
    //
    // In which case, the value is required.  Otherwise if an arg str has not
    // been specified, we are of the form:
    //
    //    -O2 or O2 or -la (where -l and -a are separate options)
    //
    // If this is the case, we cannot allow a value.
    //
    if (hasArgStr)
      return ValueRequired;
    else
      return ValueDisallowed;
  }

  // findOption - Return the option number corresponding to the specified
  // argument string.  If the option is not found, getNumOptions() is returned.
  //
  unsigned findOption(const char *Name);

protected:
  bool hasArgStr;
};

// Default parser implementation - This implementation depends on having a
// mapping of recognized options to values of some sort.  In addition to this,
// each entry in the mapping also tracks a help message that is printed with the
// command line option for -help.  Because this is a simple mapping parser, the
// data type can be any unsupported type.
//
template <class DataType>
class parser : public generic_parser_base {
protected:
  class OptionInfo : public GenericOptionInfo {
  public:
    OptionInfo(const char *name, DataType v, const char *helpStr) :
      GenericOptionInfo(name, helpStr), V(v) {}
    OptionValue<DataType> V;
  };
  SmallVector<OptionInfo, 8> Values;
public:
  typedef DataType parser_data_type;

  // Implement virtual functions needed by generic_parser_base
  unsigned getNumOptions() const { return unsigned(Values.size()); }
  const char *getOption(unsigned N) const { return Values[N].Name; }
  const char *getDescription(unsigned N) const {
    return Values[N].HelpStr;
  }

  // getOptionValue - Return the value of option name N.
  virtual const GenericOptionValue &getOptionValue(unsigned N) const {
    return Values[N].V;
  }

  // parse - Return true on error.
  bool parse(Option &O, StringRef ArgName, StringRef Arg, DataType &V) {
    StringRef ArgVal;
    if (hasArgStr)
      ArgVal = Arg;
    else
      ArgVal = ArgName;

    for (unsigned i = 0, e = static_cast<unsigned>(Values.size());
         i != e; ++i)
      if (Values[i].Name == ArgVal) {
        V = Values[i].V.getValue();
        return false;
      }

    return O.error("Cannot find option named '" + ArgVal + "'!");
  }

  /// addLiteralOption - Add an entry to the mapping table.
  ///
  template <class DT>
  void addLiteralOption(const char *Name, const DT &V, const char *HelpStr) {
    assert(findOption(Name) == Values.size() && "Option already exists!");
    OptionInfo X(Name, static_cast<DataType>(V), HelpStr);
    Values.push_back(X);
    MarkOptionsChanged();
  }

  /// removeLiteralOption - Remove the specified option.
  ///
  void removeLiteralOption(const char *Name) {
    unsigned N = findOption(Name);
    assert(N != Values.size() && "Option not found!");
    Values.erase(Values.begin()+N);
  }
};

//--------------------------------------------------
// basic_parser - Super class of parsers to provide boilerplate code
//
class basic_parser_impl {  // non-template implementation of basic_parser<t>
public:
  virtual ~basic_parser_impl() {}

  enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueRequired;
  }

  void getExtraOptionNames(SmallVectorImpl<const char*> &) {}

  void initialize(Option &) {}

  // Return the width of the option tag for printing...
  size_t getOptionWidth(const Option &O) const;

  // printOptionInfo - Print out information about this option.  The
  // to-be-maintained width is specified.
  //
  void printOptionInfo(const Option &O, size_t GlobalWidth) const;

  // printOptionNoValue - Print a placeholder for options that don't yet support
  // printOptionDiff().
  void printOptionNoValue(const Option &O, size_t GlobalWidth) const;

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "value"; }

  // An out-of-line virtual method to provide a 'home' for this class.
  virtual void anchor();

protected:
  // A helper for basic_parser::printOptionDiff.
  void printOptionName(const Option &O, size_t GlobalWidth) const;
};

// basic_parser - The real basic parser is just a template wrapper that provides
// a typedef for the provided data type.
//
template<class DataType>
class basic_parser : public basic_parser_impl {
public:
  typedef DataType parser_data_type;
  typedef OptionValue<DataType> OptVal;
};

//--------------------------------------------------
// parser<bool>
//
template<>
class parser<bool> : public basic_parser<bool> {
  const char *ArgStr;
public:

  // parse - Return true on error.
  bool parse(Option &O, StringRef ArgName, StringRef Arg, bool &Val);

  template <class Opt>
  void initialize(Opt &O) {
    ArgStr = O.ArgStr;
  }

  enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueOptional;
  }

  // getValueName - Do not print =<value> at all.
  virtual const char *getValueName() const { return 0; }

  void printOptionDiff(const Option &O, bool V, OptVal Default,
                       size_t GlobalWidth) const;

  // An out-of-line virtual method to provide a 'home' for this class.
  virtual void anchor();
};

EXTERN_TEMPLATE_INSTANTIATION(class basic_parser<bool>);

//--------------------------------------------------
// parser<boolOrDefault>
template<>
class parser<boolOrDefault> : public basic_parser<boolOrDefault> {
public:
  // parse - Return true on error.
  bool parse(Option &O, StringRef ArgName, StringRef Arg, boolOrDefault &Val);

  enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueOptional;
  }

  // getValueName - Do not print =<value> at all.
  virtual const char *getValueName() const { return 0; }

  void printOptionDiff(const Option &O, boolOrDefault V, OptVal Default,
                       size_t GlobalWidth) const;

  // An out-of-line virtual method to provide a 'home' for this class.
  virtual void anchor();
};

EXTERN_TEMPLATE_INSTANTIATION(class basic_parser<boolOrDefault>);

//--------------------------------------------------
// parser<int>
//
template<>
class parser<int> : public basic_parser<int> {
public:
  // parse - Return true on error.
  bool parse(Option &O, StringRef ArgName, StringRef Arg, int &Val);

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "int"; }

  void printOptionDiff(const Option &O, int V, OptVal Default,
                       size_t GlobalWidth) const;

  // An out-of-line virtual method to provide a 'home' for this class.
  virtual void anchor();
};

EXTERN_TEMPLATE_INSTANTIATION(class basic_parser<int>);


//--------------------------------------------------
// parser<unsigned>
//
template<>
class parser<unsigned> : public basic_parser<unsigned> {
public:
  // parse - Return true on error.
  bool parse(Option &O, StringRef ArgName, StringRef Arg, unsigned &Val);

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "uint"; }

  void printOptionDiff(const Option &O, unsigned V, OptVal Default,
                       size_t GlobalWidth) const;

  // An out-of-line virtual method to provide a 'home' for this class.
  virtual void anchor();
};

EXTERN_TEMPLATE_INSTANTIATION(class basic_parser<unsigned>);

//--------------------------------------------------
// parser<unsigned long long>
//
template<>
class parser<unsigned long long> : public basic_parser<unsigned long long> {
public:
  // parse - Return true on error.
  bool parse(Option &O, StringRef ArgName, StringRef Arg,
             unsigned long long &Val);

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "uint"; }

  void printOptionDiff(const Option &O, unsigned long long V, OptVal Default,
                       size_t GlobalWidth) const;

  // An out-of-line virtual method to provide a 'home' for this class.
  virtual void anchor();
};

EXTERN_TEMPLATE_INSTANTIATION(class basic_parser<unsigned long long>);

//--------------------------------------------------
// parser<double>
//
template<>
class parser<double> : public basic_parser<double> {
public:
  // parse - Return true on error.
  bool parse(Option &O, StringRef ArgName, StringRef Arg, double &Val);

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "number"; }

  void printOptionDiff(const Option &O, double V, OptVal Default,
                       size_t GlobalWidth) const;

  // An out-of-line virtual method to provide a 'home' for this class.
  virtual void anchor();
};

EXTERN_TEMPLATE_INSTANTIATION(class basic_parser<double>);

//--------------------------------------------------
// parser<float>
//
template<>
class parser<float> : public basic_parser<float> {
public:
  // parse - Return true on error.
  bool parse(Option &O, StringRef ArgName, StringRef Arg, float &Val);

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "number"; }

  void printOptionDiff(const Option &O, float V, OptVal Default,
                       size_t GlobalWidth) const;

  // An out-of-line virtual method to provide a 'home' for this class.
  virtual void anchor();
};

EXTERN_TEMPLATE_INSTANTIATION(class basic_parser<float>);

//--------------------------------------------------
// parser<std::string>
//
template<>
class parser<std::string> : public basic_parser<std::string> {
public:
  // parse - Return true on error.
  bool parse(Option &, StringRef, StringRef Arg, std::string &Value) {
    Value = Arg.str();
    return false;
  }

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "string"; }

  void printOptionDiff(const Option &O, StringRef V, OptVal Default,
                       size_t GlobalWidth) const;

  // An out-of-line virtual method to provide a 'home' for this class.
  virtual void anchor();
};

EXTERN_TEMPLATE_INSTANTIATION(class basic_parser<std::string>);

//--------------------------------------------------
// parser<char>
//
template<>
class parser<char> : public basic_parser<char> {
public:
  // parse - Return true on error.
  bool parse(Option &, StringRef, StringRef Arg, char &Value) {
    Value = Arg[0];
    return false;
  }

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "char"; }

  void printOptionDiff(const Option &O, char V, OptVal Default,
                       size_t GlobalWidth) const;

  // An out-of-line virtual method to provide a 'home' for this class.
  virtual void anchor();
};

EXTERN_TEMPLATE_INSTANTIATION(class basic_parser<char>);

//--------------------------------------------------
// PrintOptionDiff
//
// This collection of wrappers is the intermediary between class opt and class
// parser to handle all the template nastiness.

// This overloaded function is selected by the generic parser.
template<class ParserClass, class DT>
void printOptionDiff(const Option &O, const generic_parser_base &P, const DT &V,
                     const OptionValue<DT> &Default, size_t GlobalWidth) {
  OptionValue<DT> OV = V;
  P.printOptionDiff(O, OV, Default, GlobalWidth);
}

// This is instantiated for basic parsers when the parsed value has a different
// type than the option value. e.g. HelpPrinter.
template<class ParserDT, class ValDT>
struct OptionDiffPrinter {
  void print(const Option &O, const parser<ParserDT> P, const ValDT &/*V*/,
             const OptionValue<ValDT> &/*Default*/, size_t GlobalWidth) {
    P.printOptionNoValue(O, GlobalWidth);
  }
};

// This is instantiated for basic parsers when the parsed value has the same
// type as the option value.
template<class DT>
struct OptionDiffPrinter<DT, DT> {
  void print(const Option &O, const parser<DT> P, const DT &V,
             const OptionValue<DT> &Default, size_t GlobalWidth) {
    P.printOptionDiff(O, V, Default, GlobalWidth);
  }
};

// This overloaded function is selected by the basic parser, which may parse a
// different type than the option type.
template<class ParserClass, class ValDT>
void printOptionDiff(
  const Option &O,
  const basic_parser<typename ParserClass::parser_data_type> &P,
  const ValDT &V, const OptionValue<ValDT> &Default,
  size_t GlobalWidth) {

  OptionDiffPrinter<typename ParserClass::parser_data_type, ValDT> printer;
  printer.print(O, static_cast<const ParserClass&>(P), V, Default,
                GlobalWidth);
}

//===----------------------------------------------------------------------===//
// applicator class - This class is used because we must use partial
// specialization to handle literal string arguments specially (const char* does
// not correctly respond to the apply method).  Because the syntax to use this
// is a pain, we have the 'apply' method below to handle the nastiness...
//
template<class Mod> struct applicator {
  template<class Opt>
  static void opt(const Mod &M, Opt &O) { M.apply(O); }
};

// Handle const char* as a special case...
template<unsigned n> struct applicator<char[n]> {
  template<class Opt>
  static void opt(const char *Str, Opt &O) { O.setArgStr(Str); }
};
template<unsigned n> struct applicator<const char[n]> {
  template<class Opt>
  static void opt(const char *Str, Opt &O) { O.setArgStr(Str); }
};
template<> struct applicator<const char*> {
  template<class Opt>
  static void opt(const char *Str, Opt &O) { O.setArgStr(Str); }
};

template<> struct applicator<NumOccurrencesFlag> {
  static void opt(NumOccurrencesFlag NO, Option &O) {
    O.setNumOccurrencesFlag(NO);
  }
};
template<> struct applicator<ValueExpected> {
  static void opt(ValueExpected VE, Option &O) { O.setValueExpectedFlag(VE); }
};
template<> struct applicator<OptionHidden> {
  static void opt(OptionHidden OH, Option &O) { O.setHiddenFlag(OH); }
};
template<> struct applicator<FormattingFlags> {
  static void opt(FormattingFlags FF, Option &O) { O.setFormattingFlag(FF); }
};
template<> struct applicator<MiscFlags> {
  static void opt(MiscFlags MF, Option &O) { O.setMiscFlag(MF); }
};

// apply method - Apply a modifier to an option in a type safe way.
template<class Mod, class Opt>
void apply(const Mod &M, Opt *O) {
  applicator<Mod>::opt(M, *O);
}

//===----------------------------------------------------------------------===//
// opt_storage class

// Default storage class definition: external storage.  This implementation
// assumes the user will specify a variable to store the data into with the
// cl::location(x) modifier.
//
template<class DataType, bool ExternalStorage, bool isClass>
class opt_storage {
  DataType *Location;   // Where to store the object...
  OptionValue<DataType> Default;

  void check() const {
    assert(Location != 0 && "cl::location(...) not specified for a command "
           "line option with external storage, "
           "or cl::init specified before cl::location()!!");
  }
public:
  opt_storage() : Location(0) {}

  bool setLocation(Option &O, DataType &L) {
    if (Location)
      return O.error("cl::location(x) specified more than once!");
    Location = &L;
    Default = L;
    return false;
  }

  template<class T>
  void setValue(const T &V, bool initial = false) {
    check();
    *Location = V;
    if (initial)
      Default = V;
  }

  DataType &getValue() { check(); return *Location; }
  const DataType &getValue() const { check(); return *Location; }

  operator DataType() const { return this->getValue(); }

  const OptionValue<DataType> &getDefault() const { return Default; }
};

// Define how to hold a class type object, such as a string.  Since we can
// inherit from a class, we do so.  This makes us exactly compatible with the
// object in all cases that it is used.
//
template<class DataType>
class opt_storage<DataType,false,true> : public DataType {
public:
  OptionValue<DataType> Default;

  template<class T>
  void setValue(const T &V, bool initial = false) {
    DataType::operator=(V);
    if (initial)
      Default = V;
  }

  DataType &getValue() { return *this; }
  const DataType &getValue() const { return *this; }

  const OptionValue<DataType> &getDefault() const { return Default; }
};

// Define a partial specialization to handle things we cannot inherit from.  In
// this case, we store an instance through containment, and overload operators
// to get at the value.
//
template<class DataType>
class opt_storage<DataType, false, false> {
public:
  DataType Value;
  OptionValue<DataType> Default;

  // Make sure we initialize the value with the default constructor for the
  // type.
  opt_storage() : Value(DataType()) {}

  template<class T>
  void setValue(const T &V, bool initial = false) {
    Value = V;
    if (initial)
      Default = V;
  }
  DataType &getValue() { return Value; }
  DataType getValue() const { return Value; }

  const OptionValue<DataType> &getDefault() const { return Default; }

  operator DataType() const { return getValue(); }

  // If the datatype is a pointer, support -> on it.
  DataType operator->() const { return Value; }
};


//===----------------------------------------------------------------------===//
// opt - A scalar command line option.
//
template <class DataType, bool ExternalStorage = false,
          class ParserClass = parser<DataType> >
class opt : public Option,
            public opt_storage<DataType, ExternalStorage,
                               is_class<DataType>::value> {
  ParserClass Parser;

  virtual bool handleOccurrence(unsigned pos, StringRef ArgName,
                                StringRef Arg) {
    typename ParserClass::parser_data_type Val =
       typename ParserClass::parser_data_type();
    if (Parser.parse(*this, ArgName, Arg, Val))
      return true;                            // Parse error!
    this->setValue(Val);
    this->setPosition(pos);
    return false;
  }

  virtual enum ValueExpected getValueExpectedFlagDefault() const {
    return Parser.getValueExpectedFlagDefault();
  }
  virtual void getExtraOptionNames(SmallVectorImpl<const char*> &OptionNames) {
    return Parser.getExtraOptionNames(OptionNames);
  }

  // Forward printing stuff to the parser...
  virtual size_t getOptionWidth() const {return Parser.getOptionWidth(*this);}
  virtual void printOptionInfo(size_t GlobalWidth) const {
    Parser.printOptionInfo(*this, GlobalWidth);
  }

  virtual void printOptionValue(size_t GlobalWidth, bool Force) const {
    if (Force || this->getDefault().compare(this->getValue())) {
      cl::printOptionDiff<ParserClass>(
        *this, Parser, this->getValue(), this->getDefault(), GlobalWidth);
    }
  }

  void done() {
    addArgument();
    Parser.initialize(*this);
  }
public:
  // setInitialValue - Used by the cl::init modifier...
  void setInitialValue(const DataType &V) { this->setValue(V, true); }

  ParserClass &getParser() { return Parser; }

  template<class T>
  DataType &operator=(const T &Val) {
    this->setValue(Val);
    return this->getValue();
  }

  // One option...
  template<class M0t>
  explicit opt(const M0t &M0) : Option(Optional, NotHidden) {
    apply(M0, this);
    done();
  }

  // Two options...
  template<class M0t, class M1t>
  opt(const M0t &M0, const M1t &M1) : Option(Optional, NotHidden) {
    apply(M0, this); apply(M1, this);
    done();
  }

  // Three options...
  template<class M0t, class M1t, class M2t>
  opt(const M0t &M0, const M1t &M1,
      const M2t &M2) : Option(Optional, NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this);
    done();
  }
  // Four options...
  template<class M0t, class M1t, class M2t, class M3t>
  opt(const M0t &M0, const M1t &M1, const M2t &M2,
      const M3t &M3) : Option(Optional, NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    done();
  }
  // Five options...
  template<class M0t, class M1t, class M2t, class M3t, class M4t>
  opt(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
      const M4t &M4) : Option(Optional, NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this);
    done();
  }
  // Six options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t>
  opt(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
      const M4t &M4, const M5t &M5) : Option(Optional, NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this);
    done();
  }
  // Seven options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t, class M6t>
  opt(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
      const M4t &M4, const M5t &M5,
      const M6t &M6) : Option(Optional, NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this); apply(M6, this);
    done();
  }
  // Eight options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t, class M6t, class M7t>
  opt(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
      const M4t &M4, const M5t &M5, const M6t &M6,
      const M7t &M7) : Option(Optional, NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this); apply(M6, this); apply(M7, this);
    done();
  }
};

EXTERN_TEMPLATE_INSTANTIATION(class opt<unsigned>);
EXTERN_TEMPLATE_INSTANTIATION(class opt<int>);
EXTERN_TEMPLATE_INSTANTIATION(class opt<std::string>);
EXTERN_TEMPLATE_INSTANTIATION(class opt<char>);
EXTERN_TEMPLATE_INSTANTIATION(class opt<bool>);

//===----------------------------------------------------------------------===//
// list_storage class

// Default storage class definition: external storage.  This implementation
// assumes the user will specify a variable to store the data into with the
// cl::location(x) modifier.
//
template<class DataType, class StorageClass>
class list_storage {
  StorageClass *Location;   // Where to store the object...

public:
  list_storage() : Location(0) {}

  bool setLocation(Option &O, StorageClass &L) {
    if (Location)
      return O.error("cl::location(x) specified more than once!");
    Location = &L;
    return false;
  }

  template<class T>
  void addValue(const T &V) {
    assert(Location != 0 && "cl::location(...) not specified for a command "
           "line option with external storage!");
    Location->push_back(V);
  }
};


// Define how to hold a class type object, such as a string.  Since we can
// inherit from a class, we do so.  This makes us exactly compatible with the
// object in all cases that it is used.
//
template<class DataType>
class list_storage<DataType, bool> : public std::vector<DataType> {
public:
  template<class T>
  void addValue(const T &V) { std::vector<DataType>::push_back(V); }
};


//===----------------------------------------------------------------------===//
// list - A list of command line options.
//
template <class DataType, class Storage = bool,
          class ParserClass = parser<DataType> >
class list : public Option, public list_storage<DataType, Storage> {
  std::vector<unsigned> Positions;
  ParserClass Parser;

  virtual enum ValueExpected getValueExpectedFlagDefault() const {
    return Parser.getValueExpectedFlagDefault();
  }
  virtual void getExtraOptionNames(SmallVectorImpl<const char*> &OptionNames) {
    return Parser.getExtraOptionNames(OptionNames);
  }

  virtual bool handleOccurrence(unsigned pos, StringRef ArgName, StringRef Arg){
    typename ParserClass::parser_data_type Val =
      typename ParserClass::parser_data_type();
    if (Parser.parse(*this, ArgName, Arg, Val))
      return true;  // Parse Error!
    list_storage<DataType, Storage>::addValue(Val);
    setPosition(pos);
    Positions.push_back(pos);
    return false;
  }

  // Forward printing stuff to the parser...
  virtual size_t getOptionWidth() const {return Parser.getOptionWidth(*this);}
  virtual void printOptionInfo(size_t GlobalWidth) const {
    Parser.printOptionInfo(*this, GlobalWidth);
  }

  // Unimplemented: list options don't currently store their default value.
  virtual void printOptionValue(size_t /*GlobalWidth*/, bool /*Force*/) const {}

  void done() {
    addArgument();
    Parser.initialize(*this);
  }
public:
  ParserClass &getParser() { return Parser; }

  unsigned getPosition(unsigned optnum) const {
    assert(optnum < this->size() && "Invalid option index");
    return Positions[optnum];
  }

  void setNumAdditionalVals(unsigned n) {
    Option::setNumAdditionalVals(n);
  }

  // One option...
  template<class M0t>
  explicit list(const M0t &M0) : Option(ZeroOrMore, NotHidden) {
    apply(M0, this);
    done();
  }
  // Two options...
  template<class M0t, class M1t>
  list(const M0t &M0, const M1t &M1) : Option(ZeroOrMore, NotHidden) {
    apply(M0, this); apply(M1, this);
    done();
  }
  // Three options...
  template<class M0t, class M1t, class M2t>
  list(const M0t &M0, const M1t &M1, const M2t &M2)
    : Option(ZeroOrMore, NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this);
    done();
  }
  // Four options...
  template<class M0t, class M1t, class M2t, class M3t>
  list(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3)
    : Option(ZeroOrMore, NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    done();
  }
  // Five options...
  template<class M0t, class M1t, class M2t, class M3t, class M4t>
  list(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
       const M4t &M4) : Option(ZeroOrMore, NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this);
    done();
  }
  // Six options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t>
  list(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
       const M4t &M4, const M5t &M5) : Option(ZeroOrMore, NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this);
    done();
  }
  // Seven options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t, class M6t>
  list(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
       const M4t &M4, const M5t &M5, const M6t &M6)
    : Option(ZeroOrMore, NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this); apply(M6, this);
    done();
  }
  // Eight options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t, class M6t, class M7t>
  list(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
       const M4t &M4, const M5t &M5, const M6t &M6,
       const M7t &M7) : Option(ZeroOrMore, NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this); apply(M6, this); apply(M7, this);
    done();
  }
};

// multi_val - Modifier to set the number of additional values.
struct multi_val {
  unsigned AdditionalVals;
  explicit multi_val(unsigned N) : AdditionalVals(N) {}

  template <typename D, typename S, typename P>
  void apply(list<D, S, P> &L) const { L.setNumAdditionalVals(AdditionalVals); }
};


//===----------------------------------------------------------------------===//
// bits_storage class

// Default storage class definition: external storage.  This implementation
// assumes the user will specify a variable to store the data into with the
// cl::location(x) modifier.
//
template<class DataType, class StorageClass>
class bits_storage {
  unsigned *Location;   // Where to store the bits...

  template<class T>
  static unsigned Bit(const T &V) {
    unsigned BitPos = reinterpret_cast<unsigned>(V);
    assert(BitPos < sizeof(unsigned) * CHAR_BIT &&
          "enum exceeds width of bit vector!");
    return 1 << BitPos;
  }

public:
  bits_storage() : Location(0) {}

  bool setLocation(Option &O, unsigned &L) {
    if (Location)
      return O.error("cl::location(x) specified more than once!");
    Location = &L;
    return false;
  }

  template<class T>
  void addValue(const T &V) {
    assert(Location != 0 && "cl::location(...) not specified for a command "
           "line option with external storage!");
    *Location |= Bit(V);
  }

  unsigned getBits() { return *Location; }

  template<class T>
  bool isSet(const T &V) {
    return (*Location & Bit(V)) != 0;
  }
};


// Define how to hold bits.  Since we can inherit from a class, we do so.
// This makes us exactly compatible with the bits in all cases that it is used.
//
template<class DataType>
class bits_storage<DataType, bool> {
  unsigned Bits;   // Where to store the bits...

  template<class T>
  static unsigned Bit(const T &V) {
    unsigned BitPos = (unsigned)V;
    assert(BitPos < sizeof(unsigned) * CHAR_BIT &&
          "enum exceeds width of bit vector!");
    return 1 << BitPos;
  }

public:
  template<class T>
  void addValue(const T &V) {
    Bits |=  Bit(V);
  }

  unsigned getBits() { return Bits; }

  template<class T>
  bool isSet(const T &V) {
    return (Bits & Bit(V)) != 0;
  }
};


//===----------------------------------------------------------------------===//
// bits - A bit vector of command options.
//
template <class DataType, class Storage = bool,
          class ParserClass = parser<DataType> >
class bits : public Option, public bits_storage<DataType, Storage> {
  std::vector<unsigned> Positions;
  ParserClass Parser;

  virtual enum ValueExpected getValueExpectedFlagDefault() const {
    return Parser.getValueExpectedFlagDefault();
  }
  virtual void getExtraOptionNames(SmallVectorImpl<const char*> &OptionNames) {
    return Parser.getExtraOptionNames(OptionNames);
  }

  virtual bool handleOccurrence(unsigned pos, StringRef ArgName, StringRef Arg){
    typename ParserClass::parser_data_type Val =
      typename ParserClass::parser_data_type();
    if (Parser.parse(*this, ArgName, Arg, Val))
      return true;  // Parse Error!
    this->addValue(Val);
    setPosition(pos);
    Positions.push_back(pos);
    return false;
  }

  // Forward printing stuff to the parser...
  virtual size_t getOptionWidth() const {return Parser.getOptionWidth(*this);}
  virtual void printOptionInfo(size_t GlobalWidth) const {
    Parser.printOptionInfo(*this, GlobalWidth);
  }

  // Unimplemented: bits options don't currently store their default values.
  virtual void printOptionValue(size_t /*GlobalWidth*/, bool /*Force*/) const {}

  void done() {
    addArgument();
    Parser.initialize(*this);
  }
public:
  ParserClass &getParser() { return Parser; }

  unsigned getPosition(unsigned optnum) const {
    assert(optnum < this->size() && "Invalid option index");
    return Positions[optnum];
  }

  // One option...
  template<class M0t>
  explicit bits(const M0t &M0) : Option(ZeroOrMore, NotHidden) {
    apply(M0, this);
    done();
  }
  // Two options...
  template<class M0t, class M1t>
  bits(const M0t &M0, const M1t &M1) : Option(ZeroOrMore, NotHidden) {
    apply(M0, this); apply(M1, this);
    done();
  }
  // Three options...
  template<class M0t, class M1t, class M2t>
  bits(const M0t &M0, const M1t &M1, const M2t &M2)
    : Option(ZeroOrMore, NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this);
    done();
  }
  // Four options...
  template<class M0t, class M1t, class M2t, class M3t>
  bits(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3)
    : Option(ZeroOrMore, NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    done();
  }
  // Five options...
  template<class M0t, class M1t, class M2t, class M3t, class M4t>
  bits(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
       const M4t &M4) : Option(ZeroOrMore, NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this);
    done();
  }
  // Six options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t>
  bits(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
       const M4t &M4, const M5t &M5) : Option(ZeroOrMore, NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this);
    done();
  }
  // Seven options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t, class M6t>
  bits(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
       const M4t &M4, const M5t &M5, const M6t &M6)
    : Option(ZeroOrMore, NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this); apply(M6, this);
    done();
  }
  // Eight options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t, class M6t, class M7t>
  bits(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
       const M4t &M4, const M5t &M5, const M6t &M6,
       const M7t &M7) : Option(ZeroOrMore, NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this); apply(M6, this); apply(M7, this);
    done();
  }
};

//===----------------------------------------------------------------------===//
// Aliased command line option (alias this name to a preexisting name)
//

class alias : public Option {
  Option *AliasFor;
  virtual bool handleOccurrence(unsigned pos, StringRef /*ArgName*/,
                                StringRef Arg) LLVM_OVERRIDE {
    return AliasFor->handleOccurrence(pos, AliasFor->ArgStr, Arg);
  }
  // Handle printing stuff...
  virtual size_t getOptionWidth() const LLVM_OVERRIDE;
  virtual void printOptionInfo(size_t GlobalWidth) const LLVM_OVERRIDE;

  // Aliases do not need to print their values.
  virtual void printOptionValue(size_t /*GlobalWidth*/,
                                bool /*Force*/) const LLVM_OVERRIDE {}

  void done() {
    if (!hasArgStr())
      error("cl::alias must have argument name specified!");
    if (AliasFor == 0)
      error("cl::alias must have an cl::aliasopt(option) specified!");
      addArgument();
  }
public:
  void setAliasFor(Option &O) {
    if (AliasFor)
      error("cl::alias must only have one cl::aliasopt(...) specified!");
    AliasFor = &O;
  }

  // One option...
  template<class M0t>
  explicit alias(const M0t &M0) : Option(Optional, Hidden), AliasFor(0) {
    apply(M0, this);
    done();
  }
  // Two options...
  template<class M0t, class M1t>
  alias(const M0t &M0, const M1t &M1) : Option(Optional, Hidden), AliasFor(0) {
    apply(M0, this); apply(M1, this);
    done();
  }
  // Three options...
  template<class M0t, class M1t, class M2t>
  alias(const M0t &M0, const M1t &M1, const M2t &M2)
    : Option(Optional, Hidden), AliasFor(0) {
    apply(M0, this); apply(M1, this); apply(M2, this);
    done();
  }
  // Four options...
  template<class M0t, class M1t, class M2t, class M3t>
  alias(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3)
    : Option(Optional, Hidden), AliasFor(0) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    done();
  }
};

// aliasfor - Modifier to set the option an alias aliases.
struct aliasopt {
  Option &Opt;
  explicit aliasopt(Option &O) : Opt(O) {}
  void apply(alias &A) const { A.setAliasFor(Opt); }
};

// extrahelp - provide additional help at the end of the normal help
// output. All occurrences of cl::extrahelp will be accumulated and
// printed to stderr at the end of the regular help, just before
// exit is called.
struct extrahelp {
  const char * morehelp;
  explicit extrahelp(const char* help);
};

void PrintVersionMessage();
// This function just prints the help message, exactly the same way as if the
// -help option had been given on the command line.
// NOTE: THIS FUNCTION TERMINATES THE PROGRAM!
void PrintHelpMessage();

} // End namespace cl

} // End namespace llvm

#endif
