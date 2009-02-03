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
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <cstdarg>
#include <string>
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
void ParseCommandLineOptions(int argc, char **argv,
                             const char *Overview = 0,
                             bool ReadResponseFiles = false);

//===----------------------------------------------------------------------===//
// ParseEnvironmentOptions - Environment variable option processing alternate
//                           entry point.
//
void ParseEnvironmentOptions(const char *progName, const char *envvar,
                             const char *Overview = 0,
                             bool ReadResponseFiles = false);

///===---------------------------------------------------------------------===//
/// SetVersionPrinter - Override the default (LLVM specific) version printer
///                     used to print out the version when --version is given
///                     on the command line. This allows other systems using the
///                     CommandLine utilities to print their own version string.
void SetVersionPrinter(void (*func)());


// MarkOptionsChanged - Internal helper function.
void MarkOptionsChanged();

//===----------------------------------------------------------------------===//
// Flags permitted to be passed to command line arguments
//

enum NumOccurrences {          // Flags for the number of occurrences allowed
  Optional        = 0x01,      // Zero or One occurrence
  ZeroOrMore      = 0x02,      // Zero or more occurrences allowed
  Required        = 0x03,      // One occurrence required
  OneOrMore       = 0x04,      // One or more occurrences required

  // ConsumeAfter - Indicates that this option is fed anything that follows the
  // last positional argument required by the application (it is an error if
  // there are zero positional arguments, and a ConsumeAfter option is used).
  // Thus, for example, all arguments to LLI are processed until a filename is
  // found.  Once a filename is found, all of the succeeding arguments are
  // passed, unprocessed, to the ConsumeAfter option.
  //
  ConsumeAfter    = 0x05,

  OccurrencesMask  = 0x07
};

enum ValueExpected {           // Is a value required for the option?
  ValueOptional   = 0x08,      // The value can appear... or not
  ValueRequired   = 0x10,      // The value is required to appear!
  ValueDisallowed = 0x18,      // A value may not be specified (for flags)
  ValueMask       = 0x18
};

enum OptionHidden {            // Control whether -help shows this option
  NotHidden       = 0x20,      // Option included in --help & --help-hidden
  Hidden          = 0x40,      // -help doesn't, but --help-hidden does
  ReallyHidden    = 0x60,      // Neither --help nor --help-hidden show this arg
  HiddenMask      = 0x60
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
  NormalFormatting = 0x000,     // Nothing special
  Positional       = 0x080,     // Is a positional argument, no '-' required
  Prefix           = 0x100,     // Can this option directly prefix its value?
  Grouping         = 0x180,     // Can this option group with other options?
  FormattingMask   = 0x180      // Union of the above flags.
};

enum MiscFlags {               // Miscellaneous flags to adjust argument
  CommaSeparated     = 0x200,  // Should this cl::list split between commas?
  PositionalEatsArgs = 0x400,  // Should this positional cl::list eat -args?
  Sink               = 0x800,  // Should this cl::list eat all unknown options?
  AllowInverse	     = 0x1000, // Can this option take a -Xno- form?
  MiscMask           = 0x1E00  // Union of the above flags.
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
  virtual bool handleOccurrence(unsigned pos, const char *ArgName,
                                const std::string &Arg) = 0;

  virtual enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueOptional;
  }

  // Out of line virtual function to provide home for the class.
  virtual void anchor();

  int NumOccurrences;     // The number of times specified
  int Flags;              // Flags for the argument
  unsigned Position;      // Position of last occurrence of the option
  unsigned AdditionalVals;// Greater than 0 for multi-valued option.
  Option *NextRegistered; // Singly linked list of registered options.
public:
  const char *ArgStr;     // The argument string itself (ex: "help", "o")
  const char *HelpStr;    // The descriptive text message for --help
  const char *ValueStr;   // String describing what the value of this option is

  inline enum NumOccurrences getNumOccurrencesFlag() const {
    return static_cast<enum NumOccurrences>(Flags & OccurrencesMask);
  }
  inline enum ValueExpected getValueExpectedFlag() const {
    int VE = Flags & ValueMask;
    return VE ? static_cast<enum ValueExpected>(VE)
              : getValueExpectedFlagDefault();
  }
  inline enum OptionHidden getOptionHiddenFlag() const {
    return static_cast<enum OptionHidden>(Flags & HiddenMask);
  }
  inline enum FormattingFlags getFormattingFlag() const {
    return static_cast<enum FormattingFlags>(Flags & FormattingMask);
  }
  inline unsigned getMiscFlags() const {
    return Flags & MiscMask;
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

  void setFlag(unsigned Flag, unsigned FlagMask) {
    Flags &= ~FlagMask;
    Flags |= Flag;
  }

  void setNumOccurrencesFlag(enum NumOccurrences Val) {
    setFlag(Val, OccurrencesMask);
  }
  void setValueExpectedFlag(enum ValueExpected Val) { setFlag(Val, ValueMask); }
  void setHiddenFlag(enum OptionHidden Val) { setFlag(Val, HiddenMask); }
  void setFormattingFlag(enum FormattingFlags V) { setFlag(V, FormattingMask); }
  void setMiscFlag(enum MiscFlags M) { setFlag(M, M); }
  void setPosition(unsigned pos) { Position = pos; }
protected:
  explicit Option(unsigned DefaultFlags)
    : NumOccurrences(0), Flags(DefaultFlags | NormalFormatting), Position(0),
      AdditionalVals(0), NextRegistered(0),
      ArgStr(""), HelpStr(""), ValueStr("") {
    assert(getNumOccurrencesFlag() != 0 &&
           getOptionHiddenFlag() != 0 && "Not all default flags specified!");
  }

  inline void setNumAdditionalVals(unsigned n)
  { AdditionalVals = n; }
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

  virtual void getExtraOptionNames(std::vector<const char*> &) {}

  // addOccurrence - Wrapper around handleOccurrence that enforces Flags
  //
  bool addOccurrence(unsigned pos, const char *ArgName,
                     const std::string &Value, bool MultiArg = false);

  // Prints option name followed by message.  Always returns true.
  bool error(std::string Message, const char *ArgName = 0);

public:
  inline int getNumOccurrences() const { return NumOccurrences; }
  virtual ~Option() {}
};


//===----------------------------------------------------------------------===//
// Command line option modifiers that can be used to modify the behavior of
// command line option parsers...
//

// desc - Modifier to set the description shown in the --help output...
struct desc {
  const char *Desc;
  desc(const char *Str) : Desc(Str) {}
  void apply(Option &O) const { O.setDescription(Desc); }
};

// value_desc - Modifier to set the value description shown in the --help
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
struct generic_parser_base {
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

  // printOptionInfo - Print out information about this option.  The
  // to-be-maintained width is specified.
  //
  virtual void printOptionInfo(const Option &O, size_t GlobalWidth) const;

  void initialize(Option &O) {
    // All of the modifiers for the option have been processed by now, so the
    // argstr field should be stable, copy it down now.
    //
    hasArgStr = O.hasArgStr();
  }

  void getExtraOptionNames(std::vector<const char*> &OptionNames) {
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
// command line option for --help.  Because this is a simple mapping parser, the
// data type can be any unsupported type.
//
template <class DataType>
class parser : public generic_parser_base {
protected:
  SmallVector<std::pair<const char *,
                        std::pair<DataType, const char *> >, 8> Values;
public:
  typedef DataType parser_data_type;

  // Implement virtual functions needed by generic_parser_base
  unsigned getNumOptions() const { return unsigned(Values.size()); }
  const char *getOption(unsigned N) const { return Values[N].first; }
  const char *getDescription(unsigned N) const {
    return Values[N].second.second;
  }

  // parse - Return true on error.
  bool parse(Option &O, const char *ArgName, const std::string &Arg,
             DataType &V) {
    std::string ArgVal;
    if (hasArgStr)
      ArgVal = Arg;
    else
      ArgVal = ArgName;

    for (unsigned i = 0, e = static_cast<unsigned>(Values.size());
         i != e; ++i)
      if (ArgVal == Values[i].first) {
        V = Values[i].second.first;
        return false;
      }

    return O.error(": Cannot find option named '" + ArgVal + "'!");
  }

  /// addLiteralOption - Add an entry to the mapping table.
  ///
  template <class DT>
  void addLiteralOption(const char *Name, const DT &V, const char *HelpStr) {
    assert(findOption(Name) == Values.size() && "Option already exists!");
    Values.push_back(std::make_pair(Name,
                             std::make_pair(static_cast<DataType>(V),HelpStr)));
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
struct basic_parser_impl {  // non-template implementation of basic_parser<t>
  virtual ~basic_parser_impl() {}

  enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueRequired;
  }

  void getExtraOptionNames(std::vector<const char*> &) {}

  void initialize(Option &) {}

  // Return the width of the option tag for printing...
  size_t getOptionWidth(const Option &O) const;

  // printOptionInfo - Print out information about this option.  The
  // to-be-maintained width is specified.
  //
  void printOptionInfo(const Option &O, size_t GlobalWidth) const;

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "value"; }

  // An out-of-line virtual method to provide a 'home' for this class.
  virtual void anchor();
};

// basic_parser - The real basic parser is just a template wrapper that provides
// a typedef for the provided data type.
//
template<class DataType>
struct basic_parser : public basic_parser_impl {
  typedef DataType parser_data_type;
};

//--------------------------------------------------
// parser<bool>
//
template<>
class parser<bool> : public basic_parser<bool> {
  bool IsInvertable;	// Should we synthezise a -xno- style option?
  const char *ArgStr;
public:
  void getExtraOptionNames(std::vector<const char*> &OptionNames) {
    if (IsInvertable) {
      char *s = new char [strlen(ArgStr) + 3 + 1];
      s[0] = ArgStr[0];
      s[1] = 'n';
      s[2] = 'o';
      s[3] = '-';
      strcpy(&s[4], ArgStr+1);
      OptionNames.push_back(s);
    }
  }

  // parse - Return true on error.
  bool parse(Option &O, const char *ArgName, const std::string &Arg, bool &Val);

  template <class Opt>
  void initialize(Opt &O) {
    if (O.getMiscFlags() & llvm::cl::AllowInverse)
      IsInvertable = true;
    else
      IsInvertable = false;
    ArgStr = O.ArgStr;
  }

  enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueOptional;
  }

  // getValueName - Do not print =<value> at all.
  virtual const char *getValueName() const { return 0; }

  // An out-of-line virtual method to provide a 'home' for this class.
  virtual void anchor();
};

EXTERN_TEMPLATE_INSTANTIATION(class basic_parser<bool>);

//--------------------------------------------------
// parser<boolOrDefault>
enum boolOrDefault { BOU_UNSET, BOU_TRUE, BOU_FALSE };
template<>
class parser<boolOrDefault> : public basic_parser<boolOrDefault> {
public:
  // parse - Return true on error.
  bool parse(Option &O, const char *ArgName, const std::string &Arg,
             boolOrDefault &Val);

  enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueOptional;
  }

  // getValueName - Do not print =<value> at all.
  virtual const char *getValueName() const { return 0; }

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
  bool parse(Option &O, const char *ArgName, const std::string &Arg, int &Val);

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "int"; }

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
  bool parse(Option &O, const char *AN, const std::string &Arg, unsigned &Val);

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "uint"; }

  // An out-of-line virtual method to provide a 'home' for this class.
  virtual void anchor();
};

EXTERN_TEMPLATE_INSTANTIATION(class basic_parser<unsigned>);

//--------------------------------------------------
// parser<double>
//
template<>
class parser<double> : public basic_parser<double> {
public:
  // parse - Return true on error.
  bool parse(Option &O, const char *AN, const std::string &Arg, double &Val);

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "number"; }

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
  bool parse(Option &O, const char *AN, const std::string &Arg, float &Val);

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "number"; }

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
  bool parse(Option &, const char *, const std::string &Arg,
             std::string &Value) {
    Value = Arg;
    return false;
  }

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "string"; }

  // An out-of-line virtual method to provide a 'home' for this class.
  virtual void anchor();
};

EXTERN_TEMPLATE_INSTANTIATION(class basic_parser<std::string>);

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

template<> struct applicator<NumOccurrences> {
  static void opt(NumOccurrences NO, Option &O) { O.setNumOccurrencesFlag(NO); }
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

  void check() {
    assert(Location != 0 && "cl::location(...) not specified for a command "
           "line option with external storage, "
           "or cl::init specified before cl::location()!!");
  }
public:
  opt_storage() : Location(0) {}

  bool setLocation(Option &O, DataType &L) {
    if (Location)
      return O.error(": cl::location(x) specified more than once!");
    Location = &L;
    return false;
  }

  template<class T>
  void setValue(const T &V) {
    check();
    *Location = V;
  }

  DataType &getValue() { check(); return *Location; }
  const DataType &getValue() const { check(); return *Location; }
};


// Define how to hold a class type object, such as a string.  Since we can
// inherit from a class, we do so.  This makes us exactly compatible with the
// object in all cases that it is used.
//
template<class DataType>
class opt_storage<DataType,false,true> : public DataType {
public:
  template<class T>
  void setValue(const T &V) { DataType::operator=(V); }

  DataType &getValue() { return *this; }
  const DataType &getValue() const { return *this; }
};

// Define a partial specialization to handle things we cannot inherit from.  In
// this case, we store an instance through containment, and overload operators
// to get at the value.
//
template<class DataType>
class opt_storage<DataType, false, false> {
public:
  DataType Value;

  // Make sure we initialize the value with the default constructor for the
  // type.
  opt_storage() : Value(DataType()) {}

  template<class T>
  void setValue(const T &V) { Value = V; }
  DataType &getValue() { return Value; }
  DataType getValue() const { return Value; }

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

  virtual bool handleOccurrence(unsigned pos, const char *ArgName,
                                const std::string &Arg) {
    typename ParserClass::parser_data_type Val =
       typename ParserClass::parser_data_type();
    if (Parser.parse(*this, ArgName, Arg, Val))
      return true;                            // Parse error!
    setValue(Val);
    setPosition(pos);
    return false;
  }

  virtual enum ValueExpected getValueExpectedFlagDefault() const {
    return Parser.getValueExpectedFlagDefault();
  }
  virtual void getExtraOptionNames(std::vector<const char*> &OptionNames) {
    return Parser.getExtraOptionNames(OptionNames);
  }

  // Forward printing stuff to the parser...
  virtual size_t getOptionWidth() const {return Parser.getOptionWidth(*this);}
  virtual void printOptionInfo(size_t GlobalWidth) const {
    Parser.printOptionInfo(*this, GlobalWidth);
  }

  void done() {
    addArgument();
    Parser.initialize(*this);
  }
public:
  // setInitialValue - Used by the cl::init modifier...
  void setInitialValue(const DataType &V) { this->setValue(V); }

  ParserClass &getParser() { return Parser; }

  operator DataType() const { return this->getValue(); }

  template<class T>
  DataType &operator=(const T &Val) {
    this->setValue(Val);
    return this->getValue();
  }

  // One option...
  template<class M0t>
  explicit opt(const M0t &M0) : Option(Optional | NotHidden) {
    apply(M0, this);
    done();
  }

  // Two options...
  template<class M0t, class M1t>
  opt(const M0t &M0, const M1t &M1) : Option(Optional | NotHidden) {
    apply(M0, this); apply(M1, this);
    done();
  }

  // Three options...
  template<class M0t, class M1t, class M2t>
  opt(const M0t &M0, const M1t &M1,
      const M2t &M2) : Option(Optional | NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this);
    done();
  }
  // Four options...
  template<class M0t, class M1t, class M2t, class M3t>
  opt(const M0t &M0, const M1t &M1, const M2t &M2,
      const M3t &M3) : Option(Optional | NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    done();
  }
  // Five options...
  template<class M0t, class M1t, class M2t, class M3t, class M4t>
  opt(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
      const M4t &M4) : Option(Optional | NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this);
    done();
  }
  // Six options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t>
  opt(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
      const M4t &M4, const M5t &M5) : Option(Optional | NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this);
    done();
  }
  // Seven options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t, class M6t>
  opt(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
      const M4t &M4, const M5t &M5,
      const M6t &M6) : Option(Optional | NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this); apply(M6, this);
    done();
  }
  // Eight options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t, class M6t, class M7t>
  opt(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
      const M4t &M4, const M5t &M5, const M6t &M6,
      const M7t &M7) : Option(Optional | NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this); apply(M6, this); apply(M7, this);
    done();
  }
};

EXTERN_TEMPLATE_INSTANTIATION(class opt<unsigned>);
EXTERN_TEMPLATE_INSTANTIATION(class opt<int>);
EXTERN_TEMPLATE_INSTANTIATION(class opt<std::string>);
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
      return O.error(": cl::location(x) specified more than once!");
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
  void addValue(const T &V) { push_back(V); }
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
  virtual void getExtraOptionNames(std::vector<const char*> &OptionNames) {
    return Parser.getExtraOptionNames(OptionNames);
  }

  virtual bool handleOccurrence(unsigned pos, const char *ArgName,
                                const std::string &Arg) {
    typename ParserClass::parser_data_type Val =
      typename ParserClass::parser_data_type();
    if (Parser.parse(*this, ArgName, Arg, Val))
      return true;  // Parse Error!
    addValue(Val);
    setPosition(pos);
    Positions.push_back(pos);
    return false;
  }

  // Forward printing stuff to the parser...
  virtual size_t getOptionWidth() const {return Parser.getOptionWidth(*this);}
  virtual void printOptionInfo(size_t GlobalWidth) const {
    Parser.printOptionInfo(*this, GlobalWidth);
  }

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
  explicit list(const M0t &M0) : Option(ZeroOrMore | NotHidden) {
    apply(M0, this);
    done();
  }
  // Two options...
  template<class M0t, class M1t>
  list(const M0t &M0, const M1t &M1) : Option(ZeroOrMore | NotHidden) {
    apply(M0, this); apply(M1, this);
    done();
  }
  // Three options...
  template<class M0t, class M1t, class M2t>
  list(const M0t &M0, const M1t &M1, const M2t &M2)
    : Option(ZeroOrMore | NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this);
    done();
  }
  // Four options...
  template<class M0t, class M1t, class M2t, class M3t>
  list(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3)
    : Option(ZeroOrMore | NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    done();
  }
  // Five options...
  template<class M0t, class M1t, class M2t, class M3t, class M4t>
  list(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
       const M4t &M4) : Option(ZeroOrMore | NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this);
    done();
  }
  // Six options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t>
  list(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
       const M4t &M4, const M5t &M5) : Option(ZeroOrMore | NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this);
    done();
  }
  // Seven options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t, class M6t>
  list(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
       const M4t &M4, const M5t &M5, const M6t &M6)
    : Option(ZeroOrMore | NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this); apply(M6, this);
    done();
  }
  // Eight options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t, class M6t, class M7t>
  list(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
       const M4t &M4, const M5t &M5, const M6t &M6,
       const M7t &M7) : Option(ZeroOrMore | NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this); apply(M6, this); apply(M7, this);
    done();
  }
};

// multi_arg - Modifier to set the number of additional values.
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
    assert(BitPos < sizeof(unsigned) * 8 &&
          "enum exceeds width of bit vector!");
    return 1 << BitPos;
  }

public:
  bits_storage() : Location(0) {}

  bool setLocation(Option &O, unsigned &L) {
    if (Location)
      return O.error(": cl::location(x) specified more than once!");
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
    unsigned BitPos = reinterpret_cast<unsigned>(V);
    assert(BitPos < sizeof(unsigned) * 8 &&
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
  virtual void getExtraOptionNames(std::vector<const char*> &OptionNames) {
    return Parser.getExtraOptionNames(OptionNames);
  }

  virtual bool handleOccurrence(unsigned pos, const char *ArgName,
                                const std::string &Arg) {
    typename ParserClass::parser_data_type Val =
      typename ParserClass::parser_data_type();
    if (Parser.parse(*this, ArgName, Arg, Val))
      return true;  // Parse Error!
    addValue(Val);
    setPosition(pos);
    Positions.push_back(pos);
    return false;
  }

  // Forward printing stuff to the parser...
  virtual size_t getOptionWidth() const {return Parser.getOptionWidth(*this);}
  virtual void printOptionInfo(size_t GlobalWidth) const {
    Parser.printOptionInfo(*this, GlobalWidth);
  }

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
  explicit bits(const M0t &M0) : Option(ZeroOrMore | NotHidden) {
    apply(M0, this);
    done();
  }
  // Two options...
  template<class M0t, class M1t>
  bits(const M0t &M0, const M1t &M1) : Option(ZeroOrMore | NotHidden) {
    apply(M0, this); apply(M1, this);
    done();
  }
  // Three options...
  template<class M0t, class M1t, class M2t>
  bits(const M0t &M0, const M1t &M1, const M2t &M2)
    : Option(ZeroOrMore | NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this);
    done();
  }
  // Four options...
  template<class M0t, class M1t, class M2t, class M3t>
  bits(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3)
    : Option(ZeroOrMore | NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    done();
  }
  // Five options...
  template<class M0t, class M1t, class M2t, class M3t, class M4t>
  bits(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
       const M4t &M4) : Option(ZeroOrMore | NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this);
    done();
  }
  // Six options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t>
  bits(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
       const M4t &M4, const M5t &M5) : Option(ZeroOrMore | NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this);
    done();
  }
  // Seven options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t, class M6t>
  bits(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
       const M4t &M4, const M5t &M5, const M6t &M6)
    : Option(ZeroOrMore | NotHidden) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this); apply(M6, this);
    done();
  }
  // Eight options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t, class M6t, class M7t>
  bits(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
       const M4t &M4, const M5t &M5, const M6t &M6,
       const M7t &M7) : Option(ZeroOrMore | NotHidden) {
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
  virtual bool handleOccurrence(unsigned pos, const char * /*ArgName*/,
                                const std::string &Arg) {
    return AliasFor->handleOccurrence(pos, AliasFor->ArgStr, Arg);
  }
  // Handle printing stuff...
  virtual size_t getOptionWidth() const;
  virtual void printOptionInfo(size_t GlobalWidth) const;

  void done() {
    if (!hasArgStr())
      error(": cl::alias must have argument name specified!");
    if (AliasFor == 0)
      error(": cl::alias must have an cl::aliasopt(option) specified!");
      addArgument();
  }
public:
  void setAliasFor(Option &O) {
    if (AliasFor)
      error(": cl::alias must only have one cl::aliasopt(...) specified!");
    AliasFor = &O;
  }

  // One option...
  template<class M0t>
  explicit alias(const M0t &M0) : Option(Optional | Hidden), AliasFor(0) {
    apply(M0, this);
    done();
  }
  // Two options...
  template<class M0t, class M1t>
  alias(const M0t &M0, const M1t &M1) : Option(Optional | Hidden), AliasFor(0) {
    apply(M0, this); apply(M1, this);
    done();
  }
  // Three options...
  template<class M0t, class M1t, class M2t>
  alias(const M0t &M0, const M1t &M1, const M2t &M2)
    : Option(Optional | Hidden), AliasFor(0) {
    apply(M0, this); apply(M1, this); apply(M2, this);
    done();
  }
  // Four options...
  template<class M0t, class M1t, class M2t, class M3t>
  alias(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3)
    : Option(Optional | Hidden), AliasFor(0) {
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
// printed to std::cerr at the end of the regular help, just before
// exit is called.
struct extrahelp {
  const char * morehelp;
  explicit extrahelp(const char* help);
};

void PrintVersionMessage();
// This function just prints the help message, exactly the same way as if the
// --help option had been given on the command line.
// NOTE: THIS FUNCTION TERMINATES THE PROGRAM!
void PrintHelpMessage();

} // End namespace cl

} // End namespace llvm

#endif
