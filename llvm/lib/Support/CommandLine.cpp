//===-- CommandLine.cpp - Command line parser implementation --------------===//
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
// Note that rather than trying to figure out what this code does, you could try
// reading the library documentation located in docs/CommandLine.html
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/config.h"
#include <cerrno>
#include <cstdlib>
using namespace llvm;
using namespace cl;

//===----------------------------------------------------------------------===//
// Template instantiations and anchors.
//
namespace llvm { namespace cl {
TEMPLATE_INSTANTIATION(class basic_parser<bool>);
TEMPLATE_INSTANTIATION(class basic_parser<boolOrDefault>);
TEMPLATE_INSTANTIATION(class basic_parser<int>);
TEMPLATE_INSTANTIATION(class basic_parser<unsigned>);
TEMPLATE_INSTANTIATION(class basic_parser<unsigned long long>);
TEMPLATE_INSTANTIATION(class basic_parser<double>);
TEMPLATE_INSTANTIATION(class basic_parser<float>);
TEMPLATE_INSTANTIATION(class basic_parser<std::string>);
TEMPLATE_INSTANTIATION(class basic_parser<char>);

TEMPLATE_INSTANTIATION(class opt<unsigned>);
TEMPLATE_INSTANTIATION(class opt<int>);
TEMPLATE_INSTANTIATION(class opt<std::string>);
TEMPLATE_INSTANTIATION(class opt<char>);
TEMPLATE_INSTANTIATION(class opt<bool>);
} } // end namespace llvm::cl

void GenericOptionValue::anchor() {}
void OptionValue<boolOrDefault>::anchor() {}
void OptionValue<std::string>::anchor() {}
void Option::anchor() {}
void basic_parser_impl::anchor() {}
void parser<bool>::anchor() {}
void parser<boolOrDefault>::anchor() {}
void parser<int>::anchor() {}
void parser<unsigned>::anchor() {}
void parser<unsigned long long>::anchor() {}
void parser<double>::anchor() {}
void parser<float>::anchor() {}
void parser<std::string>::anchor() {}
void parser<char>::anchor() {}

//===----------------------------------------------------------------------===//

// Globals for name and overview of program.  Program name is not a string to
// avoid static ctor/dtor issues.
static char ProgramName[80] = "<premain>";
static const char *ProgramOverview = 0;

// This collects additional help to be printed.
static ManagedStatic<std::vector<const char*> > MoreHelp;

extrahelp::extrahelp(const char *Help)
  : morehelp(Help) {
  MoreHelp->push_back(Help);
}

static bool OptionListChanged = false;

// MarkOptionsChanged - Internal helper function.
void cl::MarkOptionsChanged() {
  OptionListChanged = true;
}

/// RegisteredOptionList - This is the list of the command line options that
/// have statically constructed themselves.
static Option *RegisteredOptionList = 0;

void Option::addArgument() {
  assert(NextRegistered == 0 && "argument multiply registered!");

  NextRegistered = RegisteredOptionList;
  RegisteredOptionList = this;
  MarkOptionsChanged();
}


//===----------------------------------------------------------------------===//
// Basic, shared command line option processing machinery.
//

/// GetOptionInfo - Scan the list of registered options, turning them into data
/// structures that are easier to handle.
static void GetOptionInfo(SmallVectorImpl<Option*> &PositionalOpts,
                          SmallVectorImpl<Option*> &SinkOpts,
                          StringMap<Option*> &OptionsMap) {
  SmallVector<const char*, 16> OptionNames;
  Option *CAOpt = 0;  // The ConsumeAfter option if it exists.
  for (Option *O = RegisteredOptionList; O; O = O->getNextRegisteredOption()) {
    // If this option wants to handle multiple option names, get the full set.
    // This handles enum options like "-O1 -O2" etc.
    O->getExtraOptionNames(OptionNames);
    if (O->ArgStr[0])
      OptionNames.push_back(O->ArgStr);

    // Handle named options.
    for (size_t i = 0, e = OptionNames.size(); i != e; ++i) {
      // Add argument to the argument map!
      if (OptionsMap.GetOrCreateValue(OptionNames[i], O).second != O) {
        errs() << ProgramName << ": CommandLine Error: Argument '"
             << OptionNames[i] << "' defined more than once!\n";
      }
    }

    OptionNames.clear();

    // Remember information about positional options.
    if (O->getFormattingFlag() == cl::Positional)
      PositionalOpts.push_back(O);
    else if (O->getMiscFlags() & cl::Sink) // Remember sink options
      SinkOpts.push_back(O);
    else if (O->getNumOccurrencesFlag() == cl::ConsumeAfter) {
      if (CAOpt)
        O->error("Cannot specify more than one option with cl::ConsumeAfter!");
      CAOpt = O;
    }
  }

  if (CAOpt)
    PositionalOpts.push_back(CAOpt);

  // Make sure that they are in order of registration not backwards.
  std::reverse(PositionalOpts.begin(), PositionalOpts.end());
}


/// LookupOption - Lookup the option specified by the specified option on the
/// command line.  If there is a value specified (after an equal sign) return
/// that as well.  This assumes that leading dashes have already been stripped.
static Option *LookupOption(StringRef &Arg, StringRef &Value,
                            const StringMap<Option*> &OptionsMap) {
  // Reject all dashes.
  if (Arg.empty()) return 0;

  size_t EqualPos = Arg.find('=');

  // If we have an equals sign, remember the value.
  if (EqualPos == StringRef::npos) {
    // Look up the option.
    StringMap<Option*>::const_iterator I = OptionsMap.find(Arg);
    return I != OptionsMap.end() ? I->second : 0;
  }

  // If the argument before the = is a valid option name, we match.  If not,
  // return Arg unmolested.
  StringMap<Option*>::const_iterator I =
    OptionsMap.find(Arg.substr(0, EqualPos));
  if (I == OptionsMap.end()) return 0;

  Value = Arg.substr(EqualPos+1);
  Arg = Arg.substr(0, EqualPos);
  return I->second;
}

/// LookupNearestOption - Lookup the closest match to the option specified by
/// the specified option on the command line.  If there is a value specified
/// (after an equal sign) return that as well.  This assumes that leading dashes
/// have already been stripped.
static Option *LookupNearestOption(StringRef Arg,
                                   const StringMap<Option*> &OptionsMap,
                                   std::string &NearestString) {
  // Reject all dashes.
  if (Arg.empty()) return 0;

  // Split on any equal sign.
  std::pair<StringRef, StringRef> SplitArg = Arg.split('=');
  StringRef &LHS = SplitArg.first;  // LHS == Arg when no '=' is present.
  StringRef &RHS = SplitArg.second;

  // Find the closest match.
  Option *Best = 0;
  unsigned BestDistance = 0;
  for (StringMap<Option*>::const_iterator it = OptionsMap.begin(),
         ie = OptionsMap.end(); it != ie; ++it) {
    Option *O = it->second;
    SmallVector<const char*, 16> OptionNames;
    O->getExtraOptionNames(OptionNames);
    if (O->ArgStr[0])
      OptionNames.push_back(O->ArgStr);

    bool PermitValue = O->getValueExpectedFlag() != cl::ValueDisallowed;
    StringRef Flag = PermitValue ? LHS : Arg;
    for (size_t i = 0, e = OptionNames.size(); i != e; ++i) {
      StringRef Name = OptionNames[i];
      unsigned Distance = StringRef(Name).edit_distance(
        Flag, /*AllowReplacements=*/true, /*MaxEditDistance=*/BestDistance);
      if (!Best || Distance < BestDistance) {
        Best = O;
        BestDistance = Distance;
        if (RHS.empty() || !PermitValue)
          NearestString = OptionNames[i];
        else
          NearestString = std::string(OptionNames[i]) + "=" + RHS.str();
      }
    }
  }

  return Best;
}

/// CommaSeparateAndAddOccurence - A wrapper around Handler->addOccurence() that
/// does special handling of cl::CommaSeparated options.
static bool CommaSeparateAndAddOccurence(Option *Handler, unsigned pos,
                                         StringRef ArgName,
                                         StringRef Value, bool MultiArg = false)
{
  // Check to see if this option accepts a comma separated list of values.  If
  // it does, we have to split up the value into multiple values.
  if (Handler->getMiscFlags() & CommaSeparated) {
    StringRef Val(Value);
    StringRef::size_type Pos = Val.find(',');

    while (Pos != StringRef::npos) {
      // Process the portion before the comma.
      if (Handler->addOccurrence(pos, ArgName, Val.substr(0, Pos), MultiArg))
        return true;
      // Erase the portion before the comma, AND the comma.
      Val = Val.substr(Pos+1);
      Value.substr(Pos+1);  // Increment the original value pointer as well.
      // Check for another comma.
      Pos = Val.find(',');
    }

    Value = Val;
  }

  if (Handler->addOccurrence(pos, ArgName, Value, MultiArg))
    return true;

  return false;
}

/// ProvideOption - For Value, this differentiates between an empty value ("")
/// and a null value (StringRef()).  The later is accepted for arguments that
/// don't allow a value (-foo) the former is rejected (-foo=).
static inline bool ProvideOption(Option *Handler, StringRef ArgName,
                                 StringRef Value, int argc,
                                 const char *const *argv, int &i) {
  // Is this a multi-argument option?
  unsigned NumAdditionalVals = Handler->getNumAdditionalVals();

  // Enforce value requirements
  switch (Handler->getValueExpectedFlag()) {
  case ValueRequired:
    if (Value.data() == 0) {       // No value specified?
      if (i+1 >= argc)
        return Handler->error("requires a value!");
      // Steal the next argument, like for '-o filename'
      Value = argv[++i];
    }
    break;
  case ValueDisallowed:
    if (NumAdditionalVals > 0)
      return Handler->error("multi-valued option specified"
                            " with ValueDisallowed modifier!");

    if (Value.data())
      return Handler->error("does not allow a value! '" +
                            Twine(Value) + "' specified.");
    break;
  case ValueOptional:
    break;
  }

  // If this isn't a multi-arg option, just run the handler.
  if (NumAdditionalVals == 0)
    return CommaSeparateAndAddOccurence(Handler, i, ArgName, Value);

  // If it is, run the handle several times.
  bool MultiArg = false;

  if (Value.data()) {
    if (CommaSeparateAndAddOccurence(Handler, i, ArgName, Value, MultiArg))
      return true;
    --NumAdditionalVals;
    MultiArg = true;
  }

  while (NumAdditionalVals > 0) {
    if (i+1 >= argc)
      return Handler->error("not enough values!");
    Value = argv[++i];

    if (CommaSeparateAndAddOccurence(Handler, i, ArgName, Value, MultiArg))
      return true;
    MultiArg = true;
    --NumAdditionalVals;
  }
  return false;
}

static bool ProvidePositionalOption(Option *Handler, StringRef Arg, int i) {
  int Dummy = i;
  return ProvideOption(Handler, Handler->ArgStr, Arg, 0, 0, Dummy);
}


// Option predicates...
static inline bool isGrouping(const Option *O) {
  return O->getFormattingFlag() == cl::Grouping;
}
static inline bool isPrefixedOrGrouping(const Option *O) {
  return isGrouping(O) || O->getFormattingFlag() == cl::Prefix;
}

// getOptionPred - Check to see if there are any options that satisfy the
// specified predicate with names that are the prefixes in Name.  This is
// checked by progressively stripping characters off of the name, checking to
// see if there options that satisfy the predicate.  If we find one, return it,
// otherwise return null.
//
static Option *getOptionPred(StringRef Name, size_t &Length,
                             bool (*Pred)(const Option*),
                             const StringMap<Option*> &OptionsMap) {

  StringMap<Option*>::const_iterator OMI = OptionsMap.find(Name);

  // Loop while we haven't found an option and Name still has at least two
  // characters in it (so that the next iteration will not be the empty
  // string.
  while (OMI == OptionsMap.end() && Name.size() > 1) {
    Name = Name.substr(0, Name.size()-1);   // Chop off the last character.
    OMI = OptionsMap.find(Name);
  }

  if (OMI != OptionsMap.end() && Pred(OMI->second)) {
    Length = Name.size();
    return OMI->second;    // Found one!
  }
  return 0;                // No option found!
}

/// HandlePrefixedOrGroupedOption - The specified argument string (which started
/// with at least one '-') does not fully match an available option.  Check to
/// see if this is a prefix or grouped option.  If so, split arg into output an
/// Arg/Value pair and return the Option to parse it with.
static Option *HandlePrefixedOrGroupedOption(StringRef &Arg, StringRef &Value,
                                             bool &ErrorParsing,
                                         const StringMap<Option*> &OptionsMap) {
  if (Arg.size() == 1) return 0;

  // Do the lookup!
  size_t Length = 0;
  Option *PGOpt = getOptionPred(Arg, Length, isPrefixedOrGrouping, OptionsMap);
  if (PGOpt == 0) return 0;

  // If the option is a prefixed option, then the value is simply the
  // rest of the name...  so fall through to later processing, by
  // setting up the argument name flags and value fields.
  if (PGOpt->getFormattingFlag() == cl::Prefix) {
    Value = Arg.substr(Length);
    Arg = Arg.substr(0, Length);
    assert(OptionsMap.count(Arg) && OptionsMap.find(Arg)->second == PGOpt);
    return PGOpt;
  }

  // This must be a grouped option... handle them now.  Grouping options can't
  // have values.
  assert(isGrouping(PGOpt) && "Broken getOptionPred!");

  do {
    // Move current arg name out of Arg into OneArgName.
    StringRef OneArgName = Arg.substr(0, Length);
    Arg = Arg.substr(Length);

    // Because ValueRequired is an invalid flag for grouped arguments,
    // we don't need to pass argc/argv in.
    assert(PGOpt->getValueExpectedFlag() != cl::ValueRequired &&
           "Option can not be cl::Grouping AND cl::ValueRequired!");
    int Dummy = 0;
    ErrorParsing |= ProvideOption(PGOpt, OneArgName,
                                  StringRef(), 0, 0, Dummy);

    // Get the next grouping option.
    PGOpt = getOptionPred(Arg, Length, isGrouping, OptionsMap);
  } while (PGOpt && Length != Arg.size());

  // Return the last option with Arg cut down to just the last one.
  return PGOpt;
}



static bool RequiresValue(const Option *O) {
  return O->getNumOccurrencesFlag() == cl::Required ||
         O->getNumOccurrencesFlag() == cl::OneOrMore;
}

static bool EatsUnboundedNumberOfValues(const Option *O) {
  return O->getNumOccurrencesFlag() == cl::ZeroOrMore ||
         O->getNumOccurrencesFlag() == cl::OneOrMore;
}

/// ParseCStringVector - Break INPUT up wherever one or more
/// whitespace characters are found, and store the resulting tokens in
/// OUTPUT. The tokens stored in OUTPUT are dynamically allocated
/// using strdup(), so it is the caller's responsibility to free()
/// them later.
///
static void ParseCStringVector(std::vector<char *> &OutputVector,
                               const char *Input) {
  // Characters which will be treated as token separators:
  StringRef Delims = " \v\f\t\r\n";

  StringRef WorkStr(Input);
  while (!WorkStr.empty()) {
    // If the first character is a delimiter, strip them off.
    if (Delims.find(WorkStr[0]) != StringRef::npos) {
      size_t Pos = WorkStr.find_first_not_of(Delims);
      if (Pos == StringRef::npos) Pos = WorkStr.size();
      WorkStr = WorkStr.substr(Pos);
      continue;
    }

    // Find position of first delimiter.
    size_t Pos = WorkStr.find_first_of(Delims);
    if (Pos == StringRef::npos) Pos = WorkStr.size();

    // Everything from 0 to Pos is the next word to copy.
    char *NewStr = (char*)malloc(Pos+1);
    memcpy(NewStr, WorkStr.data(), Pos);
    NewStr[Pos] = 0;
    OutputVector.push_back(NewStr);

    WorkStr = WorkStr.substr(Pos);
  }
}

/// ParseEnvironmentOptions - An alternative entry point to the
/// CommandLine library, which allows you to read the program's name
/// from the caller (as PROGNAME) and its command-line arguments from
/// an environment variable (whose name is given in ENVVAR).
///
void cl::ParseEnvironmentOptions(const char *progName, const char *envVar,
                                 const char *Overview) {
  // Check args.
  assert(progName && "Program name not specified");
  assert(envVar && "Environment variable name missing");

  // Get the environment variable they want us to parse options out of.
  const char *envValue = getenv(envVar);
  if (!envValue)
    return;

  // Get program's "name", which we wouldn't know without the caller
  // telling us.
  std::vector<char*> newArgv;
  newArgv.push_back(strdup(progName));

  // Parse the value of the environment variable into a "command line"
  // and hand it off to ParseCommandLineOptions().
  ParseCStringVector(newArgv, envValue);
  int newArgc = static_cast<int>(newArgv.size());
  ParseCommandLineOptions(newArgc, &newArgv[0], Overview);

  // Free all the strdup()ed strings.
  for (std::vector<char*>::iterator i = newArgv.begin(), e = newArgv.end();
       i != e; ++i)
    free(*i);
}


/// ExpandResponseFiles - Copy the contents of argv into newArgv,
/// substituting the contents of the response files for the arguments
/// of type @file.
static void ExpandResponseFiles(unsigned argc, const char*const* argv,
                                std::vector<char*>& newArgv) {
  for (unsigned i = 1; i != argc; ++i) {
    const char *arg = argv[i];

    if (arg[0] == '@') {
      sys::PathWithStatus respFile(++arg);

      // Check that the response file is not empty (mmap'ing empty
      // files can be problematic).
      const sys::FileStatus *FileStat = respFile.getFileStatus();
      if (FileStat && FileStat->getSize() != 0) {

        // If we could open the file, parse its contents, otherwise
        // pass the @file option verbatim.

        // TODO: we should also support recursive loading of response files,
        // since this is how gcc behaves. (From their man page: "The file may
        // itself contain additional @file options; any such options will be
        // processed recursively.")

        // Mmap the response file into memory.
        OwningPtr<MemoryBuffer> respFilePtr;
        if (!MemoryBuffer::getFile(respFile.c_str(), respFilePtr)) {
          ParseCStringVector(newArgv, respFilePtr->getBufferStart());
          continue;
        }
      }
    }
    newArgv.push_back(strdup(arg));
  }
}

void cl::ParseCommandLineOptions(int argc, const char * const *argv,
                                 const char *Overview) {
  // Process all registered options.
  SmallVector<Option*, 4> PositionalOpts;
  SmallVector<Option*, 4> SinkOpts;
  StringMap<Option*> Opts;
  GetOptionInfo(PositionalOpts, SinkOpts, Opts);

  assert((!Opts.empty() || !PositionalOpts.empty()) &&
         "No options specified!");

  // Expand response files.
  std::vector<char*> newArgv;
  newArgv.push_back(strdup(argv[0]));
  ExpandResponseFiles(argc, argv, newArgv);
  argv = &newArgv[0];
  argc = static_cast<int>(newArgv.size());

  // Copy the program name into ProgName, making sure not to overflow it.
  std::string ProgName = sys::path::filename(argv[0]);
  size_t Len = std::min(ProgName.size(), size_t(79));
  memcpy(ProgramName, ProgName.data(), Len);
  ProgramName[Len] = '\0';

  ProgramOverview = Overview;
  bool ErrorParsing = false;

  // Check out the positional arguments to collect information about them.
  unsigned NumPositionalRequired = 0;

  // Determine whether or not there are an unlimited number of positionals
  bool HasUnlimitedPositionals = false;

  Option *ConsumeAfterOpt = 0;
  if (!PositionalOpts.empty()) {
    if (PositionalOpts[0]->getNumOccurrencesFlag() == cl::ConsumeAfter) {
      assert(PositionalOpts.size() > 1 &&
             "Cannot specify cl::ConsumeAfter without a positional argument!");
      ConsumeAfterOpt = PositionalOpts[0];
    }

    // Calculate how many positional values are _required_.
    bool UnboundedFound = false;
    for (size_t i = ConsumeAfterOpt != 0, e = PositionalOpts.size();
         i != e; ++i) {
      Option *Opt = PositionalOpts[i];
      if (RequiresValue(Opt))
        ++NumPositionalRequired;
      else if (ConsumeAfterOpt) {
        // ConsumeAfter cannot be combined with "optional" positional options
        // unless there is only one positional argument...
        if (PositionalOpts.size() > 2)
          ErrorParsing |=
            Opt->error("error - this positional option will never be matched, "
                       "because it does not Require a value, and a "
                       "cl::ConsumeAfter option is active!");
      } else if (UnboundedFound && !Opt->ArgStr[0]) {
        // This option does not "require" a value...  Make sure this option is
        // not specified after an option that eats all extra arguments, or this
        // one will never get any!
        //
        ErrorParsing |= Opt->error("error - option can never match, because "
                                   "another positional argument will match an "
                                   "unbounded number of values, and this option"
                                   " does not require a value!");
      }
      UnboundedFound |= EatsUnboundedNumberOfValues(Opt);
    }
    HasUnlimitedPositionals = UnboundedFound || ConsumeAfterOpt;
  }

  // PositionalVals - A vector of "positional" arguments we accumulate into
  // the process at the end.
  //
  SmallVector<std::pair<StringRef,unsigned>, 4> PositionalVals;

  // If the program has named positional arguments, and the name has been run
  // across, keep track of which positional argument was named.  Otherwise put
  // the positional args into the PositionalVals list...
  Option *ActivePositionalArg = 0;

  // Loop over all of the arguments... processing them.
  bool DashDashFound = false;  // Have we read '--'?
  for (int i = 1; i < argc; ++i) {
    Option *Handler = 0;
    Option *NearestHandler = 0;
    std::string NearestHandlerString;
    StringRef Value;
    StringRef ArgName = "";

    // If the option list changed, this means that some command line
    // option has just been registered or deregistered.  This can occur in
    // response to things like -load, etc.  If this happens, rescan the options.
    if (OptionListChanged) {
      PositionalOpts.clear();
      SinkOpts.clear();
      Opts.clear();
      GetOptionInfo(PositionalOpts, SinkOpts, Opts);
      OptionListChanged = false;
    }

    // Check to see if this is a positional argument.  This argument is
    // considered to be positional if it doesn't start with '-', if it is "-"
    // itself, or if we have seen "--" already.
    //
    if (argv[i][0] != '-' || argv[i][1] == 0 || DashDashFound) {
      // Positional argument!
      if (ActivePositionalArg) {
        ProvidePositionalOption(ActivePositionalArg, argv[i], i);
        continue;  // We are done!
      }

      if (!PositionalOpts.empty()) {
        PositionalVals.push_back(std::make_pair(argv[i],i));

        // All of the positional arguments have been fulfulled, give the rest to
        // the consume after option... if it's specified...
        //
        if (PositionalVals.size() >= NumPositionalRequired &&
            ConsumeAfterOpt != 0) {
          for (++i; i < argc; ++i)
            PositionalVals.push_back(std::make_pair(argv[i],i));
          break;   // Handle outside of the argument processing loop...
        }

        // Delay processing positional arguments until the end...
        continue;
      }
    } else if (argv[i][0] == '-' && argv[i][1] == '-' && argv[i][2] == 0 &&
               !DashDashFound) {
      DashDashFound = true;  // This is the mythical "--"?
      continue;              // Don't try to process it as an argument itself.
    } else if (ActivePositionalArg &&
               (ActivePositionalArg->getMiscFlags() & PositionalEatsArgs)) {
      // If there is a positional argument eating options, check to see if this
      // option is another positional argument.  If so, treat it as an argument,
      // otherwise feed it to the eating positional.
      ArgName = argv[i]+1;
      // Eat leading dashes.
      while (!ArgName.empty() && ArgName[0] == '-')
        ArgName = ArgName.substr(1);

      Handler = LookupOption(ArgName, Value, Opts);
      if (!Handler || Handler->getFormattingFlag() != cl::Positional) {
        ProvidePositionalOption(ActivePositionalArg, argv[i], i);
        continue;  // We are done!
      }

    } else {     // We start with a '-', must be an argument.
      ArgName = argv[i]+1;
      // Eat leading dashes.
      while (!ArgName.empty() && ArgName[0] == '-')
        ArgName = ArgName.substr(1);

      Handler = LookupOption(ArgName, Value, Opts);

      // Check to see if this "option" is really a prefixed or grouped argument.
      if (Handler == 0)
        Handler = HandlePrefixedOrGroupedOption(ArgName, Value,
                                                ErrorParsing, Opts);

      // Otherwise, look for the closest available option to report to the user
      // in the upcoming error.
      if (Handler == 0 && SinkOpts.empty())
        NearestHandler = LookupNearestOption(ArgName, Opts,
                                             NearestHandlerString);
    }

    if (Handler == 0) {
      if (SinkOpts.empty()) {
        errs() << ProgramName << ": Unknown command line argument '"
             << argv[i] << "'.  Try: '" << argv[0] << " -help'\n";

        if (NearestHandler) {
          // If we know a near match, report it as well.
          errs() << ProgramName << ": Did you mean '-"
                 << NearestHandlerString << "'?\n";
        }

        ErrorParsing = true;
      } else {
        for (SmallVectorImpl<Option*>::iterator I = SinkOpts.begin(),
               E = SinkOpts.end(); I != E ; ++I)
          (*I)->addOccurrence(i, "", argv[i]);
      }
      continue;
    }

    // If this is a named positional argument, just remember that it is the
    // active one...
    if (Handler->getFormattingFlag() == cl::Positional)
      ActivePositionalArg = Handler;
    else
      ErrorParsing |= ProvideOption(Handler, ArgName, Value, argc, argv, i);
  }

  // Check and handle positional arguments now...
  if (NumPositionalRequired > PositionalVals.size()) {
    errs() << ProgramName
         << ": Not enough positional command line arguments specified!\n"
         << "Must specify at least " << NumPositionalRequired
         << " positional arguments: See: " << argv[0] << " -help\n";

    ErrorParsing = true;
  } else if (!HasUnlimitedPositionals &&
             PositionalVals.size() > PositionalOpts.size()) {
    errs() << ProgramName
         << ": Too many positional arguments specified!\n"
         << "Can specify at most " << PositionalOpts.size()
         << " positional arguments: See: " << argv[0] << " -help\n";
    ErrorParsing = true;

  } else if (ConsumeAfterOpt == 0) {
    // Positional args have already been handled if ConsumeAfter is specified.
    unsigned ValNo = 0, NumVals = static_cast<unsigned>(PositionalVals.size());
    for (size_t i = 0, e = PositionalOpts.size(); i != e; ++i) {
      if (RequiresValue(PositionalOpts[i])) {
        ProvidePositionalOption(PositionalOpts[i], PositionalVals[ValNo].first,
                                PositionalVals[ValNo].second);
        ValNo++;
        --NumPositionalRequired;  // We fulfilled our duty...
      }

      // If we _can_ give this option more arguments, do so now, as long as we
      // do not give it values that others need.  'Done' controls whether the
      // option even _WANTS_ any more.
      //
      bool Done = PositionalOpts[i]->getNumOccurrencesFlag() == cl::Required;
      while (NumVals-ValNo > NumPositionalRequired && !Done) {
        switch (PositionalOpts[i]->getNumOccurrencesFlag()) {
        case cl::Optional:
          Done = true;          // Optional arguments want _at most_ one value
          // FALL THROUGH
        case cl::ZeroOrMore:    // Zero or more will take all they can get...
        case cl::OneOrMore:     // One or more will take all they can get...
          ProvidePositionalOption(PositionalOpts[i],
                                  PositionalVals[ValNo].first,
                                  PositionalVals[ValNo].second);
          ValNo++;
          break;
        default:
          llvm_unreachable("Internal error, unexpected NumOccurrences flag in "
                 "positional argument processing!");
        }
      }
    }
  } else {
    assert(ConsumeAfterOpt && NumPositionalRequired <= PositionalVals.size());
    unsigned ValNo = 0;
    for (size_t j = 1, e = PositionalOpts.size(); j != e; ++j)
      if (RequiresValue(PositionalOpts[j])) {
        ErrorParsing |= ProvidePositionalOption(PositionalOpts[j],
                                                PositionalVals[ValNo].first,
                                                PositionalVals[ValNo].second);
        ValNo++;
      }

    // Handle the case where there is just one positional option, and it's
    // optional.  In this case, we want to give JUST THE FIRST option to the
    // positional option and keep the rest for the consume after.  The above
    // loop would have assigned no values to positional options in this case.
    //
    if (PositionalOpts.size() == 2 && ValNo == 0 && !PositionalVals.empty()) {
      ErrorParsing |= ProvidePositionalOption(PositionalOpts[1],
                                              PositionalVals[ValNo].first,
                                              PositionalVals[ValNo].second);
      ValNo++;
    }

    // Handle over all of the rest of the arguments to the
    // cl::ConsumeAfter command line option...
    for (; ValNo != PositionalVals.size(); ++ValNo)
      ErrorParsing |= ProvidePositionalOption(ConsumeAfterOpt,
                                              PositionalVals[ValNo].first,
                                              PositionalVals[ValNo].second);
  }

  // Loop over args and make sure all required args are specified!
  for (StringMap<Option*>::iterator I = Opts.begin(),
         E = Opts.end(); I != E; ++I) {
    switch (I->second->getNumOccurrencesFlag()) {
    case Required:
    case OneOrMore:
      if (I->second->getNumOccurrences() == 0) {
        I->second->error("must be specified at least once!");
        ErrorParsing = true;
      }
      // Fall through
    default:
      break;
    }
  }

  // Now that we know if -debug is specified, we can use it.
  // Note that if ReadResponseFiles == true, this must be done before the
  // memory allocated for the expanded command line is free()d below.
  DEBUG(dbgs() << "Args: ";
        for (int i = 0; i < argc; ++i)
          dbgs() << argv[i] << ' ';
        dbgs() << '\n';
       );

  // Free all of the memory allocated to the map.  Command line options may only
  // be processed once!
  Opts.clear();
  PositionalOpts.clear();
  MoreHelp->clear();

  // Free the memory allocated by ExpandResponseFiles.
  // Free all the strdup()ed strings.
  for (std::vector<char*>::iterator i = newArgv.begin(), e = newArgv.end();
       i != e; ++i)
    free(*i);

  // If we had an error processing our arguments, don't let the program execute
  if (ErrorParsing) exit(1);
}

//===----------------------------------------------------------------------===//
// Option Base class implementation
//

bool Option::error(const Twine &Message, StringRef ArgName) {
  if (ArgName.data() == 0) ArgName = ArgStr;
  if (ArgName.empty())
    errs() << HelpStr;  // Be nice for positional arguments
  else
    errs() << ProgramName << ": for the -" << ArgName;

  errs() << " option: " << Message << "\n";
  return true;
}

bool Option::addOccurrence(unsigned pos, StringRef ArgName,
                           StringRef Value, bool MultiArg) {
  if (!MultiArg)
    NumOccurrences++;   // Increment the number of times we have been seen

  switch (getNumOccurrencesFlag()) {
  case Optional:
    if (NumOccurrences > 1)
      return error("may only occur zero or one times!", ArgName);
    break;
  case Required:
    if (NumOccurrences > 1)
      return error("must occur exactly one time!", ArgName);
    // Fall through
  case OneOrMore:
  case ZeroOrMore:
  case ConsumeAfter: break;
  }

  return handleOccurrence(pos, ArgName, Value);
}


// getValueStr - Get the value description string, using "DefaultMsg" if nothing
// has been specified yet.
//
static const char *getValueStr(const Option &O, const char *DefaultMsg) {
  if (O.ValueStr[0] == 0) return DefaultMsg;
  return O.ValueStr;
}

//===----------------------------------------------------------------------===//
// cl::alias class implementation
//

// Return the width of the option tag for printing...
size_t alias::getOptionWidth() const {
  return std::strlen(ArgStr)+6;
}

// Print out the option for the alias.
void alias::printOptionInfo(size_t GlobalWidth) const {
  size_t L = std::strlen(ArgStr);
  outs() << "  -" << ArgStr;
  outs().indent(GlobalWidth-L-6) << " - " << HelpStr << "\n";
}

//===----------------------------------------------------------------------===//
// Parser Implementation code...
//

// basic_parser implementation
//

// Return the width of the option tag for printing...
size_t basic_parser_impl::getOptionWidth(const Option &O) const {
  size_t Len = std::strlen(O.ArgStr);
  if (const char *ValName = getValueName())
    Len += std::strlen(getValueStr(O, ValName))+3;

  return Len + 6;
}

// printOptionInfo - Print out information about this option.  The
// to-be-maintained width is specified.
//
void basic_parser_impl::printOptionInfo(const Option &O,
                                        size_t GlobalWidth) const {
  outs() << "  -" << O.ArgStr;

  if (const char *ValName = getValueName())
    outs() << "=<" << getValueStr(O, ValName) << '>';

  outs().indent(GlobalWidth-getOptionWidth(O)) << " - " << O.HelpStr << '\n';
}

void basic_parser_impl::printOptionName(const Option &O,
                                        size_t GlobalWidth) const {
  outs() << "  -" << O.ArgStr;
  outs().indent(GlobalWidth-std::strlen(O.ArgStr));
}


// parser<bool> implementation
//
bool parser<bool>::parse(Option &O, StringRef ArgName,
                         StringRef Arg, bool &Value) {
  if (Arg == "" || Arg == "true" || Arg == "TRUE" || Arg == "True" ||
      Arg == "1") {
    Value = true;
    return false;
  }

  if (Arg == "false" || Arg == "FALSE" || Arg == "False" || Arg == "0") {
    Value = false;
    return false;
  }
  return O.error("'" + Arg +
                 "' is invalid value for boolean argument! Try 0 or 1");
}

// parser<boolOrDefault> implementation
//
bool parser<boolOrDefault>::parse(Option &O, StringRef ArgName,
                                  StringRef Arg, boolOrDefault &Value) {
  if (Arg == "" || Arg == "true" || Arg == "TRUE" || Arg == "True" ||
      Arg == "1") {
    Value = BOU_TRUE;
    return false;
  }
  if (Arg == "false" || Arg == "FALSE" || Arg == "False" || Arg == "0") {
    Value = BOU_FALSE;
    return false;
  }

  return O.error("'" + Arg +
                 "' is invalid value for boolean argument! Try 0 or 1");
}

// parser<int> implementation
//
bool parser<int>::parse(Option &O, StringRef ArgName,
                        StringRef Arg, int &Value) {
  if (Arg.getAsInteger(0, Value))
    return O.error("'" + Arg + "' value invalid for integer argument!");
  return false;
}

// parser<unsigned> implementation
//
bool parser<unsigned>::parse(Option &O, StringRef ArgName,
                             StringRef Arg, unsigned &Value) {

  if (Arg.getAsInteger(0, Value))
    return O.error("'" + Arg + "' value invalid for uint argument!");
  return false;
}

// parser<unsigned long long> implementation
//
bool parser<unsigned long long>::parse(Option &O, StringRef ArgName,
                                      StringRef Arg, unsigned long long &Value){

  if (Arg.getAsInteger(0, Value))
    return O.error("'" + Arg + "' value invalid for uint argument!");
  return false;
}

// parser<double>/parser<float> implementation
//
static bool parseDouble(Option &O, StringRef Arg, double &Value) {
  SmallString<32> TmpStr(Arg.begin(), Arg.end());
  const char *ArgStart = TmpStr.c_str();
  char *End;
  Value = strtod(ArgStart, &End);
  if (*End != 0)
    return O.error("'" + Arg + "' value invalid for floating point argument!");
  return false;
}

bool parser<double>::parse(Option &O, StringRef ArgName,
                           StringRef Arg, double &Val) {
  return parseDouble(O, Arg, Val);
}

bool parser<float>::parse(Option &O, StringRef ArgName,
                          StringRef Arg, float &Val) {
  double dVal;
  if (parseDouble(O, Arg, dVal))
    return true;
  Val = (float)dVal;
  return false;
}



// generic_parser_base implementation
//

// findOption - Return the option number corresponding to the specified
// argument string.  If the option is not found, getNumOptions() is returned.
//
unsigned generic_parser_base::findOption(const char *Name) {
  unsigned e = getNumOptions();

  for (unsigned i = 0; i != e; ++i) {
    if (strcmp(getOption(i), Name) == 0)
      return i;
  }
  return e;
}


// Return the width of the option tag for printing...
size_t generic_parser_base::getOptionWidth(const Option &O) const {
  if (O.hasArgStr()) {
    size_t Size = std::strlen(O.ArgStr)+6;
    for (unsigned i = 0, e = getNumOptions(); i != e; ++i)
      Size = std::max(Size, std::strlen(getOption(i))+8);
    return Size;
  } else {
    size_t BaseSize = 0;
    for (unsigned i = 0, e = getNumOptions(); i != e; ++i)
      BaseSize = std::max(BaseSize, std::strlen(getOption(i))+8);
    return BaseSize;
  }
}

// printOptionInfo - Print out information about this option.  The
// to-be-maintained width is specified.
//
void generic_parser_base::printOptionInfo(const Option &O,
                                          size_t GlobalWidth) const {
  if (O.hasArgStr()) {
    size_t L = std::strlen(O.ArgStr);
    outs() << "  -" << O.ArgStr;
    outs().indent(GlobalWidth-L-6) << " - " << O.HelpStr << '\n';

    for (unsigned i = 0, e = getNumOptions(); i != e; ++i) {
      size_t NumSpaces = GlobalWidth-strlen(getOption(i))-8;
      outs() << "    =" << getOption(i);
      outs().indent(NumSpaces) << " -   " << getDescription(i) << '\n';
    }
  } else {
    if (O.HelpStr[0])
      outs() << "  " << O.HelpStr << '\n';
    for (unsigned i = 0, e = getNumOptions(); i != e; ++i) {
      size_t L = std::strlen(getOption(i));
      outs() << "    -" << getOption(i);
      outs().indent(GlobalWidth-L-8) << " - " << getDescription(i) << '\n';
    }
  }
}

static const size_t MaxOptWidth = 8; // arbitrary spacing for printOptionDiff

// printGenericOptionDiff - Print the value of this option and it's default.
//
// "Generic" options have each value mapped to a name.
void generic_parser_base::
printGenericOptionDiff(const Option &O, const GenericOptionValue &Value,
                       const GenericOptionValue &Default,
                       size_t GlobalWidth) const {
  outs() << "  -" << O.ArgStr;
  outs().indent(GlobalWidth-std::strlen(O.ArgStr));

  unsigned NumOpts = getNumOptions();
  for (unsigned i = 0; i != NumOpts; ++i) {
    if (Value.compare(getOptionValue(i)))
      continue;

    outs() << "= " << getOption(i);
    size_t L = std::strlen(getOption(i));
    size_t NumSpaces = MaxOptWidth > L ? MaxOptWidth - L : 0;
    outs().indent(NumSpaces) << " (default: ";
    for (unsigned j = 0; j != NumOpts; ++j) {
      if (Default.compare(getOptionValue(j)))
        continue;
      outs() << getOption(j);
      break;
    }
    outs() << ")\n";
    return;
  }
  outs() << "= *unknown option value*\n";
}

// printOptionDiff - Specializations for printing basic value types.
//
#define PRINT_OPT_DIFF(T)                                               \
  void parser<T>::                                                      \
  printOptionDiff(const Option &O, T V, OptionValue<T> D,               \
                  size_t GlobalWidth) const {                           \
    printOptionName(O, GlobalWidth);                                    \
    std::string Str;                                                    \
    {                                                                   \
      raw_string_ostream SS(Str);                                       \
      SS << V;                                                          \
    }                                                                   \
    outs() << "= " << Str;                                              \
    size_t NumSpaces = MaxOptWidth > Str.size() ? MaxOptWidth - Str.size() : 0;\
    outs().indent(NumSpaces) << " (default: ";                          \
    if (D.hasValue())                                                   \
      outs() << D.getValue();                                           \
    else                                                                \
      outs() << "*no default*";                                         \
    outs() << ")\n";                                                    \
  }                                                                     \

PRINT_OPT_DIFF(bool)
PRINT_OPT_DIFF(boolOrDefault)
PRINT_OPT_DIFF(int)
PRINT_OPT_DIFF(unsigned)
PRINT_OPT_DIFF(unsigned long long)
PRINT_OPT_DIFF(double)
PRINT_OPT_DIFF(float)
PRINT_OPT_DIFF(char)

void parser<std::string>::
printOptionDiff(const Option &O, StringRef V, OptionValue<std::string> D,
                size_t GlobalWidth) const {
  printOptionName(O, GlobalWidth);
  outs() << "= " << V;
  size_t NumSpaces = MaxOptWidth > V.size() ? MaxOptWidth - V.size() : 0;
  outs().indent(NumSpaces) << " (default: ";
  if (D.hasValue())
    outs() << D.getValue();
  else
    outs() << "*no default*";
  outs() << ")\n";
}

// Print a placeholder for options that don't yet support printOptionDiff().
void basic_parser_impl::
printOptionNoValue(const Option &O, size_t GlobalWidth) const {
  printOptionName(O, GlobalWidth);
  outs() << "= *cannot print option value*\n";
}

//===----------------------------------------------------------------------===//
// -help and -help-hidden option implementation
//

static int OptNameCompare(const void *LHS, const void *RHS) {
  typedef std::pair<const char *, Option*> pair_ty;

  return strcmp(((const pair_ty*)LHS)->first, ((const pair_ty*)RHS)->first);
}

// Copy Options into a vector so we can sort them as we like.
static void
sortOpts(StringMap<Option*> &OptMap,
         SmallVectorImpl< std::pair<const char *, Option*> > &Opts,
         bool ShowHidden) {
  SmallPtrSet<Option*, 128> OptionSet;  // Duplicate option detection.

  for (StringMap<Option*>::iterator I = OptMap.begin(), E = OptMap.end();
       I != E; ++I) {
    // Ignore really-hidden options.
    if (I->second->getOptionHiddenFlag() == ReallyHidden)
      continue;

    // Unless showhidden is set, ignore hidden flags.
    if (I->second->getOptionHiddenFlag() == Hidden && !ShowHidden)
      continue;

    // If we've already seen this option, don't add it to the list again.
    if (!OptionSet.insert(I->second))
      continue;

    Opts.push_back(std::pair<const char *, Option*>(I->getKey().data(),
                                                    I->second));
  }

  // Sort the options list alphabetically.
  qsort(Opts.data(), Opts.size(), sizeof(Opts[0]), OptNameCompare);
}

namespace {

class HelpPrinter {
  size_t MaxArgLen;
  const Option *EmptyArg;
  const bool ShowHidden;

public:
  explicit HelpPrinter(bool showHidden) : ShowHidden(showHidden) {
    EmptyArg = 0;
  }

  void operator=(bool Value) {
    if (Value == false) return;

    // Get all the options.
    SmallVector<Option*, 4> PositionalOpts;
    SmallVector<Option*, 4> SinkOpts;
    StringMap<Option*> OptMap;
    GetOptionInfo(PositionalOpts, SinkOpts, OptMap);

    SmallVector<std::pair<const char *, Option*>, 128> Opts;
    sortOpts(OptMap, Opts, ShowHidden);

    if (ProgramOverview)
      outs() << "OVERVIEW: " << ProgramOverview << "\n";

    outs() << "USAGE: " << ProgramName << " [options]";

    // Print out the positional options.
    Option *CAOpt = 0;   // The cl::ConsumeAfter option, if it exists...
    if (!PositionalOpts.empty() &&
        PositionalOpts[0]->getNumOccurrencesFlag() == ConsumeAfter)
      CAOpt = PositionalOpts[0];

    for (size_t i = CAOpt != 0, e = PositionalOpts.size(); i != e; ++i) {
      if (PositionalOpts[i]->ArgStr[0])
        outs() << " --" << PositionalOpts[i]->ArgStr;
      outs() << " " << PositionalOpts[i]->HelpStr;
    }

    // Print the consume after option info if it exists...
    if (CAOpt) outs() << " " << CAOpt->HelpStr;

    outs() << "\n\n";

    // Compute the maximum argument length...
    MaxArgLen = 0;
    for (size_t i = 0, e = Opts.size(); i != e; ++i)
      MaxArgLen = std::max(MaxArgLen, Opts[i].second->getOptionWidth());

    outs() << "OPTIONS:\n";
    for (size_t i = 0, e = Opts.size(); i != e; ++i)
      Opts[i].second->printOptionInfo(MaxArgLen);

    // Print any extra help the user has declared.
    for (std::vector<const char *>::iterator I = MoreHelp->begin(),
          E = MoreHelp->end(); I != E; ++I)
      outs() << *I;
    MoreHelp->clear();

    // Halt the program since help information was printed
    exit(1);
  }
};
} // End anonymous namespace

// Define the two HelpPrinter instances that are used to print out help, or
// help-hidden...
//
static HelpPrinter NormalPrinter(false);
static HelpPrinter HiddenPrinter(true);

static cl::opt<HelpPrinter, true, parser<bool> >
HOp("help", cl::desc("Display available options (-help-hidden for more)"),
    cl::location(NormalPrinter), cl::ValueDisallowed);

static cl::opt<HelpPrinter, true, parser<bool> >
HHOp("help-hidden", cl::desc("Display all available options"),
     cl::location(HiddenPrinter), cl::Hidden, cl::ValueDisallowed);

static cl::opt<bool>
PrintOptions("print-options",
             cl::desc("Print non-default options after command line parsing"),
             cl::Hidden, cl::init(false));

static cl::opt<bool>
PrintAllOptions("print-all-options",
                cl::desc("Print all option values after command line parsing"),
                cl::Hidden, cl::init(false));

// Print the value of each option.
void cl::PrintOptionValues() {
  if (!PrintOptions && !PrintAllOptions) return;

  // Get all the options.
  SmallVector<Option*, 4> PositionalOpts;
  SmallVector<Option*, 4> SinkOpts;
  StringMap<Option*> OptMap;
  GetOptionInfo(PositionalOpts, SinkOpts, OptMap);

  SmallVector<std::pair<const char *, Option*>, 128> Opts;
  sortOpts(OptMap, Opts, /*ShowHidden*/true);

  // Compute the maximum argument length...
  size_t MaxArgLen = 0;
  for (size_t i = 0, e = Opts.size(); i != e; ++i)
    MaxArgLen = std::max(MaxArgLen, Opts[i].second->getOptionWidth());

  for (size_t i = 0, e = Opts.size(); i != e; ++i)
    Opts[i].second->printOptionValue(MaxArgLen, PrintAllOptions);
}

static void (*OverrideVersionPrinter)() = 0;

static std::vector<void (*)()>* ExtraVersionPrinters = 0;

namespace {
class VersionPrinter {
public:
  void print() {
    raw_ostream &OS = outs();
    OS << "LLVM (http://llvm.org/):\n"
       << "  " << PACKAGE_NAME << " version " << PACKAGE_VERSION;
#ifdef LLVM_VERSION_INFO
    OS << LLVM_VERSION_INFO;
#endif
    OS << "\n  ";
#ifndef __OPTIMIZE__
    OS << "DEBUG build";
#else
    OS << "Optimized build";
#endif
#ifndef NDEBUG
    OS << " with assertions";
#endif
    std::string CPU = sys::getHostCPUName();
    if (CPU == "generic") CPU = "(unknown)";
    OS << ".\n"
#if (ENABLE_TIMESTAMPS == 1)
       << "  Built " << __DATE__ << " (" << __TIME__ << ").\n"
#endif
       << "  Default target: " << sys::getDefaultTargetTriple() << '\n'
       << "  Host CPU: " << CPU << '\n';
  }
  void operator=(bool OptionWasSpecified) {
    if (!OptionWasSpecified) return;

    if (OverrideVersionPrinter != 0) {
      (*OverrideVersionPrinter)();
      exit(1);
    }
    print();

    // Iterate over any registered extra printers and call them to add further
    // information.
    if (ExtraVersionPrinters != 0) {
      outs() << '\n';
      for (std::vector<void (*)()>::iterator I = ExtraVersionPrinters->begin(),
                                             E = ExtraVersionPrinters->end();
           I != E; ++I)
        (*I)();
    }

    exit(1);
  }
};
} // End anonymous namespace


// Define the --version option that prints out the LLVM version for the tool
static VersionPrinter VersionPrinterInstance;

static cl::opt<VersionPrinter, true, parser<bool> >
VersOp("version", cl::desc("Display the version of this program"),
    cl::location(VersionPrinterInstance), cl::ValueDisallowed);

// Utility function for printing the help message.
void cl::PrintHelpMessage() {
  // This looks weird, but it actually prints the help message. The
  // NormalPrinter variable is a HelpPrinter and the help gets printed when
  // its operator= is invoked. That's because the "normal" usages of the
  // help printer is to be assigned true/false depending on whether the
  // -help option was given or not. Since we're circumventing that we have
  // to make it look like -help was given, so we assign true.
  NormalPrinter = true;
}

/// Utility function for printing version number.
void cl::PrintVersionMessage() {
  VersionPrinterInstance.print();
}

void cl::SetVersionPrinter(void (*func)()) {
  OverrideVersionPrinter = func;
}

void cl::AddExtraVersionPrinter(void (*func)()) {
  if (ExtraVersionPrinters == 0)
    ExtraVersionPrinters = new std::vector<void (*)()>;

  ExtraVersionPrinters->push_back(func);
}
