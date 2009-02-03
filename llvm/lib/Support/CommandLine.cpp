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

#include "llvm/Config/config.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Streams.h"
#include "llvm/System/Path.h"
#include <algorithm>
#include <functional>
#include <map>
#include <ostream>
#include <set>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <climits>
using namespace llvm;
using namespace cl;

//===----------------------------------------------------------------------===//
// Template instantiations and anchors.
//
TEMPLATE_INSTANTIATION(class basic_parser<bool>);
TEMPLATE_INSTANTIATION(class basic_parser<boolOrDefault>);
TEMPLATE_INSTANTIATION(class basic_parser<int>);
TEMPLATE_INSTANTIATION(class basic_parser<unsigned>);
TEMPLATE_INSTANTIATION(class basic_parser<double>);
TEMPLATE_INSTANTIATION(class basic_parser<float>);
TEMPLATE_INSTANTIATION(class basic_parser<std::string>);

TEMPLATE_INSTANTIATION(class opt<unsigned>);
TEMPLATE_INSTANTIATION(class opt<int>);
TEMPLATE_INSTANTIATION(class opt<std::string>);
TEMPLATE_INSTANTIATION(class opt<bool>);

void Option::anchor() {}
void basic_parser_impl::anchor() {}
void parser<bool>::anchor() {}
void parser<boolOrDefault>::anchor() {}
void parser<int>::anchor() {}
void parser<unsigned>::anchor() {}
void parser<double>::anchor() {}
void parser<float>::anchor() {}
void parser<std::string>::anchor() {}

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
static void GetOptionInfo(std::vector<Option*> &PositionalOpts,
                          std::vector<Option*> &SinkOpts,
                          std::map<std::string, Option*> &OptionsMap) {
  std::vector<const char*> OptionNames;
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
      if (!OptionsMap.insert(std::pair<std::string,Option*>(OptionNames[i],
                                                            O)).second) {
        cerr << ProgramName << ": CommandLine Error: Argument '"
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
/// that as well.
static Option *LookupOption(const char *&Arg, const char *&Value,
                            std::map<std::string, Option*> &OptionsMap) {
  while (*Arg == '-') ++Arg;  // Eat leading dashes

  const char *ArgEnd = Arg;
  while (*ArgEnd && *ArgEnd != '=')
    ++ArgEnd; // Scan till end of argument name.

  if (*ArgEnd == '=')  // If we have an equals sign...
    Value = ArgEnd+1;  // Get the value, not the equals


  if (*Arg == 0) return 0;

  // Look up the option.
  std::map<std::string, Option*>::iterator I =
    OptionsMap.find(std::string(Arg, ArgEnd));
  return I != OptionsMap.end() ? I->second : 0;
}

static inline bool ProvideOption(Option *Handler, const char *ArgName,
                                 const char *Value, int argc, char **argv,
                                 int &i) {
  // Is this a multi-argument option?
  unsigned NumAdditionalVals = Handler->getNumAdditionalVals();

  // Enforce value requirements
  switch (Handler->getValueExpectedFlag()) {
  case ValueRequired:
    if (Value == 0) {       // No value specified?
      if (i+1 < argc) {     // Steal the next argument, like for '-o filename'
        Value = argv[++i];
      } else {
        return Handler->error(" requires a value!");
      }
    }
    break;
  case ValueDisallowed:
    if (NumAdditionalVals > 0)
      return Handler->error(": multi-valued option specified"
      " with ValueDisallowed modifier!");

    if (Value)
      return Handler->error(" does not allow a value! '" +
                            std::string(Value) + "' specified.");
    break;
  case ValueOptional:
    break;
  default:
    cerr << ProgramName
         << ": Bad ValueMask flag! CommandLine usage error:"
         << Handler->getValueExpectedFlag() << "\n";
    abort();
    break;
  }

  // If this isn't a multi-arg option, just run the handler.
  if (NumAdditionalVals == 0) {
    return Handler->addOccurrence(i, ArgName, Value ? Value : "");
  }
  // If it is, run the handle several times.
  else {
    bool MultiArg = false;

    if (Value) {
      if (Handler->addOccurrence(i, ArgName, Value, MultiArg))
        return true;
      --NumAdditionalVals;
      MultiArg = true;
    }

    while (NumAdditionalVals > 0) {

      if (i+1 < argc) {
        Value = argv[++i];
      } else {
        return Handler->error(": not enough values!");
      }
      if (Handler->addOccurrence(i, ArgName, Value, MultiArg))
        return true;
      MultiArg = true;
      --NumAdditionalVals;
    }
    return false;
  }
}

static bool ProvidePositionalOption(Option *Handler, const std::string &Arg,
                                    int i) {
  int Dummy = i;
  return ProvideOption(Handler, Handler->ArgStr, Arg.c_str(), 0, 0, Dummy);
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
static Option *getOptionPred(std::string Name, size_t &Length,
                             bool (*Pred)(const Option*),
                             std::map<std::string, Option*> &OptionsMap) {

  std::map<std::string, Option*>::iterator OMI = OptionsMap.find(Name);
  if (OMI != OptionsMap.end() && Pred(OMI->second)) {
    Length = Name.length();
    return OMI->second;
  }

  if (Name.size() == 1) return 0;
  do {
    Name.erase(Name.end()-1, Name.end());   // Chop off the last character...
    OMI = OptionsMap.find(Name);

    // Loop while we haven't found an option and Name still has at least two
    // characters in it (so that the next iteration will not be the empty
    // string...
  } while ((OMI == OptionsMap.end() || !Pred(OMI->second)) && Name.size() > 1);

  if (OMI != OptionsMap.end() && Pred(OMI->second)) {
    Length = Name.length();
    return OMI->second;    // Found one!
  }
  return 0;                // No option found!
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
/// using strdup (), so it is the caller's responsibility to free ()
/// them later.
///
static void ParseCStringVector(std::vector<char *> &output,
                               const char *input) {
  // Characters which will be treated as token separators:
  static const char *const delims = " \v\f\t\r\n";

  std::string work (input);
  // Skip past any delims at head of input string.
  size_t pos = work.find_first_not_of (delims);
  // If the string consists entirely of delims, then exit early.
  if (pos == std::string::npos) return;
  // Otherwise, jump forward to beginning of first word.
  work = work.substr (pos);
  // Find position of first delimiter.
  pos = work.find_first_of (delims);

  while (!work.empty() && pos != std::string::npos) {
    // Everything from 0 to POS is the next word to copy.
    output.push_back (strdup (work.substr (0,pos).c_str ()));
    // Is there another word in the string?
    size_t nextpos = work.find_first_not_of (delims, pos + 1);
    if (nextpos != std::string::npos) {
      // Yes? Then remove delims from beginning ...
      work = work.substr (work.find_first_not_of (delims, pos + 1));
      // and find the end of the word.
      pos = work.find_first_of (delims);
    } else {
      // No? (Remainder of string is delims.) End the loop.
      work = "";
      pos = std::string::npos;
    }
  }

  // If `input' ended with non-delim char, then we'll get here with
  // the last word of `input' in `work'; copy it now.
  if (!work.empty ()) {
    output.push_back (strdup (work.c_str ()));
  }
}

/// ParseEnvironmentOptions - An alternative entry point to the
/// CommandLine library, which allows you to read the program's name
/// from the caller (as PROGNAME) and its command-line arguments from
/// an environment variable (whose name is given in ENVVAR).
///
void cl::ParseEnvironmentOptions(const char *progName, const char *envVar,
                                 const char *Overview, bool ReadResponseFiles) {
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
  ParseCommandLineOptions(newArgc, &newArgv[0], Overview, ReadResponseFiles);

  // Free all the strdup()ed strings.
  for (std::vector<char*>::iterator i = newArgv.begin(), e = newArgv.end();
       i != e; ++i)
    free (*i);
}


/// ExpandResponseFiles - Copy the contents of argv into newArgv,
/// substituting the contents of the response files for the arguments
/// of type @file.
static void ExpandResponseFiles(int argc, char** argv,
                                std::vector<char*>& newArgv) {
  for (int i = 1; i != argc; ++i) {
    char* arg = argv[i];

    if (arg[0] == '@') {

      sys::PathWithStatus respFile(++arg);

      // Check that the response file is not empty (mmap'ing empty
      // files can be problematic).
      const sys::FileStatus *FileStat = respFile.getFileStatus();
      if (FileStat && FileStat->getSize() != 0) {

        // Mmap the response file into memory.
        OwningPtr<MemoryBuffer>
          respFilePtr(MemoryBuffer::getFile(respFile.c_str()));

        // If we could open the file, parse its contents, otherwise
        // pass the @file option verbatim.

        // TODO: we should also support recursive loading of response files,
        // since this is how gcc behaves. (From their man page: "The file may
        // itself contain additional @file options; any such options will be
        // processed recursively.")

        if (respFilePtr != 0) {
          ParseCStringVector(newArgv, respFilePtr->getBufferStart());
          continue;
        }
      }
    }
    newArgv.push_back(strdup(arg));
  }
}

void cl::ParseCommandLineOptions(int argc, char **argv,
                                 const char *Overview, bool ReadResponseFiles) {
  // Process all registered options.
  std::vector<Option*> PositionalOpts;
  std::vector<Option*> SinkOpts;
  std::map<std::string, Option*> Opts;
  GetOptionInfo(PositionalOpts, SinkOpts, Opts);

  assert((!Opts.empty() || !PositionalOpts.empty()) &&
         "No options specified!");

  // Expand response files.
  std::vector<char*> newArgv;
  if (ReadResponseFiles) {
    newArgv.push_back(strdup(argv[0]));
    ExpandResponseFiles(argc, argv, newArgv);
    argv = &newArgv[0];
    argc = static_cast<int>(newArgv.size());
  }

  // Copy the program name into ProgName, making sure not to overflow it.
  std::string ProgName = sys::Path(argv[0]).getLast();
  if (ProgName.size() > 79) ProgName.resize(79);
  strcpy(ProgramName, ProgName.c_str());

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
            Opt->error(" error - this positional option will never be matched, "
                       "because it does not Require a value, and a "
                       "cl::ConsumeAfter option is active!");
      } else if (UnboundedFound && !Opt->ArgStr[0]) {
        // This option does not "require" a value...  Make sure this option is
        // not specified after an option that eats all extra arguments, or this
        // one will never get any!
        //
        ErrorParsing |= Opt->error(" error - option can never match, because "
                                   "another positional argument will match an "
                                   "unbounded number of values, and this option"
                                   " does not require a value!");
      }
      UnboundedFound |= EatsUnboundedNumberOfValues(Opt);
    }
    HasUnlimitedPositionals = UnboundedFound || ConsumeAfterOpt;
  }

  // PositionalVals - A vector of "positional" arguments we accumulate into
  // the process at the end...
  //
  std::vector<std::pair<std::string,unsigned> > PositionalVals;

  // If the program has named positional arguments, and the name has been run
  // across, keep track of which positional argument was named.  Otherwise put
  // the positional args into the PositionalVals list...
  Option *ActivePositionalArg = 0;

  // Loop over all of the arguments... processing them.
  bool DashDashFound = false;  // Have we read '--'?
  for (int i = 1; i < argc; ++i) {
    Option *Handler = 0;
    const char *Value = 0;
    const char *ArgName = "";

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
      } else if (!PositionalOpts.empty()) {
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
      Handler = LookupOption(ArgName, Value, Opts);
      if (!Handler || Handler->getFormattingFlag() != cl::Positional) {
        ProvidePositionalOption(ActivePositionalArg, argv[i], i);
        continue;  // We are done!
      }

    } else {     // We start with a '-', must be an argument...
      ArgName = argv[i]+1;
      Handler = LookupOption(ArgName, Value, Opts);

      // Check to see if this "option" is really a prefixed or grouped argument.
      if (Handler == 0) {
        std::string RealName(ArgName);
        if (RealName.size() > 1) {
          size_t Length = 0;
          Option *PGOpt = getOptionPred(RealName, Length, isPrefixedOrGrouping,
                                        Opts);

          // If the option is a prefixed option, then the value is simply the
          // rest of the name...  so fall through to later processing, by
          // setting up the argument name flags and value fields.
          //
          if (PGOpt && PGOpt->getFormattingFlag() == cl::Prefix) {
            Value = ArgName+Length;
            assert(Opts.find(std::string(ArgName, Value)) != Opts.end() &&
                   Opts.find(std::string(ArgName, Value))->second == PGOpt);
            Handler = PGOpt;
          } else if (PGOpt) {
            // This must be a grouped option... handle them now.
            assert(isGrouping(PGOpt) && "Broken getOptionPred!");

            do {
              // Move current arg name out of RealName into RealArgName...
              std::string RealArgName(RealName.begin(),
                                      RealName.begin() + Length);
              RealName.erase(RealName.begin(), RealName.begin() + Length);

              // Because ValueRequired is an invalid flag for grouped arguments,
              // we don't need to pass argc/argv in...
              //
              assert(PGOpt->getValueExpectedFlag() != cl::ValueRequired &&
                     "Option can not be cl::Grouping AND cl::ValueRequired!");
              int Dummy;
              ErrorParsing |= ProvideOption(PGOpt, RealArgName.c_str(),
                                            0, 0, 0, Dummy);

              // Get the next grouping option...
              PGOpt = getOptionPred(RealName, Length, isGrouping, Opts);
            } while (PGOpt && Length != RealName.size());

            Handler = PGOpt; // Ate all of the options.
          }
        }
      }
    }

    if (Handler == 0) {
      if (SinkOpts.empty()) {
        cerr << ProgramName << ": Unknown command line argument '"
             << argv[i] << "'.  Try: '" << argv[0] << " --help'\n";
        ErrorParsing = true;
      } else {
        for (std::vector<Option*>::iterator I = SinkOpts.begin(),
               E = SinkOpts.end(); I != E ; ++I)
          (*I)->addOccurrence(i, "", argv[i]);
      }
      continue;
    }

    // Check to see if this option accepts a comma separated list of values.  If
    // it does, we have to split up the value into multiple values...
    if (Value && Handler->getMiscFlags() & CommaSeparated) {
      std::string Val(Value);
      std::string::size_type Pos = Val.find(',');

      while (Pos != std::string::npos) {
        // Process the portion before the comma...
        ErrorParsing |= ProvideOption(Handler, ArgName,
                                      std::string(Val.begin(),
                                                  Val.begin()+Pos).c_str(),
                                      argc, argv, i);
        // Erase the portion before the comma, AND the comma...
        Val.erase(Val.begin(), Val.begin()+Pos+1);
        Value += Pos+1;  // Increment the original value pointer as well...

        // Check for another comma...
        Pos = Val.find(',');
      }
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
    cerr << ProgramName
         << ": Not enough positional command line arguments specified!\n"
         << "Must specify at least " << NumPositionalRequired
         << " positional arguments: See: " << argv[0] << " --help\n";

    ErrorParsing = true;
  } else if (!HasUnlimitedPositionals
             && PositionalVals.size() > PositionalOpts.size()) {
    cerr << ProgramName
         << ": Too many positional arguments specified!\n"
         << "Can specify at most " << PositionalOpts.size()
         << " positional arguments: See: " << argv[0] << " --help\n";
    ErrorParsing = true;

  } else if (ConsumeAfterOpt == 0) {
    // Positional args have already been handled if ConsumeAfter is specified...
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
          assert(0 && "Internal error, unexpected NumOccurrences flag in "
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
  for (std::map<std::string, Option*>::iterator I = Opts.begin(),
         E = Opts.end(); I != E; ++I) {
    switch (I->second->getNumOccurrencesFlag()) {
    case Required:
    case OneOrMore:
      if (I->second->getNumOccurrences() == 0) {
        I->second->error(" must be specified at least once!");
        ErrorParsing = true;
      }
      // Fall through
    default:
      break;
    }
  }

  // Free all of the memory allocated to the map.  Command line options may only
  // be processed once!
  Opts.clear();
  PositionalOpts.clear();
  MoreHelp->clear();

  // Free the memory allocated by ExpandResponseFiles.
  if (ReadResponseFiles) {
    // Free all the strdup()ed strings.
    for (std::vector<char*>::iterator i = newArgv.begin(), e = newArgv.end();
         i != e; ++i)
      free (*i);
  }

  // If we had an error processing our arguments, don't let the program execute
  if (ErrorParsing) exit(1);
}

//===----------------------------------------------------------------------===//
// Option Base class implementation
//

bool Option::error(std::string Message, const char *ArgName) {
  if (ArgName == 0) ArgName = ArgStr;
  if (ArgName[0] == 0)
    cerr << HelpStr;  // Be nice for positional arguments
  else
    cerr << ProgramName << ": for the -" << ArgName;

  cerr << " option: " << Message << "\n";
  return true;
}

bool Option::addOccurrence(unsigned pos, const char *ArgName,
                           const std::string &Value,
                           bool MultiArg) {
  if (!MultiArg)
    NumOccurrences++;   // Increment the number of times we have been seen

  switch (getNumOccurrencesFlag()) {
  case Optional:
    if (NumOccurrences > 1)
      return error(": may only occur zero or one times!", ArgName);
    break;
  case Required:
    if (NumOccurrences > 1)
      return error(": must occur exactly one time!", ArgName);
    // Fall through
  case OneOrMore:
  case ZeroOrMore:
  case ConsumeAfter: break;
  default: return error(": bad num occurrences flag value!");
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
  cout << "  -" << ArgStr << std::string(GlobalWidth-L-6, ' ') << " - "
       << HelpStr << "\n";
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
  cout << "  -" << O.ArgStr;

  if (const char *ValName = getValueName())
    cout << "=<" << getValueStr(O, ValName) << ">";

  cout << std::string(GlobalWidth-getOptionWidth(O), ' ') << " - "
       << O.HelpStr << "\n";
}




// parser<bool> implementation
//
bool parser<bool>::parse(Option &O, const char *ArgName,
                         const std::string &Arg, bool &Value) {
  if (Arg == "" || Arg == "true" || Arg == "TRUE" || Arg == "True" ||
      Arg == "1") {
    Value = true;
  } else if (Arg == "false" || Arg == "FALSE" || Arg == "False" || Arg == "0") {
    Value = false;
  } else {
    return O.error(": '" + Arg +
                   "' is invalid value for boolean argument! Try 0 or 1");
  }
  if (IsInvertable && strncmp(ArgName+1, "no-", 3) == 0)
    Value = !Value;
  return false;
}

// parser<boolOrDefault> implementation
//
bool parser<boolOrDefault>::parse(Option &O, const char *ArgName,
                         const std::string &Arg, boolOrDefault &Value) {
  if (Arg == "" || Arg == "true" || Arg == "TRUE" || Arg == "True" ||
      Arg == "1") {
    Value = BOU_TRUE;
  } else if (Arg == "false" || Arg == "FALSE"
             || Arg == "False" || Arg == "0") {
    Value = BOU_FALSE;
  } else {
    return O.error(": '" + Arg +
                   "' is invalid value for boolean argument! Try 0 or 1");
  }
  return false;
}

// parser<int> implementation
//
bool parser<int>::parse(Option &O, const char *ArgName,
                        const std::string &Arg, int &Value) {
  char *End;
  Value = (int)strtol(Arg.c_str(), &End, 0);
  if (*End != 0)
    return O.error(": '" + Arg + "' value invalid for integer argument!");
  return false;
}

// parser<unsigned> implementation
//
bool parser<unsigned>::parse(Option &O, const char *ArgName,
                             const std::string &Arg, unsigned &Value) {
  char *End;
  errno = 0;
  unsigned long V = strtoul(Arg.c_str(), &End, 0);
  Value = (unsigned)V;
  if (((V == ULONG_MAX) && (errno == ERANGE))
      || (*End != 0)
      || (Value != V))
    return O.error(": '" + Arg + "' value invalid for uint argument!");
  return false;
}

// parser<double>/parser<float> implementation
//
static bool parseDouble(Option &O, const std::string &Arg, double &Value) {
  const char *ArgStart = Arg.c_str();
  char *End;
  Value = strtod(ArgStart, &End);
  if (*End != 0)
    return O.error(": '" +Arg+ "' value invalid for floating point argument!");
  return false;
}

bool parser<double>::parse(Option &O, const char *AN,
                           const std::string &Arg, double &Val) {
  return parseDouble(O, Arg, Val);
}

bool parser<float>::parse(Option &O, const char *AN,
                          const std::string &Arg, float &Val) {
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
  unsigned i = 0, e = getNumOptions();
  std::string N(Name);

  while (i != e)
    if (getOption(i) == N)
      return i;
    else
      ++i;
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
    cout << "  -" << O.ArgStr << std::string(GlobalWidth-L-6, ' ')
         << " - " << O.HelpStr << "\n";

    for (unsigned i = 0, e = getNumOptions(); i != e; ++i) {
      size_t NumSpaces = GlobalWidth-strlen(getOption(i))-8;
      cout << "    =" << getOption(i) << std::string(NumSpaces, ' ')
           << " -   " << getDescription(i) << "\n";
    }
  } else {
    if (O.HelpStr[0])
      cout << "  " << O.HelpStr << "\n";
    for (unsigned i = 0, e = getNumOptions(); i != e; ++i) {
      size_t L = std::strlen(getOption(i));
      cout << "    -" << getOption(i) << std::string(GlobalWidth-L-8, ' ')
           << " - " << getDescription(i) << "\n";
    }
  }
}


//===----------------------------------------------------------------------===//
// --help and --help-hidden option implementation
//

namespace {

class HelpPrinter {
  size_t MaxArgLen;
  const Option *EmptyArg;
  const bool ShowHidden;

  // isHidden/isReallyHidden - Predicates to be used to filter down arg lists.
  inline static bool isHidden(std::pair<std::string, Option *> &OptPair) {
    return OptPair.second->getOptionHiddenFlag() >= Hidden;
  }
  inline static bool isReallyHidden(std::pair<std::string, Option *> &OptPair) {
    return OptPair.second->getOptionHiddenFlag() == ReallyHidden;
  }

public:
  explicit HelpPrinter(bool showHidden) : ShowHidden(showHidden) {
    EmptyArg = 0;
  }

  void operator=(bool Value) {
    if (Value == false) return;

    // Get all the options.
    std::vector<Option*> PositionalOpts;
    std::vector<Option*> SinkOpts;
    std::map<std::string, Option*> OptMap;
    GetOptionInfo(PositionalOpts, SinkOpts, OptMap);

    // Copy Options into a vector so we can sort them as we like...
    std::vector<std::pair<std::string, Option*> > Opts;
    copy(OptMap.begin(), OptMap.end(), std::back_inserter(Opts));

    // Eliminate Hidden or ReallyHidden arguments, depending on ShowHidden
    Opts.erase(std::remove_if(Opts.begin(), Opts.end(),
                          std::ptr_fun(ShowHidden ? isReallyHidden : isHidden)),
               Opts.end());

    // Eliminate duplicate entries in table (from enum flags options, f.e.)
    {  // Give OptionSet a scope
      std::set<Option*> OptionSet;
      for (unsigned i = 0; i != Opts.size(); ++i)
        if (OptionSet.count(Opts[i].second) == 0)
          OptionSet.insert(Opts[i].second);   // Add new entry to set
        else
          Opts.erase(Opts.begin()+i--);    // Erase duplicate
    }

    if (ProgramOverview)
      cout << "OVERVIEW: " << ProgramOverview << "\n";

    cout << "USAGE: " << ProgramName << " [options]";

    // Print out the positional options.
    Option *CAOpt = 0;   // The cl::ConsumeAfter option, if it exists...
    if (!PositionalOpts.empty() &&
        PositionalOpts[0]->getNumOccurrencesFlag() == ConsumeAfter)
      CAOpt = PositionalOpts[0];

    for (size_t i = CAOpt != 0, e = PositionalOpts.size(); i != e; ++i) {
      if (PositionalOpts[i]->ArgStr[0])
        cout << " --" << PositionalOpts[i]->ArgStr;
      cout << " " << PositionalOpts[i]->HelpStr;
    }

    // Print the consume after option info if it exists...
    if (CAOpt) cout << " " << CAOpt->HelpStr;

    cout << "\n\n";

    // Compute the maximum argument length...
    MaxArgLen = 0;
    for (size_t i = 0, e = Opts.size(); i != e; ++i)
      MaxArgLen = std::max(MaxArgLen, Opts[i].second->getOptionWidth());

    cout << "OPTIONS:\n";
    for (size_t i = 0, e = Opts.size(); i != e; ++i)
      Opts[i].second->printOptionInfo(MaxArgLen);

    // Print any extra help the user has declared.
    for (std::vector<const char *>::iterator I = MoreHelp->begin(),
          E = MoreHelp->end(); I != E; ++I)
      cout << *I;
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
HOp("help", cl::desc("Display available options (--help-hidden for more)"),
    cl::location(NormalPrinter), cl::ValueDisallowed);

static cl::opt<HelpPrinter, true, parser<bool> >
HHOp("help-hidden", cl::desc("Display all available options"),
     cl::location(HiddenPrinter), cl::Hidden, cl::ValueDisallowed);

static void (*OverrideVersionPrinter)() = 0;

namespace {
class VersionPrinter {
public:
  void print() {
        cout << "Low Level Virtual Machine (http://llvm.org/):\n";
        cout << "  " << PACKAGE_NAME << " version " << PACKAGE_VERSION;
#ifdef LLVM_VERSION_INFO
        cout << LLVM_VERSION_INFO;
#endif
        cout << "\n  ";
#ifndef __OPTIMIZE__
        cout << "DEBUG build";
#else
        cout << "Optimized build";
#endif
#ifndef NDEBUG
        cout << " with assertions";
#endif
        cout << ".\n";
        cout << "  Built " << __DATE__ << "(" << __TIME__ << ").\n";
  }
  void operator=(bool OptionWasSpecified) {
    if (OptionWasSpecified) {
      if (OverrideVersionPrinter == 0) {
        print();
        exit(1);
      } else {
        (*OverrideVersionPrinter)();
        exit(1);
      }
    }
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
  // --help option was given or not. Since we're circumventing that we have
  // to make it look like --help was given, so we assign true.
  NormalPrinter = true;
}

/// Utility function for printing version number.
void cl::PrintVersionMessage() {
  VersionPrinterInstance.print();
}

void cl::SetVersionPrinter(void (*func)()) {
  OverrideVersionPrinter = func;
}
