#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::opt;

namespace {
enum ID {
  OPT_INVALID = 0,
#define OPTION(PREFIXES, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS,       \
               VISIBILITY, PARAM, HELPTEXT, HELPTEXTSFORVARIANTS, METAVAR,     \
               VALUES, COMMANDIDS_OFFSET)                                      \
  OPT_##ID,
#include "Opts.inc"
#undef OPTION
};
#define OPTTABLE_STR_TABLE_CODE
#include "Opts.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "Opts.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

#define OPTTABLE_COMMAND_IDS_TABLE_CODE
#include "Opts.inc"
#undef OPTTABLE_COMMAND_IDS_TABLE_CODE

#define OPTTABLE_COMMANDS_CODE
#include "Opts.inc"
#undef OPTTABLE_COMMANDS_CODE

static constexpr OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

class HelloSubOptTable : public GenericOptTable {
public:
  HelloSubOptTable()
      : GenericOptTable(OptionStrTable, OptionPrefixesTable, InfoTable,
                        OptionCommands, OptionCommandIDsTable) {}
};
} // namespace

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  HelloSubOptTable T;
  unsigned MissingArgIndex, MissingArgCount;

  auto HandleMultipleSubcommands = [](const ArrayRef<StringRef> SubCommands) {
    assert(SubCommands.size() > 1);
    llvm::errs() << "error: more than one subcommand passed [\n";
    for (auto SC : SubCommands) {
      llvm::errs() << " `" << SC << "`\n";
    }
    llvm::errs() << "]\n";
    llvm::errs() << "See --help.\n";
    exit(1);
  };

  auto HandleOtherPositionals = [](const ArrayRef<StringRef> Positionals) {
    assert(!Positionals.empty());
    llvm::errs() << "error: unknown positional argument(s) [\n";
    for (auto SC : Positionals) {
      llvm::errs() << " `" << SC << "`\n";
    }
    llvm::errs() << "]\n";
    llvm::errs() << "See --help.\n";
    exit(1);
  };

  InputArgList Args = T.ParseArgs(ArrayRef(argv + 1, argc - 1), MissingArgIndex,
                                  MissingArgCount);

  StringRef Subcommand = Args.getSubcommand(
      T.getCommands(), HandleMultipleSubcommands, HandleOtherPositionals);
  // Handle help. When help options is found, ignore all other options and exit
  // after printing help.
  if (Args.hasArg(OPT_help)) {
    T.printHelp(llvm::outs(), "llvm-hello-sub [subcommand] [options]",
                "LLVM Hello Subcommand Example", false, false, Visibility(),
                Subcommand);
    return 0;
  }
  bool HasUnknownOptions = false;
  for (const Arg *A : Args.filtered(OPT_UNKNOWN)) {
    HasUnknownOptions = true;
    llvm::errs() << "Unknown option `" << A->getAsString(Args) << "'\n";
  }
  if (HasUnknownOptions) {
    llvm::errs() << "See `OptSubcommand --help`.\n";
    return 1;
  }
  if (Subcommand.empty()) {
    if (Args.hasArg(OPT_version)) {
      llvm::outs() << "LLVM Hello Subcommand Example 1.0\n";
    }
  } else if (Subcommand == "foo") {
    if (Args.hasArg(OPT_uppercase))
      llvm::outs() << "FOO\n";
    else if (Args.hasArg(OPT_lowercase))
      llvm::outs() << "foo\n";

    if (Args.hasArg(OPT_version))
      llvm::outs() << "LLVM Hello Subcommand foo Example 1.0\n";

  } else if (Subcommand == "bar") {
    if (Args.hasArg(OPT_lowercase))
      llvm::outs() << "bar\n";
    else if (Args.hasArg(OPT_uppercase))
      llvm::outs() << "BAR\n";

    if (Args.hasArg(OPT_version))
      llvm::outs() << "LLVM Hello Subcommand bar Example 1.0\n";
  }

  return 0;
}
