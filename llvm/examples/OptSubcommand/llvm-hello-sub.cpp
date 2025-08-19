#include "llvm/ADT/ArrayRef.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
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
  InputArgList Args = T.ParseArgs(ArrayRef(argv + 1, argc - 1), MissingArgIndex,
                                  MissingArgCount);

  StringRef Subcommand = Args.getSubcommand();
  if (Args.hasArg(OPT_help)) {
    T.printHelp(llvm::outs(), "llvm-hello-sub [subcommand] [options]",
                "LLVM Hello Subcommand Example", false, false, Visibility(),
                Subcommand);
    return 0;
  }

  if (Args.hasArg(OPT_version)) {
    llvm::outs() << "LLVM Hello Subcommand Example 1.0\n";
    return 0;
  }

  if (Subcommand == "foo") {
    if (Args.hasArg(OPT_uppercase))
      llvm::outs() << "FOO\n";
    else if (Args.hasArg(OPT_lowercase))
      llvm::outs() << "foo\n";
    else
      llvm::errs() << "error: unknown option for subcommand '" << Subcommand
                   << "'. See -help.\n";
    return 1;
  } else if (Subcommand == "bar") {
    if (Args.hasArg(OPT_lowercase))
      llvm::outs() << "bar\n";
    else if (Args.hasArg(OPT_uppercase))
      llvm::outs() << "BAR\n";
    else
      llvm::errs() << "error: unknown option for subcommand '" << Subcommand
                   << "'. See -help.\n";
  } else {
    llvm::errs() << "error: unknown subcommand '" << Subcommand
                 << "'. See --help.\n";
    return 1;
  }

  return 0;
}
