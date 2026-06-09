// RUN: llvm-tblgen -gen-opt-parser-defs -I %p/../../include %s | FileCheck %s

include "llvm/Option/OptParser.td"

def ToolVis : OptionVisibility;
def sub : SubCommand<"sub", "Subcommand help", "tool sub [options]">;

def target : Joined<["--"], "target=", [sub]>,
             HelpText<"Select the target */ safely">,
             HelpTextForVariants<[ToolVis], "Select the tool target">,
             MetaVarName<"<triple>">,
             Values<"alpha,beta">;
def target_alias : Flag<["--"], "target-host">,
                   Alias<target>,
                   AliasArgs<["host", "default"]>;
def computed : Joined<["--"], "computed=">, ValuesCode<[{
  static constexpr char VALUES_CODE[] = "gamma,delta";
}]>;

// CHECK-LABEL: #ifdef OPTTABLE_STR_TABLE_CODE
// CHECK: static constexpr char OptionStrTable[] =
// CHECK-NOT: llvm::StringTable

// CHECK-LABEL: #ifdef OPTTABLE_OPTION_STR_TABLE_CODE
// CHECK: static constexpr char OptionNameStrTable[] =
// CHECK-NOT: Select the target

// CHECK-LABEL: #ifdef OPTTABLE_PREFIXES_TABLE_CODE
// CHECK: static constexpr llvm::opt::OptTable::HelpTextVariant OptionHelpTextVariants[] = {
// CHECK: {[[#]] /* Select the tool target */, ToolVis}, // target

// CHECK-LABEL: #ifdef OPTTABLE_VALUES_CODE
// CHECK: static const char *getGeneratedOptionValues(unsigned Index) {
// CHECK: case 0: return computed_Values;

// CHECK-LABEL: // Options
// CHECK: OPTION(1, 22 /* --computed= */, computed, Joined, INVALID, INVALID, 0, 0, DefaultVis, 0, 0, 0, 0, llvm::opt::OptTable::ValuesCodeFlag | 0, 0)
// CHECK: OPTION(1, 34 /* --target-host */, target_alias, Flag, INVALID, target, {{[0-9]+}}, 0, DefaultVis, 0, 0, 0, 0, 0, 0)
// CHECK: OPTION(1, 48 /* --target= */, target, Joined, INVALID, INVALID, 0, 0, DefaultVis, 0,
// CHECK-NEXT: {{[0-9]+}} /* Select the target * / safely */, 1, {{[0-9]+}} /* <triple> */, {{[0-9]+}} /* alpha,beta */, 1)

// CHECK-LABEL: #ifdef OPTTABLE_SUBCOMMANDS_CODE
// CHECK: static constexpr llvm::opt::OptTable::SubCommand OptionSubCommands[] = {
// CHECK-NEXT: { {{[0-9]+}} /* sub */, {{[0-9]+}} /* Subcommand help */, {{[0-9]+}} /* tool sub [options] */ },
