// RUN: not llvm-tblgen -gen-opt-parser-defs -I %p/../../include %s -DUNKNOWN 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: not llvm-tblgen -gen-opt-parser-defs -I %p/../../include %s -DCONDITION 2>&1 | FileCheck %s --check-prefix=CONDITION

include "llvm/Option/LibraryOptions.td"

def TestOptions : OptionsStruct;

defm : BoolField<"foo", "false", "">;
def Never : BuildCondition<"0">;
let Condition = Never in
defm : BoolField<"bar", "false", "">;

#ifdef UNKNOWN
// UNKNOWN: [[#@LINE+1]]:1: error: '-baz' is not an option of the OptionsStruct
def : AliasOf<"old-baz", "baz">;
#endif

#ifdef CONDITION
// CONDITION: [[#@LINE+1]]:1: error: an alias cannot refer to an option with a Condition
def : AliasOf<"disable-bar", "bar=false">;
#endif
