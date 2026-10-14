// RUN: rm -rf %t && split-file %s %t
// RUN: llvm-tblgen -gen-opt-parser-defs -I %S/../../include %t/ok.td | FileCheck %s
// RUN: not llvm-tblgen -gen-opt-parser-defs -I %S/../../include %t/kind.td 2>&1 | FileCheck %s --check-prefix=KIND
// RUN: not llvm-tblgen -gen-opt-parser-defs -I %S/../../include %t/flag-sep.td 2>&1 | FileCheck %s --check-prefix=FLAGSEP
// RUN: not llvm-tblgen -gen-opt-parser-defs -I %S/../../include %t/args.td 2>&1 | FileCheck %s --check-prefix=ARGS
// RUN: not llvm-tblgen -gen-opt-parser-defs -I %S/../../include %t/chain.td 2>&1 | FileCheck %s --check-prefix=CHAIN
// RUN: not llvm-tblgen -gen-opt-parser-defs -I %S/../../include %t/dup.td 2>&1 | FileCheck %s --check-prefix=DUP

// An alias is built under its target's Condition.

//--- ok.td
include "llvm/Option/LibraryOptions.td"
def S : OptionsStruct<"ns">;
def Cond : BuildCondition<"COND">;
let Condition = Cond in
defm foo : BoolField<"foo", "true", "Foo">;
def : FlagOrEq<["-"], "old-foo">, Alias<foo>;
def : Flag<["-"], "disable-foo">, Alias<foo>, AliasArgs<["false"]>;
defm bar : ValueField<"bar", "int", "0", "Bar">;
def : Flag<["-"], "bar-one">, Alias<bar>, AliasArgs<["1"]>;

// CHECK:      OPT_bar_one,
// CHECK-NEXT: OPT_bar,
// CHECK-NEXT: #if COND
// CHECK-NEXT:   OPT_disable_foo,
// CHECK-NEXT: #endif
// CHECK-NEXT: #if COND
// CHECK-NEXT:   OPT_foo,
// CHECK-NEXT: #endif
// CHECK-NEXT: #if COND
// CHECK-NEXT:   OPT_old_foo,
// CHECK-NEXT: #endif

// CHECK:      -bar-one */, 0, 0, OPT_bar,
// CHECK:      #if COND
// CHECK-NEXT:   -disable-foo */, 0, 0, OPT_foo,
// CHECK-NEXT: #endif
// CHECK:      -old-foo */, 0, 0, OPT_foo,
// CHECK-NEXT: #endif

// Only the target is a case of apply().
// CHECK:      case OPT_bar:
// CHECK:      #if COND
// CHECK-NEXT:   case OPT_foo:
// CHECK-NOT:  case OPT_

//--- kind.td
include "llvm/Option/LibraryOptions.td"
def S : OptionsStruct<"ns">;
defm foo : ValueField<"foo", "int", "0", "Foo">;
def : FlagOrEq<["-"], "old-foo">, Alias<foo>;
// KIND: kind.td:{{[0-9]+}}:1: error: a FlagOrEq alias must have a FlagOrEq target, not SeparateOrEq

//--- flag-sep.td
include "llvm/Option/LibraryOptions.td"
def S : OptionsStruct<"ns">;
defm foo : ValueField<"foo", "int", "0", "Foo">;
def : Flag<["-"], "old-foo">, Alias<foo>;
// FLAGSEP: flag-sep.td:{{[0-9]+}}:1: error: a Flag alias of a SeparateOrEq option needs an AliasArg to supply the value

//--- args.td
include "llvm/Option/LibraryOptions.td"
def S : OptionsStruct<"ns">;
defm foo : BoolField<"foo", "false", "Foo">;
def : FlagOrEq<["-"], "old-foo">, Alias<foo>, AliasArgs<["true"]>;
// ARGS: args.td:{{[0-9]+}}:1: error: only a Flag alias can have AliasArgs

//--- chain.td
include "llvm/Option/LibraryOptions.td"
def S : OptionsStruct<"ns">;
defm foo : BoolField<"foo", "false", "Foo">;
def older : FlagOrEq<["-"], "old-foo">, Alias<foo>;
def : FlagOrEq<["-"], "older-foo">, Alias<older>;
// CHAIN: chain.td:{{[0-9]+}}:1: error: the target of an alias must be declared with BoolField or ValueField, not be an alias

//--- dup.td
include "llvm/Option/LibraryOptions.td"
def S : OptionsStruct<"ns">;
defm foo : BoolField<"foo", "false", "Foo">;
def : FlagOrEq<["-"], "foo">, Alias<foo>;
// DUP: error: option '-foo' is declared more than once
