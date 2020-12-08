//===--- OptParser.td - Common Option Parsing Interfaces ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the common interfaces used by the option parsing TableGen
//  backend.
//
//===----------------------------------------------------------------------===//

// Define the kinds of options.

class OptionKind<string name, int precedence = 0, bit sentinel = false> {
  string Name = name;
  // The kind precedence, kinds with lower precedence are matched first.
  int Precedence = precedence;
  // Indicate a sentinel option.
  bit Sentinel = sentinel;
}

// An option group.
def KIND_GROUP : OptionKind<"Group">;
// The input option kind.
def KIND_INPUT : OptionKind<"Input", 1, true>;
// The unknown option kind.
def KIND_UNKNOWN : OptionKind<"Unknown", 2, true>;
// A flag with no values.
def KIND_FLAG : OptionKind<"Flag">;
// An option which prefixes its (single) value.
def KIND_JOINED : OptionKind<"Joined", 1>;
// An option which is followed by its value.
def KIND_SEPARATE : OptionKind<"Separate">;
// An option followed by its values, which are separated by commas.
def KIND_COMMAJOINED : OptionKind<"CommaJoined">;
// An option which is which takes multiple (separate) arguments.
def KIND_MULTIARG : OptionKind<"MultiArg">;
// An option which is either joined to its (non-empty) value, or followed by its
// value.
def KIND_JOINED_OR_SEPARATE : OptionKind<"JoinedOrSeparate">;
// An option which is both joined to its (first) value, and followed by its
// (second) value.
def KIND_JOINED_AND_SEPARATE : OptionKind<"JoinedAndSeparate">;
// An option which consumes all remaining arguments if there are any.
def KIND_REMAINING_ARGS : OptionKind<"RemainingArgs">;
// An option which consumes an optional joined argument and any other remaining
// arguments.
def KIND_REMAINING_ARGS_JOINED : OptionKind<"RemainingArgsJoined">;

// Define the option flags.

class OptionFlag {}

// HelpHidden - The option should not be displayed in --help, even if it has
// help text. Clients *can* use this in conjunction with the OptTable::PrintHelp
// arguments to implement hidden help groups.
def HelpHidden : OptionFlag;

// RenderAsInput - The option should not render the name when rendered as an
// input (i.e., the option is rendered as values).
def RenderAsInput : OptionFlag;

// RenderJoined - The option should be rendered joined, even if separate (only
// sensible on single value separate options).
def RenderJoined : OptionFlag;

// RenderSeparate - The option should be rendered separately, even if joined
// (only sensible on joined options).
def RenderSeparate : OptionFlag;

// Define the option group class.

class OptionGroup<string name> {
  string EnumName = ?; // Uses the def name if undefined.
  string Name = name;
  string HelpText = ?;
  OptionGroup Group = ?;
  list<OptionFlag> Flags = [];
}

// Define the option class.

class Option<list<string> prefixes, string name, OptionKind kind> {
  string EnumName = ?; // Uses the def name if undefined.
  list<string> Prefixes = prefixes;
  string Name = name;
  OptionKind Kind = kind;
  // Used by MultiArg option kind.
  int NumArgs = 0;
  string HelpText = ?;
  string MetaVarName = ?;
  string Values = ?;
  code ValuesCode = ?;
  list<OptionFlag> Flags = [];
  OptionGroup Group = ?;
  Option Alias = ?;
  list<string> AliasArgs = [];
  code KeyPath = ?;
  code DefaultValue = ?;
  code ImpliedValue = ?;
  code ImpliedCheck = "false";
  bit ShouldAlwaysEmit = false;
  code NormalizerRetTy = ?;
  code NormalizedValuesScope = "";
  code Normalizer = "";
  code Denormalizer = "";
  code ValueMerger = "mergeForwardValue";
  code ValueExtractor = "extractForwardValue";
  list<code> NormalizedValues = ?;
}

// Helpers for defining options.

class Flag<list<string> prefixes, string name>
  : Option<prefixes, name, KIND_FLAG>;
class Joined<list<string> prefixes, string name>
  : Option<prefixes, name, KIND_JOINED>;
class Separate<list<string> prefixes, string name>
  : Option<prefixes, name, KIND_SEPARATE>;
class CommaJoined<list<string> prefixes, string name>
  : Option<prefixes, name, KIND_COMMAJOINED>;
class MultiArg<list<string> prefixes, string name, int numargs>
  : Option<prefixes, name, KIND_MULTIARG> {
  int NumArgs = numargs;
}
class JoinedOrSeparate<list<string> prefixes, string name>
  : Option<prefixes, name, KIND_JOINED_OR_SEPARATE>;
class JoinedAndSeparate<list<string> prefixes, string name>
  : Option<prefixes, name, KIND_JOINED_AND_SEPARATE>;

// Mix-ins for adding optional attributes.

class Alias<Option alias> { Option Alias = alias; }
class AliasArgs<list<string> aliasargs> { list<string> AliasArgs = aliasargs; }
class EnumName<string name> { string EnumName = name; }
class Flags<list<OptionFlag> flags> { list<OptionFlag> Flags = flags; }
class Group<OptionGroup group> { OptionGroup Group = group; }
class HelpText<string text> { string HelpText = text; }
class MetaVarName<string name> { string MetaVarName = name; }
class Values<string value> { string Values = value; }
class ValuesCode<code valuecode> { code ValuesCode = valuecode; }

// Helpers for defining marshalling information.

class ImpliedByAnyOf<list<Option> options, code value = "true"> {
  code ImpliedCheck = !foldl("false", options, accumulator, option,
                             !strconcat(accumulator, " || ", !cast<string>(option.KeyPath)));
  code ImpliedValue = value;
}

class MarshallingInfo<code keypath, code defaultvalue> {
  code KeyPath = keypath;
  code DefaultValue = defaultvalue;
}

class MarshallingInfoString<code keypath, code defaultvalue>
  : MarshallingInfo<keypath, defaultvalue> {}

class MarshallingInfoFlag<code keypath, code defaultvalue = "false">
  : MarshallingInfo<keypath, defaultvalue> {
  code Normalizer = "normalizeSimpleFlag";
  code Denormalizer = "denormalizeSimpleFlag";
}

class MarshallingInfoBitfieldFlag<code keypath, code value>
  : MarshallingInfoFlag<keypath, "0u"> {
  code Normalizer = "makeFlagToValueNormalizer("#value#")";
  code ValueMerger = "mergeMaskValue";
  code ValueExtractor = "(extractMaskValue<unsigned, decltype("#value#"), "#value#">)";
}

class MarshallingInfoBooleanFlag<code keypath, code defaultvalue, code neg_name, string neg_spelling>
  : MarshallingInfoFlag<keypath, defaultvalue> {
  bit ShouldAlwaysEmit = 1;
  code Normalizer = "makeBooleanFlagNormalizer(OPT_"#neg_name#")";
  code Denormalizer = "makeBooleanFlagDenormalizer(\""#neg_spelling#"\")";
}

// Mixins for additional marshalling attributes.

class IsNegative {
  code Normalizer = "normalizeSimpleNegativeFlag";
}
class AlwaysEmit { bit ShouldAlwaysEmit = true; }
class Normalizer<code normalizer> { code Normalizer = normalizer; }
class Denormalizer<code denormalizer> { code Denormalizer = denormalizer; }
class NormalizedValuesScope<code scope> { code NormalizedValuesScope = scope; }
class NormalizedValues<list<code> definitions> { list<code> NormalizedValues = definitions; } 
class DenormalizeString { code Denormalizer = "denormalizeString"; }
class AutoNormalizeEnum {
  code Normalizer = "normalizeSimpleEnum";
  code Denormalizer = "denormalizeSimpleEnum";
}
class AutoNormalizeEnumJoined : AutoNormalizeEnum {
  code Denormalizer = "denormalizeSimpleEnumJoined";
}
class ValueMerger<code merger> { code ValueMerger = merger; }
class ValueExtractor<code extractor> { code ValueExtractor = extractor; }

// Predefined options.

// FIXME: Have generator validate that these appear in correct position (and
// aren't duplicated).
def INPUT : Option<[], "<input>", KIND_INPUT>;
def UNKNOWN : Option<[], "<unknown>", KIND_UNKNOWN>;
