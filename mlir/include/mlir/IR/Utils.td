//===-- Utils.td - General utilities file ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a number of utilities which can be used across tablegen
// files.
//
//===----------------------------------------------------------------------===//

#ifndef UTILS_TD
#define UTILS_TD

// Helper for marking deprecated classes or defs in TableGen. To mark a def as
// deprecated, mix in the `Deprecate` class with a reason.
// Usage of a deprecated def within TableGen will cause a warning with the
// given message.
class Deprecated<string reason> {
  string odsDeprecated = reason;
}

// Helper for marking entities in ODS generated C++ as deprecated.
// Usage of such an entity from C++ code will cause a warning being emitted by
// the C++ compiler with the given message.
//
// Note: Support has to be implemented by the code generator of a given
// entity.
class CppDeprecated<string reason> {
  string odsCppDeprecated = reason;
}

// A workaround for the inability to define functions in Tablegen.
//
// The template parameter defines a string that can be extracted from an
// instance of this class by accessing the "result" member. Subclasses can take
// their own template parameters as function "arguments" and use them to
// populate result.
// For example, if it didn't already exist, a concat function could be defined
// like:
//
// class StrConcat<list<string> strings> :
//     StrFunc<!foldl("", strings, prev, cur, prev # cur)>
//
// and then called like
//
// StrConcat<["a", "b", "c"]>.result
//
// to get the string "abc"
class StrFunc<string r> {
  string result = r;
}

// Marker used to identify the argument list.
def ins;

// Marker used to identify the result list.
def outs;

// This class represents a typed argument with optional default value for C
// function signatures, e.g. builders or methods.
class CArg<string ty, string value = ""> {
  string type = ty;
  string defaultValue = value;
}

// Helper which makes the first letter of a string uppercase.
// e.g. cat -> Cat
class firstCharToUpper<string str>
{
  string ret = !if(!gt(!size(str), 0),
    !toupper(!substr(str, 0, 1)) # !substr(str, 1),
    "");
}

class _snakeCaseHeper<string str> {
  int idx = !find(str, "_");
  string ret = !if(!ge(idx, 0),
    !substr(str, 0, idx) # firstCharToUpper<!substr(str, !add(idx, 1))>.ret,
    str);
}

// Converts a snake_case string to CamelCase.
// TODO: Replace with a !tocamelcase bang operator.
class snakeCaseToCamelCase<string str>
{
  string ret = !foldl(firstCharToUpper<str>.ret,
    !range(0, !size(str)), acc, idx, _snakeCaseHeper<acc>.ret);
}

#endif // UTILS_TD
