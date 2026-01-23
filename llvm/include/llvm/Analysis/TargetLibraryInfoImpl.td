//===-- TargetLibraryInfoImpl.td - File that describes library functions --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Return type or argument type of library function.
class FuncArgType {
  string Name = NAME;
}

// Recognized types of library function arguments and return types.
def Void : FuncArgType;
def Bool : FuncArgType;     // 8 bits on all targets
def Int16 : FuncArgType;
def Int32 : FuncArgType;
def Int : FuncArgType;
def IntPlus : FuncArgType;  // Int or bigger.
def Long : FuncArgType;     // Either 32 or 64 bits.
def IntX : FuncArgType;     // Any integer type.
def Int64 : FuncArgType;
def LLong : FuncArgType;    // 64 bits on all targets.
def SizeT : FuncArgType;    // size_t.
def SSizeT : FuncArgType;   // POSIX ssize_t.
def Flt : FuncArgType;      // IEEE float.
def Dbl : FuncArgType;      // IEEE double.
def LDbl : FuncArgType;     // Any floating type (TODO: tighten this up).
def Floating : FuncArgType; // Any floating type.
def Ptr : FuncArgType;      // Any pointer type.
def Struct : FuncArgType;   // Any struct type.
def Ellip : FuncArgType;    // The ellipsis (...).
def Same : FuncArgType;     // Same argument type as the previous one.


// Definition of library function.
class TargetLibCall<string Str, FuncArgType RetType,
                    list<FuncArgType> ArgTypes = []> {
  string Name = NAME;
  string String = Str;
  FuncArgType ReturnType = RetType;
  list<FuncArgType> ArgumentTypes = ArgTypes;
}
