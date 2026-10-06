//===- RegisterBank.td - Register bank definitions ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

/// Same as RegisterBankInfo::PartialMapping
class PartialMapping<int Len, int startIdx = 0> {
  int Length = Len;
  int StartIdx = startIdx;
}

class RegisterBank<string name, list<RegisterClass> classes> {
  string Name = name;
  list<RegisterClass> RegisterClasses = classes;
  
  // If some partial mappings can't be inferred from register class
  // add them to this list.
  list<PartialMapping> ExtraPartialMappings = [];

  // In the foreseeable future, we still need to maintain
  // value mappings manually, use this field to filter out unwanted
  // partial mappings.
  list<PartialMapping> IgnoredPartialMappings = [];
}
