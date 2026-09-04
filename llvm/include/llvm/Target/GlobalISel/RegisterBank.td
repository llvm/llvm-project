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

class RegisterBank<string name, list<RegisterClass> classes> {
  string Name = name;
  list<RegisterClass> RegisterClasses = classes;

  /// Extra partial mappings
  /// the length of elements in list ExtraPartMappings
  /// should <= 2. The first element in list<int> is length
  /// the second, if have, will be the offset.
  /// 
  /// let ExtraPartMappings = [[1], [32, 16]] will add a partial mapping
  /// with length 1 at offset 0 and a partial mapping
  /// with length 32 at offset 16.
  list<list<int>> ExtraPartMappings = [];
}
