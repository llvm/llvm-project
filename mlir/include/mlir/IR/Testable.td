//===-- Testable.td - Testable type definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definition of the `Testable` type. 
//
// Any type whose records can have corresponding LIT tests (eg - Pass) can extend 
// `Testable` in order to be able to embed LIT tests within record definitions.
//
//===----------------------------------------------------------------------===//

#ifndef TESTABLE
#define TESTABLE

// Represents a LIT test record in TableGen
class LitTest<string name, code snippet, list<string> run = [], list<string> check = []> {
  // The name of the generated test file
  string testFileName = name;
  
  // The IR snippet/code to be tested
  code irSnippet = snippet;
  
  // The RUN commands for the test (e.g., "mlir-opt %s")
  list<string> runLines = run;
  
  // Expected output patterns (CHECK lines)
  list<string> checkLines = check;
}

// Base class for elements that can have auto-generated LIT tests
class Testable {
  // List of LIT tests associated with this element
  list<LitTest> tests = [];
}

#endif // TESTABLE