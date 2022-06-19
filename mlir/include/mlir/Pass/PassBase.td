//===-- PassBase.td - Base pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for defining pass registration and other
// mechanisms.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PASS_PASSBASE
#define MLIR_PASS_PASSBASE

//===----------------------------------------------------------------------===//
// Options
//===----------------------------------------------------------------------===//

class Option<string varName, string arg, string valueType, string default,
             string desc, string additionalFlags = ""> {
  // The name for the C++ option variable.
  string cppName = varName;

  // The command line argument to use for this option.
  string argument = arg;

  // The C++ type of the option.
  string type = valueType;

  // The default value of the option. "" corresponds to no default.
  string defaultValue = default;

  // A description for this option.
  string description = desc;

  // A set of additional flags to pass along to the option constructor.
  string additionalOptFlags = additionalFlags;
}

class ListOption<string varName, string arg, string valueType,
                 string desc, string additionalFlags = "">
  : Option<varName, arg, valueType, /*default=*/"", desc, additionalFlags> {}

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

class Statistic<string varName, string statName, string desc> {
  // The C++ variable name for the statistic.
  string cppName = varName;

  // The displayed name of the statistic, similar to the argument of an option.
  string name = statName;

  // The description of the statistic.
  string description = desc;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class PassBase<string passArg, string base> {
  // The command line argument of the pass.
  string argument = passArg;

  // The C++ base class for the pass.
  string baseClass = base;

  // A short 1-line summary of the pass.
  string summary = "";

  // A human readable description of the pass.
  string description = "";

  // A C++ constructor call to create an instance of this pass.
  code constructor = [{}];

  // A list of dialects this pass may produce entities in.
  list<string> dependentDialects = [];

  // A set of options provided by this pass.
  list<Option> options = [];

  // A set of statistics provided by this pass.
  list<Statistic> statistics = [];
}

// This class represents an mlir::OperationPass.
class Pass<string passArg, string operation = "">
  : PassBase<passArg, "::mlir::OperationPass<" # operation # ">">;

// This class represents an mlir::InterfacePass.
class InterfacePass<string passArg, string interface>
  : PassBase<passArg, "::mlir::InterfacePass<" # interface # ">">;

#endif // MLIR_PASS_PASSBASE
