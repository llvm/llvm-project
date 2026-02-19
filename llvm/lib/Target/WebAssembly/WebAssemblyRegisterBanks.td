//=- WebAssemblyRegisterBank.td - Describe the Wasm Banks ----*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//


def I32RegBank : RegisterBank<"I32RegBank", [I32]>;
def I64RegBank : RegisterBank<"I64RegBank", [I64]>;
def F32RegBank : RegisterBank<"F32RegBank", [F32]>;
def F64RegBank : RegisterBank<"F64RegBank", [F64]>;

def V128RegBank : RegisterBank<"V128RegBank", [V128]>;

def EXTERNREFRegBank : RegisterBank<"EXTERNREFRegBank", [EXTERNREF]>;
def FUNCREFRegBank : RegisterBank<"FUNCREFRegBank", [FUNCREF]>;
def EXNREFRegBank : RegisterBank<"EXNREFRegBank", [EXNREF]>;
