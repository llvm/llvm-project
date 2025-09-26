//===- SetTheory.td - DAG set operator declarations --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FIXME: This is not used everywhere, and different files declare
// different subsets of used operators.
//
// It just happens TargetSelectionDAG.td defines records with the same
// names as the tablegen DAG operators for SelectionDAG operators.

// Target.td separately declares the special set operators.

def add; // Forward declare
def sub;
def and;
def shl;
// def trunc; // FIXME: Name collision
def rotl;
def rotr;

def sequence;
def decimate;
def interleave;
