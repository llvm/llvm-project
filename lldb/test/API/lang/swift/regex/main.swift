// main.swift
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------
import RegexBuilder

let regex = /Order from <(.*)>, type: (.*), count in dozen: ([0-9]+)/

let dslRegex = Regex { .digit }

print(regex) // Set breakpoint here

