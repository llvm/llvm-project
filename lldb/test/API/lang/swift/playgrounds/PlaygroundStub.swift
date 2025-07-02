// PlaygroundStub.swift
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

@_silgen_name ("playground_logger_initialize") func builtin_initialize();

builtin_initialize();

func break_here() {
  print("This is the frame where the playground expression will be executed.")
}

break_here()
