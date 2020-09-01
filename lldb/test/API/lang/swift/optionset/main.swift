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
import Foundation

struct Options : OptionSet {
  let rawValue: Int
}

struct ComputedOptions : OptionSet {
  init(rawValue: Int) { storedValue = rawValue }
  let storedValue: Int
  var rawValue: Int {
    get { return storedValue }
  }
}

func use<T>(_ t: T) {}

func main() {
  var user_option = Options(rawValue: 123456)
  var computed_option = ComputedOptions(rawValue: 789)
  var sdk_option_single_valued: NSBinarySearchingOptions = .insertionIndex
  var sdk_option_exhaustive: NSBinarySearchingOptions = [.firstEqual, .insertionIndex]
  var sdk_option_nonexhaustive = NSBinarySearchingOptions(rawValue: 257)
  var sdk_option_nonevalid = NSBinarySearchingOptions(rawValue: 12)
  use((user_option, computed_option, // break here
      sdk_option_single_valued,
      sdk_option_exhaustive, sdk_option_nonexhaustive,
      sdk_option_nonevalid))
  //%self.expect('frame variable user_option', substrs=['rawValue = 123456'])
  //%self.expect('frame variable computed_option', substrs=['storedValue', '789'])
  //%self.expect('expression user_option', substrs=['rawValue = 123456'])
  //%self.expect('frame variable sdk_option_single_valued', substrs=['.insertionIndex'])
  //%self.expect('frame variable sdk_option_single_valued', matching=False, substrs=['['])
  //%self.expect('expression sdk_option_single_valued', substrs=['.insertionIndex'])
  //%self.expect('expression sdk_option_single_valued', matching=False, substrs=['['])
  //%self.expect('frame variable sdk_option_exhaustive', substrs=['[.firstEqual, .insertionIndex]'])
  //%self.expect('expression sdk_option_exhaustive', substrs=['[.firstEqual, .insertionIndex]'])
  //%self.expect('frame variable sdk_option_nonexhaustive', substrs=['[.firstEqual, 0x1]'])
  //%self.expect('expression sdk_option_nonexhaustive', substrs=['[.firstEqual, 0x1]'])
  //%self.expect('frame variable sdk_option_nonevalid', substrs=['rawValue = 12'])
  //%self.expect('expression sdk_option_nonevalid', substrs=['rawValue = 12'])
}

main()
