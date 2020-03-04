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
class POClass {
  var ivar = "Hello World"
}

func main() {
  var object: POClass
  object = POClass() //% self.assertTrue(self.frame().FindVariable('object').GetObjectDescription() == '<uninitialized>', 'po correctly detects uninitialized instances')
  print("yay I am done") //% self.assertFalse(self.frame().FindVariable('object').GetObjectDescription() == '<uninitialized>', 'po incorrectly detects uninitialized instances')
}

print("Some code here")
main()
print("Some code there")
