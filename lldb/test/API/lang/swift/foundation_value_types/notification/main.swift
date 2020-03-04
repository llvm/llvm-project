// main.swift
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2018 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------
import Foundation

func main() {
  var notification = Notification(name: Notification.Name(rawValue: "MyNotification"), object: nil, userInfo: [:])
  print("done!") //% self.expect("frame variable notification", substrs=['name = "MyNotification"', 'object = nil', 'userInfo = 0 key/value pairs'])
   //% self.expect("expression -d run -- notification", substrs=['name = "MyNotification"', 'object = nil', 'userInfo = 0 key/value pairs'])
}

var g_notification = Notification(name: Notification.Name(rawValue: "MyNotification"), object: nil, userInfo: [:])

main() //% self.expect("target variable g_notification", substrs=['name = "MyNotification"', 'object = nil', 'userInfo = 0 key/value pairs'])
       //% self.expect("expression -d run -- g_notification", substrs=['name = "MyNotification"', 'object = nil', 'userInfo = 0 key/value pairs'])
