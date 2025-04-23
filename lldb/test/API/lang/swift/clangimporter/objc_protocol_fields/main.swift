import Foundation

func foo(p: NSCopying) {
  print("break here")
}

let s : NSString = "hello"
foo(p: s)

