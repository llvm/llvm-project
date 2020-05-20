import Foundation

func main<T>(_ x: T) {
  let s = NSString("This is necessary to actually pull in the Foundation dependency")
  var k = NSString.self
  print("Set breakpoint here")
}

main() { (x:Int) -> Int in return x }
