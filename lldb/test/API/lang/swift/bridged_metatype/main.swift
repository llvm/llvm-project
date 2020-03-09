import Foundation

func main<T>(_ x: T) {
  var k = NSString.self
  print("Set breakpoint here")
}

main() { (x:Int) -> Int in return x }
