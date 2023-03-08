public struct A {
  var i: Int
}
public struct B {
  var d: Double
}

let a = A(i: 23)
let b = B(d: 2.71)

public func f1<each T>(args: repeat each T) {
  print("break here")
}
 
f1(args: a, b)
 
public func f2<each U, each V>(us: repeat each U, vs: repeat each V) {
  print("break here")
}
 
f2(us: a, vs: b)
 
public func f3<each T>(ts: repeat each T, more_ts: repeat each T) {
  print("break here")
}
 
f3(ts: a, b, more_ts: a, b)

// FIXME: Crashes the compiler.
//public func f4<each U, each V>(uvs: repeat (each U, each V)) {
//  print("break here")
//}
// 
//f4(uvs: (a, b), (a, b))
 
public func f5<each T, U>(ts: repeat (each T, U)) {
  print("break here")
}
 
f5(ts: (a, b), (42, b))

// FIXME: Crashes the compiler. 
//public func f6<each U, each V>(us: repeat each U, more_us: repeat each U, vs: repeat each V) {
//  print("break here")
//}
// 
//f6(us: a, more_us: a, vs: b, b)
 
public func f7<each U, each V>(us: repeat each U, vs: repeat each V, more_us: repeat each U, more_vs: repeat each V) {
  print("break here")
}
 
f7(us: a, vs: 1, b, more_us: a, more_vs: 2, b)
