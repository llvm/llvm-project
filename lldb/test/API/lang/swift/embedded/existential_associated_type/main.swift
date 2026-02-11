protocol P {
  func printme()
}

protocol Q {
  associatedtype Printable : P

  func getPrintable() -> Printable
}

struct Small : P {
  var i: Int

  func printme() {
    print("Small: \(i)")
  }
}

struct SmallContainer : Q {
  let s: Small

  init(i: Int) {
    self.s = Small(i: i)
  }

  func getPrintable() -> Small {
    return s
  }
}

func main() {
  let a: [any Q] = [ SmallContainer(i: 1), SmallContainer(i: 2) ]

  for x0 in a {
    let x = x0.getPrintable()
    x.printme() // break here
  }
}

main()
