class Classic {
  var _prop = 30

  func fun() {
    print("check self")
  }

  func run() {
    { [weak self] in
      guard let self else { fatalError("cannot happen") }
      print("check idiomatic self")
      self.fun()
    }()
  }
}

func main() {
  var c = Classic()
  c.run()
}

main()
