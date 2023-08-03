func main() -> Int {
  return 0 //%self.expect('expr -d run-target -- class Y { var x : Int; init(_ x : Int) { self.x = x }}; Y(12)', substrs=['(x = 12)'])
}

_ = main()
