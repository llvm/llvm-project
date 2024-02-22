class C {
  init() {}
  convenience init(unused: Bool) { 
    self.init()
    print(1)//%self.expect('po self', substrs=['<C: 0x'])
  }
}

C(unused: true)

