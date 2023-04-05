protocol Tinky {}

struct Winky : Tinky {
  var x : Int
}

extension Patatino where T == Winky {
  var baciotto : Int {
    return 0
  }

  func f() {
    return //%self.expect('expression self.baciotto', substrs=["(Int) $R0 = 0"])
           //%self.expect('expression self', substrs=["a.Patatino<a.Winky>"])
  }
}

struct Patatino<T> where T : Tinky {
  let x : T
}

let pat = Patatino<Winky>(x: Winky(x: 23))
pat.f()
