private struct S {
  func s1() {
    class S2 {
      func s2() {
        func s2_2() {
          enum S3 {
            case theCase
            func s3() {
              struct S4<T> {
                let t: T
              }

              let s4 = S4<Int>(t: 839)
              let string = StaticString("Hello") // break here
              print(string) 
            }
          }
          S3.theCase.s3()
        }
        s2_2()
      }
    }
    S2().s2()
  }
}

S().s1()
