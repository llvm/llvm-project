import Foundation

public class C {}

@objc
public class Resilient : NSObject {
  private var priv: C { C() }

  public func f() {
    print("break here")
  }
}


Resilient().f()
