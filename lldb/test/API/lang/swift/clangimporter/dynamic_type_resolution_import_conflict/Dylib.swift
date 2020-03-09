import Foundation

public protocol LibraryProtocol : class {}

public class Foo : NSObject {
  public init(_ input : LibraryProtocol) {
    // When evaluating "input" here, RemoteAST will try to get its
    // dynamic type.
    print("break here")
  }
}
