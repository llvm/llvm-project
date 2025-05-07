import Foundation

protocol MyProtocol {
    func foo() -> String
}

extension MyProtocol {
  func foo() -> String {
    print("break here")
    return "\(self)"
  }
}

extension NSAttributedString: MyProtocol {}
let attributed = NSAttributedString(string: "attributed", attributes: [:]).foo()
