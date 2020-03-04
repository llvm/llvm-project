import Foundation

func f<T>(_ x : T) -> T {
  return x //%self.expect('frame var -d run-target -- foo', substrs=['(NSError)', 'domain: \"patatino\"',
           //%                                                       'code: 0', '0 key/value pairs'])
           //%self.expect('expr -d run-target -- foo', substrs=['(NSError)', 'domain: \"patatino\"',
           //%                                                  'code: 0', '0 key/value pairs'])
}

let foo = NSError(domain: "patatino", code: 0, userInfo: [:]) //%self.expect('expr -d run-target -- foo', substrs=['NSError'])
print(f(foo))
