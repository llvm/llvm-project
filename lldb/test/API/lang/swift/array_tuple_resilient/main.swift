// Make sure we print array of tuples containing elements with
// resilient types print correctly.

import Foundation

var patatino : [(Data, Int64)] = [(Data([1, 2, 3]), 1001)]
var tinky : [(Data, Data)] = [(Data([1, 2, 3]), Data([9]))]
print(patatino) //%self.expect('frame variable -d run -- patatino',
                //%             substrs=['0 = 3 bytes', '1 = 1001'])
                //%self.expect('expr -d run -- patatino',
                //%             substrs=['0 = 3 bytes', '1 = 1001'])

print(tinky)    //%self.expect('frame variable -d run -- tinky',
                //%             substrs=['0 = 3 bytes', '1 = 1 byte'])
                //%self.expect('expr -d run -- tinky',
                //%             substrs=['0 = 3 bytes', '1 = 1 byte'])
