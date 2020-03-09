import Foundation
 
func main() {
  let x = Decimal(42.5)
  let y = Decimal(422.5)
  let z = Decimal(-23.6)
  let patatino = Decimal(12345567888.1234)
  return //%self.expect("frame var -d run -- x", substrs=['42.5'])
         //%self.expect("expr -d run -- x", substrs=['42.5'])
         //%self.expect("frame var -d run -- y", substrs=['422.5'])
         //%self.expect("expr -d run -- y", substrs=['422.5'])
         //%self.expect("frame var -d run -- z", substrs=['-23.6'])
         //%self.expect("expr -d run -- z", substrs=['-23.6'])
         //%self.expect("frame var -d run -- patatino", substrs=['12345567888.123'])
         //%self.expect("expr -d run -- patatino", substrs=['12345567888.123'])
}

main()
