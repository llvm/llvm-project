import Foundation

func main() {
  let paris = TimeZone(identifier: "Europe/Paris")!

  var comps = DateComponents()
  comps.year = 2001
  comps.month = 1
  comps.day = 15
  comps.hour = 14
  comps.minute = 12
  comps.timeZone = paris

  let date = Calendar(identifier: .gregorian).date(from: comps)!
  let nsdate = date as NSDate

  print("break here")
}

main()
