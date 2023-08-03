import CoreFoundation
import Foundation // Needed for `as CFString` cast

func main() {
    let uuid = CFUUIDCreateFromString(nil, "68753A44-4D6F-1226-9C60-0050E4C00067" as CFString)
    print("break here")
}

main()
