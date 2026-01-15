import Foundation

func main() {
  let url = URL(string: "https://www.example.com/path?query#fragment")
  let relativeURL = URL(string: "relative", relativeTo: URL(string: "https://www.example.com/"))

  print("break here")
}

main()
